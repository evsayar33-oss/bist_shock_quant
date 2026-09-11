import pandas as pd
import numpy as np
import os
import requests
import json
from datetime import datetime
import yfinance as yf

from shock_fetcher import fetch_all_data
from shock_engine import calculate_shock_scores, gecmis_veriyi_yukle, GECMIS_DOSYA
from shock_learner import (
    update_realized_shock_returns,
    compute_dynamic_market_thresholds,
    calibrate_adaptive_weights,
    log_shock_signals,
    AI_STATE_FILE
)

LEDGER_FILE = "backtest_ledger.csv"

def send_telegram_message(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('CHAT_ID')
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    max_len = 3800
    messages = []
    if len(message) > max_len:
        parts = message.split("\n\n")
        current_msg = ""
        for p in parts:
            if len(current_msg) + len(p) + 2 < max_len:
                current_msg += p + "\n\n"
            else:
                messages.append(current_msg.strip())
                current_msg = p + "\n\n"
        if current_msg:
            messages.append(current_msg.strip())
    else:
        messages = [message]

    for idx, msg in enumerate(messages):
        payload = {"chat_id": chat_id, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
        try:
            res = requests.post(url, json=payload, timeout=15)
            if not res.json().get("ok"):
                payload.pop("parse_mode")
                requests.post(url, json=payload, timeout=15)
        except Exception as e:
            print(f"Telegram hatası: {e}")

def check_bist_earnings_risk(ticker):
    """BIST BİLANÇO KALKANI: Önümüzdeki 5 gün içinde bilanço varsa True döner."""
    try:
        t = yf.Ticker(f"{ticker}.IS")
        cal = t.calendar
        if cal is None:
            return False, ""
        
        ed = None
        if isinstance(cal, dict):
            ed = cal.get('Earnings Date')
        elif hasattr(cal, 'get'):
            ed = cal.get('Earnings Date')
        elif hasattr(cal, 'loc') and 'Earnings Date' in cal.index:
            ed = cal.loc['Earnings Date'].values

        if ed is not None:
            if not isinstance(ed, (list, np.ndarray)):
                ed = [ed]
            for d in ed:
                if pd.notna(d):
                    d_date = d.date() if hasattr(d, 'date') else pd.to_datetime(d).date()
                    days_diff = (d_date - datetime.now().date()).days
                    if 0 <= days_diff <= 5:
                        return True, f"{d_date.strftime('%d.%m')} ({days_diff} Gün Kaldı)"
    except Exception:
        pass
    return False, ""

def generate_exit_signals(df_current):
    if not os.path.exists(LEDGER_FILE):
        return ""
    try:
        df_ledger = pd.read_csv(LEDGER_FILE)
    except:
        return ""

    if df_ledger.empty or 'is_completed' not in df_ledger.columns:
        return ""

    open_positions = df_ledger[df_ledger['is_completed'] == 0]
    if open_positions.empty:
        return ""

    close_map = dict(zip(df_current['ticker'], df_current['close']))
    bugun = datetime.now().date()
    signals = []

    for idx, row in open_positions.iterrows():
        ticker = row['ticker']
        if ticker not in close_map:
            continue

        curr_p = close_map[ticker]
        entry_p = float(row['entry_price'])
        row_date = datetime.strptime(str(row['date']), '%Y-%m-%d').date()
        days_held = (bugun - row_date).days
        pnl = ((curr_p - entry_p) / entry_p) * 100.0

        if pnl <= -2.5:
            signals.append(f"🚨 <b>#{ticker} STOP-LOSS (ACİL ÇIKIŞ)!</b>\n  ↳ <i>Giriş: {entry_p:.2f} TL | Güncel: {curr_p:.2f} TL | Zarar: <b>%{pnl:+.2f}</b>\n  🛑 Stop kırıldı, zararı kes ve çık!</i>")
        elif pnl >= 9.0:
            signals.append(f"💰 <b>#{ticker} TAVAN KİLİTLEDİ / KÂR AL!</b>\n  ↳ <i>Giriş: {entry_p:.2f} TL | Güncel: {curr_p:.2f} TL | Kâr: <b>%{pnl:+.2f}</b>\n  🎯 Pozisyonun %50'sini sat, kalanın stopunu maliyete ({entry_p:.2f} TL) çek!</i>")
        elif days_held >= 5:
            signals.append(f"⏰ <b>#{ticker} 1 HAFTALIK VADE DOLDU</b>\n  ↳ <i>Kapanış: {curr_p:.2f} TL | Net: <b>%{pnl:+.2f}</b> (Vade bitti, nakde geç)</i>")
        else:
            signals.append(f"🟢 <b>#{ticker} TAŞIMAYA DEVAM ET</b> ({days_held}. Gün)\n  ↳ <i>Fiyat: {curr_p:.2f} TL | Durum: <b>%{pnl:+.2f}</b> (Trend Güçlü)</i>")

    if not signals:
        return ""

    exit_rep = "🛡️ <b>BIST AÇIK POZİSYONLAR & ÇIKIŞ ALARMLARI (Exit Engine):</b>\n"
    exit_rep += "━━━━━━━━━━━━━━━━━━━━\n"
    exit_rep += "\n\n".join(signals)
    exit_rep += "\n━━━━━━━━━━━━━━━━━━━━\n\n"
    return exit_rep

def record_clean_ledger_entries(df_scored):
    bugun_str = datetime.now().strftime('%Y-%m-%d')
    new_rows = []
    candidates = df_scored[df_scored['shock_score'] >= 65.0]
    for _, r in candidates.iterrows():
        if "BİLANÇO" in r.get('entry_status', ''):
            continue

        new_rows.append({
            "date": bugun_str,
            "ticker": r['ticker'],
            "entry_price": r['close'],
            "z_vol": r.get('z_vol', 0.0),
            "z_range": r.get('z_range', 0.0),
            "z_flow": r.get('z_flow', 0.0),
            "z_lambda": r.get('z_lambda', 0.0),
            "entry_status": r.get('entry_status', 'NORMAL'),
            "initial_score": r.get('shock_score', 0.0),
            "price_d1": np.nan,
            "price_d3": np.nan,
            "price_d5": np.nan,
            "return_d1": np.nan,
            "return_d3": np.nan,
            "return_d5": np.nan,
            "is_completed": 0
        })

    if not new_rows:
        return

    df_new = pd.DataFrame(new_rows)
    if os.path.exists(LEDGER_FILE):
        try:
            df_old = pd.read_csv(LEDGER_FILE)
            df_old = df_old[~((df_old['date'] == bugun_str) & (df_old['ticker'].isin(df_new['ticker'])))]
            df_final = pd.concat([df_old, df_new], ignore_index=True)
        except:
            df_final = df_new
    else:
        df_final = df_new

    df_final.to_csv(LEDGER_FILE, index=False)

def format_shock_report(df_scored, exit_signals_text, min_score=75.0):
    shocks = df_scored[df_scored['shock_score'] >= min_score].sort_values(by='shock_score', ascending=False)
    
    msg = ""
    if exit_signals_text:
        msg += exit_signals_text

    msg += f"⚡ <b>BIST GÜVEN SKORLU ŞOK LİSTESİ ({min_score:.1f}+)</b>\n"
    msg += f"🗓 <i>{datetime.now().strftime('%Y-%m-%d')} | Saat: 10:30 Seans Açılışı</i>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    
    if shocks.empty:
        msg += f"ℹ️ <i>Bugün {min_score:.1f} puan ve üzeri güven kriterini karşılayan hisse bulunamadı.</i>"
        return msg

    for idx, row in shocks.head(15).iterrows():
        msg += f"🚀 <b>#{row['ticker']}</b> ── <b>{row['shock_score']:.1f} Puan</b> ({row['stars']})\n"
        msg += f"• <b>Fiyat:</b> {row['close']:.2f} TL | <b>Değişim:</b> %{row['change_%']:+.2f}\n"
        msg += f"• <b>Giriş Marjı:</b> <i>{row['entry_status']}</i>\n"
        tp_p = row['close'] * 1.09
        sl_p = row['close'] * 0.975
        msg += f"• 🎯 <b>Kâr Al (+%9.0):</b> {tp_p:.2f} TL | 🛑 <b>Stop (-%2.5):</b> {sl_p:.2f} TL (5 Gün Vade)\n"
        msg += f"💰 <b>KELLY PAYI:</b> <b>{row['allocation']}</b>\n\n"
        
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"🎯 <i>Toplam {len(shocks)} adet yüksek güvenli hisse tespit edildi.</i>"
    return msg

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] === BIST Scanner & Triple Shield Başlıyor ===")
    
    df_current = fetch_all_data()
    if df_current.empty:
        print("Hata: BIST verisi temin edilemedi.")
        return

    update_realized_shock_returns(df_current)
    df_temp = calculate_shock_scores(df_current, pd.DataFrame())
    
    dynamic_thresholds = compute_dynamic_market_thresholds(df_temp)
    dynamic_weights, ai_status = calibrate_adaptive_weights()

    saved_min_score = 75.0
    if os.path.exists(AI_STATE_FILE):
        try:
            with open(AI_STATE_FILE, 'r') as f:
                saved_state = json.load(f)
                saved_min_score = saved_state.get('thresholds', {}).get('min_score', 75.0)
                if 'weights' in saved_state:
                    dynamic_weights = saved_state['weights']
        except Exception:
            pass

    dynamic_thresholds['min_score'] = saved_min_score

    with open(AI_STATE_FILE, 'w') as f:
        json.dump({"thresholds": dynamic_thresholds, "weights": dynamic_weights, "status": ai_status}, f, indent=4)

    df_gecmis = gecmis_veriyi_yukle()
    df_scored = calculate_shock_scores(df_current, df_gecmis, dynamic_thresholds, dynamic_weights)
    
    if df_scored.empty:
        return

    # 1. BİLANÇO KALKANI KONTROLÜ
    print("🔍 BIST Bilanço takvimi taranıyor...")
    for idx, row in df_scored.head(15).iterrows():
        if row['shock_score'] >= 65.0:
            has_earnings, e_date = check_bist_earnings_risk(row['ticker'])
            if has_earnings:
                print(f"⚠️ {row['ticker']} için BIST bilanço riski: {e_date}")
                df_scored.at[idx, 'entry_status'] = f"🚨 BİLANÇO RİSKİ ({e_date})"
                df_scored.at[idx, 'allocation'] = "İşlem Açma (%0 - Bilanço Kumarı)"
                df_scored.at[idx, 'stars'] = "⚠️"
                df_scored.at[idx, 'shock_score'] = saved_min_score - 1.0

    # 2. Çıkış ve Stop Sinyallerini Hesapla
    exit_signals_text = generate_exit_signals(df_current)

    # 3. Temiz Deftere Kaydet
    record_clean_ledger_entries(df_scored)

    if not df_gecmis.empty:
        bugun = pd.Timestamp.now().normalize()
        df_gecmis = df_gecmis[df_gecmis['tarih'] != bugun]
        df_yeni_gecmis = pd.concat([df_gecmis, df_scored], ignore_index=True)
    else:
        df_yeni_gecmis = df_scored

    df_yeni_gecmis['tarih'] = pd.to_datetime(df_yeni_gecmis['tarih'])
    limit_tarih = pd.Timestamp.now().normalize() - pd.Timedelta(days=30)
    df_yeni_gecmis = df_yeni_gecmis[df_yeni_gecmis['tarih'] >= limit_tarih]
    df_yeni_gecmis.to_csv(GECMIS_DOSYA, index=False)

    # 4. Raporu Gönder
    telegram_msg = format_shock_report(df_scored, exit_signals_text, saved_min_score)
    send_telegram_message(telegram_msg)
    print("BIST Güvenlik Kalkanlı Rapor başarıyla tamamlandı.")

if __name__ == "__main__":
    main()
