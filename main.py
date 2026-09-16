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

    for msg in messages:
        payload = {"chat_id": chat_id, "text": msg, "parse_mode": "HTML", "disable_web_page_preview": True}
        try:
            res = requests.post(url, json=payload, timeout=15)
            if not res.json().get("ok"):
                payload.pop("parse_mode")
                requests.post(url, json=payload, timeout=15)
        except Exception as e:
            print(f"Telegram hatası: {e}")

def assess_bist_market_regime(df_market):
    """
    BIST PİYASA GENELİ VE ÇÖKÜŞ REJİMİ DENETÇİSİ:
    Tüm hisselerin yüzde kaçı kırmızı ve lokomotifler ne durumda?
    """
    if df_market.empty:
        return False, "NÖTR"

    total_stocks = len(df_market)
    red_stocks = len(df_market[df_market['change_%'] < 0])
    red_ratio = (red_stocks / total_stocks) if total_stocks > 0 else 0.5
    avg_market_change = df_market['change_%'].mean()

    # Eğer taranan BIST hisselerinin %70'inden fazlası düşüyorsa veya piyasa ortalaması -%1.5 altındaysa:
    if red_ratio >= 0.70 or avg_market_change <= -1.5:
        regime_status = f"🚨 PİYASA ÇÖKÜŞ REJİMİ (Düşen: %{red_ratio*100:.0f} | Ort: %{avg_market_change:+.2f})"
        return True, regime_status
    elif red_ratio <= 0.40 and avg_market_change >= 0.8:
        regime_status = f"🚀 BOĞA REJİMİ (Yükselen Hisseler Hakim | Ort: %{avg_market_change:+.2f})"
        return False, regime_status
    else:
        regime_status = f"⚖️ NÖTR / DALGALI PİYASA (Kırmızı: %{red_ratio*100:.0f} | Ort: %{avg_market_change:+.2f})"
        return False, regime_status

def check_bist_earnings_risk(ticker):
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

        # Kademeli Kar Al ve Stop-Loss Mekanizması
        if pnl <= -3.0:
            signals.append(f"🚨 <b>#{ticker} STOP-LOSS (ACİL ÇIKIŞ)!</b>\n  ↳ <i>Giriş: {entry_p:.2f} TL | Güncel: {curr_p:.2f} TL | Zarar: <b>%{pnl:+.2f}</b>\n  🛑 Stop kırıldı, zararı kes ve çık!</i>")
        elif pnl >= 9.0:
            signals.append(f"💰 <b>#{ticker} TAVAN KİLİTLEDİ / KÂR AL!</b>\n  ↳ <i>Giriş: {entry_p:.2f} TL | Güncel: {curr_p:.2f} TL | Kâr: <b>%{pnl:+.2f}</b>\n  🎯 Pozisyonun %50'sini sat, kalanın stopunu maliyete ({entry_p:.2f} TL) çek!</i>")
        elif pnl >= 4.0:
            signals.append(f"🔒 <b>#{ticker} KÂR KORUMA / MALİYET STOPU</b>\n  ↳ <i>Fiyat: {curr_p:.2f} TL | Kâr: <b>%{pnl:+.2f}</b>\n  🛡️ Stop maliyete ({entry_p:.2f} TL) çekildi. Riski sıfırla!</i>")
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

def record_clean_ledger_entries(df_scored, market_is_crashing=False):
    """
    TEMİZ DEFTER GİRİŞİ:
    Artık tavan kovalayanlar, hacimsizler veya çöküş günündeki zayıflar deftere ASLA yazılmaz!
    """
    bugun_str = datetime.now().strftime('%Y-%m-%d')
    new_rows = []
    
    # Giriş barajı: Normal günlerde 75+, çöküş günlerinde sadece 85+ ve RVOL >= 2.5
    min_entry_score = 85.0 if market_is_crashing else 75.0
    candidates = df_scored[df_scored['shock_score'] >= min_entry_score]

    for _, r in candidates.iterrows():
        entry_status = str(r.get('entry_status', ''))
        rvol_val = float(r.get('rvol', 1.0))
        val_traded = float(r.get('value_traded', 0.0))

        # KESİN GİRİŞ ENGELLERİ:
        # 1. Bilanço riski varsa girme
        if "BİLANÇO" in entry_status: continue
        # 2. Tavan kovalamaca veya geç kalınmışsa girme!
        if "GEÇ KALINDI" in entry_status or float(r.get('change_%', 0.0)) > 6.0: continue
        # 3. Hacim şoku gerçek değilse (RVOL < 1.3) girme!
        if rvol_val < 1.3: continue
        # 4. Sığ tahtaysa (<25M TL) girme!
        if val_traded < 25000000.0: continue

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

def format_shock_report(df_scored, exit_signals_text, market_regime_text, market_is_crashing=False, min_score=75.0):
    effective_min = 85.0 if market_is_crashing else min_score
    shocks = df_scored[df_scored['shock_score'] >= effective_min].sort_values(by='shock_score', ascending=False)
    
    msg = ""
    if exit_signals_text:
        msg += exit_signals_text

    msg += f"⚡ <b>BIST GÜVEN SKORLU ŞOK LİSTESİ ({effective_min:.1f}+)</b>\n"
    msg += f"🗓 <i>{datetime.now().strftime('%Y-%m-%d')} | Seans Raporu</i>\n"
    msg += f"🌐 <b>Piyasa Rejimi:</b> <i>{market_regime_text}</i>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    
    if market_is_crashing:
        msg += "⚠️ <b>DİKKAT: PİYASA GENELİNDE SERT SATIŞ BASKISI VAR!</b>\n"
        msg += "<i>Sistem Korumalı Moda geçti. Sadece endekse kafa tutan devasa hacimli hisselere izin verilir.</i>\n\n"

    if shocks.empty:
        msg += f"🛡️ <i>Bugün güven kriterini karşılayan veya endekse kafa tutabilen risksiz hisse bulunamadı. <b>Nakit kraldır (%100 Nakit Önerisi).</b></i>"
        return msg

    for idx, row in shocks.head(10).iterrows():
        msg += f"🚀 <b>#{row['ticker']}</b> ── <b>{row['shock_score']:.1f} Puan</b> ({row['stars']})\n"
        msg += f"• <b>Fiyat:</b> {row['close']:.2f} TL | <b>Değişim:</b> %{row['change_%']:+.2f}\n"
        msg += f"• <b>Giriş Bölgesi:</b> <i>{row['entry_status']}</i>\n"
        msg += f"• <b>Hacim Çarpanı:</b> <b>{row.get('rvol', 1.0):.2f}x</b> | <b>Akış Gücü:</b> <b>{row.get('z_flow', 0.0):+.2f}σ</b>\n"
        tp_p = row['close'] * 1.09
        sl_p = row['close'] * 0.970
        msg += f"• 🎯 <b>Hedef:</b> {tp_p:.2f} TL | 🛑 <b>Stop (-%3.0):</b> {sl_p:.2f} TL\n"
        msg += f"💰 <b>KASA:</b> <b>{row['allocation']}</b>\n\n"
        
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"🎯 <i>Toplam {len(shocks)} adet filtreyi geçen hisse tespit edildi.</i>"
    return msg

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] === BIST Scanner & Crash Shield Başlıyor ===")
    
    df_current = fetch_all_data()
    if df_current.empty:
        print("Hata: BIST verisi temin edilemedi.")
        return

    # 1. Piyasa Çöküş Rejimini Tespit Et
    market_is_crashing, regime_status_text = assess_bist_market_regime(df_current)
    print(f"Piyasa Durumu: {regime_status_text}")

    update_realized_shock_returns(df_current)
    df_temp = calculate_shock_scores(df_current, pd.DataFrame(), market_is_crashing=market_is_crashing)
    
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
    df_scored = calculate_shock_scores(df_current, df_gecmis, dynamic_thresholds, dynamic_weights, market_is_crashing=market_is_crashing)
    
    if df_scored.empty:
        return

    # 2. BİLANÇO KALKANI KONTROLÜ
    print("🔍 BIST Bilanço takvimi taranıyor...")
    for idx, row in df_scored.head(15).iterrows():
        if row['shock_score'] >= 65.0:
            has_earnings, e_date = check_bist_earnings_risk(row['ticker'])
            if has_earnings:
                print(f"⚠️ {row['ticker']} için BIST bilanço riski: {e_date}")
                df_scored.at[idx, 'entry_status'] = f"🚨 BİLANÇO RİSKİ ({e_date})"
                df_scored.at[idx, 'allocation'] = "İşlem Açma (%0 - Bilanço Kumarı)"
                df_scored.at[idx, 'stars'] = "⚠️"
                df_scored.at[idx, 'shock_score'] = saved_min_score - 10.0

    # 3. Çıkış ve Stop Sinyallerini Hesapla
    exit_signals_text = generate_exit_signals(df_current)

    # 4. Temiz Deftere Kaydet (Sadece kuralları geçenler!)
    record_clean_ledger_entries(df_scored, market_is_crashing=market_is_crashing)

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

    # 5. Raporu Gönder
    telegram_msg = format_shock_report(df_scored, exit_signals_text, regime_status_text, market_is_crashing=market_is_crashing, min_score=saved_min_score)
    send_telegram_message(telegram_msg)
    print("BIST Güvenlik Kalkanlı Rapor başarıyla tamamlandı.")

if __name__ == "__main__":
    main()
