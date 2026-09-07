import pandas as pd
import numpy as np
import os
import requests
import json
from datetime import datetime

from shock_fetcher import fetch_all_data
from shock_engine import calculate_shock_scores, gecmis_veriyi_yukle, GECMIS_DOSYA
from shock_learner import (
    update_realized_shock_returns,
    compute_dynamic_market_thresholds,
    calibrate_adaptive_weights,
    log_shock_signals,
    AI_STATE_FILE
)

def send_telegram_message(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('CHAT_ID')
    
    if not token or not chat_id:
        print("Telegram Token veya Chat ID bulunamadı.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        requests.post(url, json=payload, timeout=10)
    except Exception as e:
        print(f"Telegram hatası: {e}")

def format_shock_report(df_scored, thresholds, weights, ai_status):
    min_score = thresholds.get('min_score', 75.0)
    shocks = df_scored[df_scored['shock_score'] >= min_score].sort_values(by='shock_score', ascending=False)
    
    msg = f"⚡ <b>BIST GÜVEN SKORLU ŞOK & SWING LİSTESİ ({min_score:.1f}+)</b>\n"
    msg += f"🗓 <i>{datetime.now().strftime('%Y-%m-%d')} | Saat: 10:30 Seans Açılışı</i>\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"
    
    if shocks.empty:
        msg += f"ℹ️ <i>Bugün {min_score:.1f} puan ve üzeri güven kriterini karşılayan hisse bulunamadı.</i>"
        return msg

    for idx, row in shocks.iterrows():
        trend_icon = "EMA20 Üstü ✅" if row.get('is_above_trend') else "EMA20 Altı ⚠️"
        squeeze_icon = "Sıkışma Kırılımı 💎" if row.get('is_squeezed') else "Standart Mum"
        
        msg += f"🚀 <b>#{row['ticker']}</b> ── <b>{row['shock_score']:.1f} Puan</b> ({row['stars']})\n"
        msg += f"• <b>Fiyat:</b> {row['close']:.2f} TL | <b>Değişim:</b> %{row['change_%']:+.2f}\n"
        msg += f"• <b>Bölge:</b> <i>{row['entry_status']}</i>\n"
        msg += f"• <b>Teyit:</b> <i>{trend_icon} | {squeeze_icon}</i>\n"
        msg += f"💰 <b>KASA ÖNERİSİ:</b> <b>{row['allocation']}</b>\n\n"
        
    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"🎯 <i>Toplam {len(shocks)} adet yüksek güvenli hisse tespit edildi.</i>\n\n"
    msg += "🛑 <b>RİSK KURALI:</b> <i>Stop-Loss seviyesini 10:00 - 10:15 açılış barının en dibine (ORB Low) koyunuz!</i>"
    
    return msg

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] === 10:30 Coordinated Shock Scanner Başlıyor ===")
    
    df_current = fetch_all_data()
    if df_current.empty:
        print("Hata: Piyasa verisi alınamadı.")
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
        print("Puanlanmış veri boş döndü.")
        return

    log_shock_signals(df_scored)

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

    telegram_msg = format_shock_report(df_scored, dynamic_thresholds, dynamic_weights, ai_status)
    send_telegram_message(telegram_msg)
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 10:30 Raporu Telegram'a başarıyla iletildi.")

if __name__ == "__main__":
    main()
