import pandas as pd
import numpy as np
import os
import requests
from datetime import datetime

from shock_fetcher import fetch_all_data
from shock_engine import calculate_shock_scores, gecmis_veriyi_yukle, GECMIS_DOSYA

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

def format_shock_report(df_scored):
    shocks = df_scored[df_scored['shock_score'] >= 50.0].head(10)
    
    msg = f"⚡ <b>BIST SENKRONİZE ŞOK PATLAMA RAPORU</b>\n"
    msg += f"🗓 <i>Tarih: {datetime.now().strftime('%Y-%m-%d %H:%M')}</i>\n"
    msg += "<i>(Hacim, Mum Menzili ve Likidite Boşluğunun 1 Aylık Ortalamayı Deldiği Hisseler)</i>\n\n"
    
    if shocks.empty:
        msg += "⚠️ <i>Bugün çok boyutlu senkronize bir şok patlaması tespit edilemedi.</i>"
        return msg

    msg += "🚀 <b>DAY-1 SENKRONİZE ŞOK LİDERLERİ (Top 10)</b>\n"
    for idx, row in shocks.iterrows():
        fark = f"(+{row['score_diff']:.1f})" if row.get('score_diff', 0) > 0 else f"({row.get('score_diff', 0):.1f})"
        msg += f"• <b>{row['ticker']}</b> : Şok Skoru: <b>{row['shock_score']:.1f}</b> {fark} | Fiyat: {row['close']} TL (+%{row['change_%']:.1f})\n"
        msg += f"  └ <i>Hacim Z: +{row['z_vol']:.1f}σ | Menzil: +{row['z_range']:.1f}σ | {row['regime']}</i>\n"
        
    return msg

def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] === Coordinated Shock Scanner Başlıyor ===")
    
    df_current = fetch_all_data()
    if df_current.empty:
        print("Hata: Piyasa verisi alınamadı.")
        return

    df_gecmis = gecmis_veriyi_yukle()
    df_scored = calculate_shock_scores(df_current, df_gecmis)
    
    if df_scored.empty:
        print("Puanlanmış veri boş döndü.")
        return

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
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Başarılı! {GECMIS_DOSYA} güncellendi.")

    telegram_msg = format_shock_report(df_scored)
    send_telegram_message(telegram_msg)

if __name__ == "__main__":
    main()
