import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from shock_fetcher import get_bist_raw_data

def send_telegram_audit(message):
    token = os.environ.get('TELEGRAM_TOKEN')
    chat_id = os.environ.get('CHAT_ID')
    
    if not token or not chat_id:
        print("❌ HATA: TELEGRAM_TOKEN veya CHAT_ID bulunamadı! GitHub Secrets kontrol edilmeli.")
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True
    }
    try:
        res = requests.post(url, json=payload, timeout=15)
        res_data = res.json()
        if res_data.get("ok"):
            print("✅ Telegram mesajı başarıyla teslim edildi!")
        else:
            print(f"❌ Telegram API Hatası: {res_data.get('description')}")
            # HTML etiket hatası olursa düz metin olarak tekrar dene
            payload.pop("parse_mode")
            requests.post(url, json=payload, timeout=15)
            print("ℹ️ Mesaj düz metin formatında iletildi.")
    except Exception as e:
        print(f"❌ Bağlantı hatası: {e}")

def run_evening_audit():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Gün Sonu Teşhis & Hata Analizi Başlatılıyor...")
    
    df_close = get_bist_raw_data()
    if df_close.empty:
        print("❌ Kapanış verisi alınamadı.")
        return

    if not os.path.exists("gecmis_veri.csv"):
        print("❌ gecmis_veri.csv bulunamadı.")
        return
        
    df_gecmis = pd.read_csv("gecmis_veri.csv")
    df_gecmis['tarih'] = pd.to_datetime(df_gecmis['tarih'])
    bugun = pd.Timestamp.now().normalize()
    
    # Bugün 75+ skor alanlar
    morning_picks = df_gecmis[(df_gecmis['tarih'] == bugun) & (df_gecmis['shock_score'] >= 75.0)]
    morning_tickers = morning_picks['ticker'].tolist()

    # Günün tavan/şok yapanları (%6 ve üzeri)
    actual_runners = df_close[df_close['change_%'] >= 6.0].sort_values(by='change_%', ascending=False)
    actual_tickers = actual_runners['ticker'].tolist()

    hits = [t for t in morning_tickers if t in actual_tickers]
    
    traps = []
    for t in morning_tickers:
        close_row = df_close[df_close['ticker'] == t]
        if not close_row.empty:
            chg = close_row['change_%'].values[0]
            if chg < 2.0:
                traps.append((t, chg))

    missed = [t for t in actual_tickers if t not in morning_tickers][:10]

    # HTML formatına uygun rapor oluştur ( < işaretleri temizlendi )
    report = "📋 <b>GÜN SONU SİSTEM DENETİM VE TEŞHİS RAPORU</b>\n"
    report += f"🗓 <i>{datetime.now().strftime('%Y-%m-%d')} | Piyasa Kapanışı</i>\n"
    report += "━━━━━━━━━━━━━━━━━━━━\n\n"

    report += "✅ <b>İSABETLİ TAHMİNLER:</b>\n"
    if hits:
        for t in hits:
            chg = df_close[df_close['ticker'] == t]['change_%'].values[0]
            report += f"• <b>#{t}</b>: Kapanış %{chg:+.2f}\n"
    else:
        report += "<i>Bugün sabah önerilip ralliye devam eden hisse yok.</i>\n"
    report += "\n"

    report += "🪤 <b>BOĞA TUZAKLARI (Sabah Önerilip Sönenler):</b>\n"
    if traps:
        for t, chg in traps:
            report += f"• <b>#{t}</b>: Kapanış %{chg:+.2f} <i>(İncele: Hacim tuzağı mı?)</i>\n"
    else:
        report += "<i>Tuzak sinyal oluşmadı.</i>\n"
    report += "\n"

    report += "🚨 <b>KAÇIRILAN ŞOK YÜKSELİŞLER (Neden Kaçtı?):</b>\n"
    if missed:
        for t in missed:
            row = df_close[df_close['ticker'] == t].iloc[0]
            reasons = []
            if row['value_traded'] < 3500000:
                reasons.append("Hacim Yetersiz (3.5M Altı)")
            if row['perf_3m'] < -25.0:
                reasons.append("Dip Hissesi (3 Aylık Düşüş)")
            if row['close'] < (row.get('vwap', 0) * 0.99):
                reasons.append("VWAP Altında")
            if not reasons:
                reasons.append("Düşük Senkron/Puan")
            
            report += f"• <b>#{t}</b> (%{row['change_%']:+.1f}) ➡️ <i>Neden: {', '.join(reasons)}</i>\n"
    else:
        report += "<i>Kaçırılan büyük hisse yok.</i>\n"

    report += "\n━━━━━━━━━━━━━━━━━━━━\n"
    report += "💡 <b>ALGORİTMA GELİŞTİRME TAVSİYESİ:</b>\n"
    if len(missed) > len(hits):
        report += "⚠️ <i>Bugün kaçan hisse sayısı yüksek. Parametreleri ve hacim eşiklerini gözden geçirin.</i>"
    else:
        report += "🎯 <i>Algoritma filtreleri bugünkü piyasa rejimiyle dengeli çalıştı.</i>"

    send_telegram_audit(report)

if __name__ == "__main__":
    run_evening_audit()
