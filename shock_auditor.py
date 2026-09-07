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
        print("Telegram bilgileri eksik.")
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
        if not res.json().get("ok"):
            payload.pop("parse_mode")
            requests.post(url, json=payload, timeout=15)
    except Exception as e:
        print(f"Hata: {e}")

def run_evening_audit():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Kapsamlı Röntgen Teşhisi Başlatılıyor...")
    
    df_close = get_bist_raw_data()
    if df_close.empty or not os.path.exists("gecmis_veri.csv"):
        print("Gerekli veriler bulunamadı.")
        return

    df_gecmis = pd.read_csv("gecmis_veri.csv")
    df_gecmis['tarih'] = pd.to_datetime(df_gecmis['tarih'])
    bugun = pd.Timestamp.now().normalize()
    
    # Bugün 10:30'da taranan TÜM hisseler
    df_morning_all = df_gecmis[df_gecmis['tarih'] == bugun].copy()
    
    # Sabah 75+ skor alanlar
    morning_picks = df_morning_all[df_morning_all['shock_score'] >= 75.0]
    morning_tickers = morning_picks['ticker'].tolist()

    # Akşam tavan/şok yapanlar (%6+)
    actual_runners = df_close[df_close['change_%'] >= 6.0].sort_values(by='change_%', ascending=False)
    actual_tickers = actual_runners['ticker'].tolist()

    # A) İsabetler
    hits = [t for t in morning_tickers if t in actual_tickers]

    # B) Tuzaklar (Sabah 75+ alıp akşama %3'ün altına sönenler)
    traps = []
    for t in morning_tickers:
        c_row = df_close[df_close['ticker'] == t]
        if not c_row.empty and c_row['change_%'].values[0] < 3.0:
            m_row = morning_picks[morning_picks['ticker'] == t].iloc[0]
            traps.append({
                'ticker': t,
                'close_chg': c_row['change_%'].values[0],
                'morning_chg': m_row.get('change_%', 0.0),
                'score': m_row.get('shock_score', 0.0),
                'z_vol': m_row.get('z_vol', 0.0),
                'z_flow': m_row.get('z_flow', 0.0)
            })

    # C) Kaçırılanlar (Akşam coşan ama sabah yakalanamayan ilk 10)
    missed_tickers = [t for t in actual_tickers if t not in morning_tickers][:10]
    missed_analysis = []

    for t in missed_tickers:
        c_row = df_close[df_close['ticker'] == t].iloc[0]
        m_row = df_morning_all[df_morning_all['ticker'] == t]

        if m_row.empty:
            # Sabah taramasına hiç girememiş
            missed_analysis.append({
                'ticker': t,
                'type': 'TARAMA_DISI',
                'close_chg': c_row['change_%'],
                'detail': f"10:30 Hacim Barajı Dışı (Hacim: {c_row['value_traded']/1e6:.1f}M TL)"
            })
        else:
            m_row = m_row.iloc[0]
            m_chg = m_row.get('change_%', 0.0)
            score = m_row.get('shock_score', 0.0)
            z_v = m_row.get('z_vol', 0.0)
            z_r = m_row.get('z_range', 0.0)
            z_f = m_row.get('z_flow', 0.0)
            p3m = m_row.get('perf_3m', 0.0)

            # Teşhis Kriterleri:
            if m_chg < 2.5:
                # Sabah hisse uyuyormuş, öğleden sonra patlamış
                diag_type = "ZAMANLAMA (Öğleden Sonra Hareketi)"
                detail = f"Sabah: %{m_chg:+.1f} ➔ Akşam: %{c_row['change_%']:+.1f} (Hareket 10:30'dan sonra başladı)"
            elif p3m < -25.0 and z_v < 2.5:
                diag_type = "DİP FİLTRESİNE TAKILDI"
                detail = f"3 Aylık Perf: %{p3m:.1f} (Sabah Hacim: +{z_v}σ, Dönüş için yetersiz görüldü)"
            elif score >= 60.0:
                diag_type = "KIL PAYI KAÇAN (Eşik Kurbanı)"
                detail = f"Skor: {score:.1f}/75 | Hacim: {z_v:+.1f}σ, Menzil: {z_r:+.1f}σ, Akış: {z_f:+.1f}σ"
            else:
                # Zayıf kalan indikatörü bul
                weakest = []
                if z_v < 1.5: weakest.append(f"Hacim Yetersiz ({z_v:+.1f}σ)")
                if z_f < 1.5: weakest.append(f"Alıcı Baskısı Zayıf ({z_f:+.1f}σ)")
                if z_r < 1.5: weakest.append(f"Menzil Dar ({z_r:+.1f}σ)")
                diag_type = "SİNYAL EKSİKLİĞİ"
                detail = f"Skor: {score:.1f} | Eksikler: {', '.join(weakest) if weakest else 'Senkronize Değil'}"

            missed_analysis.append({
                'ticker': t,
                'type': diag_type,
                'close_chg': c_row['change_%'],
                'detail': detail
            })

    # RAPOR METNİNİ OLUŞTUR
    rep = "🔬 <b>BIST DETAYLI QUANT RÖNTGEN RAPORU</b>\n"
    rep += f"🗓 <i>{datetime.now().strftime('%Y-%m-%d')} | Seans Kapanışı</i>\n"
    rep += "━━━━━━━━━━━━━━━━━━━━\n\n"

    # 1. Başarılar
    rep += f"🎯 <b>10:30 İSABETLERİ ({len(hits)} Hisse Tavana Taşındı):</b>\n"
    for t in hits:
        c_chg = df_close[df_close['ticker'] == t]['change_%'].values[0]
        rep += f"• <b>#{t}</b> ➔ Kapanış: <b>%{c_chg:+.2f}</b>\n"
    rep += "\n"

    # 2. Tuzaklar
    rep += f"🪤 <b>BOĞA TUZAKLARI (Neden Söndüler?):</b>\n"
    if traps:
        for trap in traps:
            rep += f"• <b>#{trap['ticker']}</b>: Kapanış %{trap['close_chg']:+.2f} (Sabah: %{trap['morning_chg']:+.1f})\n"
            rep += f"  ↳ <i>Teşhis: Skor {trap['score']:.1f} idi fakat Akış ({trap['z_flow']}σ) ve Hacim korunamadı; sabah fitili bıraktı.</i>\n"
    else:
        rep += "<i>Bugün sahte şok oluşmadı.</i>\n"
    rep += "\n"

    # 3. Kaçırılanlar (Derin Analiz)
    rep += f"🚨 <b>KAÇIRILAN ŞOKLARIN DERİN ANALİZİ:</b>\n"
    for item in missed_analysis:
        rep += f"• <b>#{item['ticker']}</b> (%{item['close_chg']:+.1f}) ➔ <b>[{item['type']}]</b>\n"
        rep += f"  ↳ <i>{item['detail']}</i>\n"
    rep += "\n━━━━━━━━━━━━━━━━━━━━\n"

    # 4. Somut Geliştirme Önerisi
    rep += "💡 <b>SİSTEM GELİŞTİRME TEŞHİSİ:</b>\n"
    timing_count = sum(1 for x in missed_analysis if "ZAMANLAMA" in x['type'])
    threshold_count = sum(1 for x in missed_analysis if "KIL PAYI" in x['type'])
    dip_count = sum(1 for x in missed_analysis if "DİP" in x['type'])

    if timing_count >= 3:
        rep += f"📌 <i>Kaçanların {timing_count} tanesi 10:30'da henüz harekete başlamamıştı. Bu hisseleri yakalamak için saat <b>14:00 Öğle Seansı Taraması</b> eklenmelidir.</i>\n"
    if threshold_count >= 2:
        rep += f"📌 <i>{threshold_count} hisse 60-74 skor bandında kıl payı takıldı. Skor eşiği 75'ten 70'e esnetilebilir.</i>\n"
    if dip_count >= 1:
        rep += f"📌 <i>{dip_count} hisse dip kuralına takıldı. Düşüşteki hisseler için hacim toleransı yumuşatılabilir.</i>\n"

    send_telegram_audit(rep)
    print("Detaylı röntgen raporu gönderildi.")

if __name__ == "__main__":
    run_evening_audit()
