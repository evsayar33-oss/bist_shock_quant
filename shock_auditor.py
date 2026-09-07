import pandas as pd
import numpy as np
import requests
import os
import json
from datetime import datetime
from shock_fetcher import get_bist_raw_data

AI_STATE_FILE = "shock_ai_state.json"

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

def recalibrate_ai_model(hits_count, traps_count, missed_analysis):
    """
    KAPALI DEVRE MATEMATİKSEL ÖĞRENME MOTORU:
    Günün hata matrisine göre ağırlıkları ve eşikleri güvenli sınırlar içinde optimize eder.
    """
    current_state = {
        "thresholds": {"th_vol": 1.5, "th_range": 1.5, "th_flow": 2.0, "th_lambda": 1.2, "min_score": 75.0},
        "weights": {"vol": 0.30, "range": 0.30, "flow": 0.25, "lambda": 0.15},
        "status": "AKTİF"
    }
    if os.path.exists(AI_STATE_FILE):
        try:
            with open(AI_STATE_FILE, 'r') as f:
                loaded = json.load(f)
                current_state.update(loaded)
                if 'min_score' not in current_state['thresholds']:
                    current_state['thresholds']['min_score'] = 75.0
        except Exception:
            pass

    th = current_state['thresholds']
    w = current_state['weights']

    old_score = th['min_score']
    old_flow_w = w['flow']
    old_vol_w = w['vol']

    # 1. TUZAK CEZASI (Boğa tuzağı çoksa alıcı baskısını (Flow) ödüllendir, hacim yanılsamasını cezalandır)
    if traps_count >= 2:
        w['flow'] = min(w['flow'] + 0.03, 0.40)
        w['vol'] = max(w['vol'] - 0.03, 0.18)
        th['th_flow'] = min(th['th_flow'] + 0.1, 2.5)
        th['min_score'] = min(th['min_score'] + 1.0, 80.0)

    # 2. KAÇAN FIRSAT DÜZELTMESİ (Kıl payı kaçanlar varsa eşiği güvenle esnet)
    threshold_victims = sum(1 for x in missed_analysis if "KIL PAYI" in x['type'])
    if threshold_victims >= 2 and traps_count <= 1:
        th['min_score'] = max(th['min_score'] - 1.5, 68.0)
        th['th_vol'] = max(th['th_vol'] - 0.1, 1.2)

    # 3. BAŞARI ÖDÜLÜ (İsabet yüksekse mevcut dengeyi pekiştir)
    if hits_count >= 5 and traps_count == 0:
        th['min_score'] = max(min(th['min_score'], 75.0), 72.0)

    # Ağırlıkları normalize et (Toplamı her zaman tam 1.0 olmalı)
    total_w = sum(w.values())
    for k in w:
        w[k] = round(w[k] / total_w, 3)

    win_rate = (hits_count / (hits_count + traps_count) * 100) if (hits_count + traps_count) > 0 else 0
    current_state['status'] = f"⚡ OTOMATİK KALİBRE EDİLDİ (Başarı: %{win_rate:.0f} | Hedef Skor: {th['min_score']:.1f})"

    # Yeni beyni kaydet
    with open(AI_STATE_FILE, 'w') as f:
        json.dump(current_state, f, indent=4)

    update_summary = (
        f"• <b>Hedef Skor Barajı:</b> {old_score:.1f} ➔ <b>{th['min_score']:.1f}</b>\n"
        f"• <b>Alıcı Akış (Flow) Ağırlığı:</b> %{old_flow_w*100:.0f} ➔ <b>%{w['flow']*100:.0f}</b>\n"
        f"• <b>Hacim Ağırlığı:</b> %{old_vol_w*100:.0f} ➔ <b>%{w['vol']*100:.0f}</b>\n"
    )
    return update_summary

def run_evening_audit():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Röntgen Teşhisi ve Otonom Öğrenme Başlıyor...")
    
    df_close = get_bist_raw_data()
    if df_close.empty or not os.path.exists("gecmis_veri.csv"):
        print("Gerekli veriler bulunamadı.")
        return

    df_gecmis = pd.read_csv("gecmis_veri.csv")
    df_gecmis['tarih'] = pd.to_datetime(df_gecmis['tarih'])
    bugun = pd.Timestamp.now().normalize()
    
    df_morning_all = df_gecmis[df_gecmis['tarih'] == bugun].copy()
    
    # 75+ skor alanlar
    morning_picks = df_morning_all[df_morning_all['shock_score'] >= 75.0]
    morning_tickers = morning_picks['ticker'].tolist()

    # Akşam kapanışta %6+ olanlar
    actual_runners = df_close[df_close['change_%'] >= 6.0].sort_values(by='change_%', ascending=False)
    actual_tickers = actual_runners['ticker'].tolist()

    hits = [t for t in morning_tickers if t in actual_tickers]

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
                'z_flow': m_row.get('z_flow', 0.0)
            })

    missed_tickers = [t for t in actual_tickers if t not in morning_tickers][:10]
    missed_analysis = []

    for t in missed_tickers:
        c_row = df_close[df_close['ticker'] == t].iloc[0]
        m_row = df_morning_all[df_morning_all['ticker'] == t]

        if m_row.empty:
            missed_analysis.append({
                'ticker': t,
                'type': 'TARAMA_DISI',
                'close_chg': c_row['change_%'],
                'detail': f"10:30 Hacim Barajı Dışı ({c_row['value_traded']/1e6:.1f}M TL)"
            })
        else:
            m_row = m_row.iloc[0]
            m_chg = m_row.get('change_%', 0.0)
            score = m_row.get('shock_score', 0.0)
            z_v = m_row.get('z_vol', 0.0)
            z_r = m_row.get('z_range', 0.0)
            z_f = m_row.get('z_flow', 0.0)
            p3m = m_row.get('perf_3m', 0.0)

            if m_chg < 2.5:
                diag_type = "ZAMANLAMA (Öğleden Sonra Hareketi)"
                detail = f"Sabah: %{m_chg:+.1f} ➔ Akşam: %{c_row['change_%']:+.1f} (10:30'da henüz hareketsizdi)"
            elif p3m < -25.0 and z_v < 2.5:
                diag_type = "DİP FİLTRESİ"
                detail = f"3 Aylık: %{p3m:.1f} (Sabah Hacim: +{z_v}σ, Dönüş teyidi yok)"
            elif score >= 60.0:
                diag_type = "KIL PAYI KAÇAN (Eşik Kurbanı)"
                detail = f"Skor: {score:.1f}/75 | Hacim: {z_v:+.1f}σ, Akış: {z_f:+.1f}σ"
            else:
                weakest = []
                if z_v < 1.5: weakest.append(f"Hacim ({z_v:+.1f}σ)")
                if z_f < 1.5: weakest.append(f"Akış ({z_f:+.1f}σ)")
                diag_type = "SİNYAL YETERSİZ"
                detail = f"Skor: {score:.1f} | Zayıf: {', '.join(weakest) if weakest else 'Senkronize Değil'}"

            missed_analysis.append({
                'ticker': t,
                'type': diag_type,
                'close_chg': c_row['change_%'],
                'detail': detail
            })

    # OTONOM KALİBRASYONU ÇALIŞTIR VE STATE'İ GÜNCELLE
    ai_tuning_report = recalibrate_ai_model(len(hits), len(traps), missed_analysis)

    # RAPOR OLUŞTUR
    rep = "🔬 <b>BIST QUANT RÖNTGEN & ÖĞRENME RAPORU</b>\n"
    rep += f"🗓 <i>{datetime.now().strftime('%Y-%m-%d')} | Seans Kapanışı</i>\n"
    rep += "━━━━━━━━━━━━━━━━━━━━\n\n"

    rep += f"🎯 <b>10:30 İSABETLERİ ({len(hits)} Hisse Tavana Taşındı):</b>\n"
    for t in hits:
        c_chg = df_close[df_close['ticker'] == t]['change_%'].values[0]
        rep += f"• <b>#{t}</b> ➔ Kapanış: <b>%{c_chg:+.2f}</b>\n"
    rep += "\n"

    rep += f"🪤 <b>BOĞA TUZAKLARI ({len(traps)} Adet):</b>\n"
    if traps:
        for trap in traps:
            rep += f"• <b>#{trap['ticker']}</b>: Kapanış %{trap['close_chg']:+.2f} (Sabah: %{trap['morning_chg']:+.1f}) | Skor: {trap['score']:.1f}\n"
    else:
        rep += "<i>Bugün sahte şok yaşanmadı.</i>\n"
    rep += "\n"

    rep += f"🚨 <b>KAÇIRILAN ŞOKLARIN SEBEPLERİ:</b>\n"
    for item in missed_analysis:
        rep += f"• <b>#{item['ticker']}</b> (%{item['close_chg']:+.1f}) ➔ <b>[{item['type']}]</b>\n"
        rep += f"  ↳ <i>{item['detail']}</i>\n"
    rep += "\n━━━━━━━━━━━━━━━━━━━━\n"

    rep += "🤖 <b>OTONOM SİSTEM GÜNCELLEMESİ (Self-Tuned):</b>\n"
    rep += "<i>Bugünkü sonuçlara göre model parametreleri yeniden kalibre edildi:</i>\n"
    rep += ai_tuning_report
    rep += "✅ <i>Yeni katsayılar shock_ai_state.json dosyasına işlendi.</i>"

    send_telegram_audit(rep)
    print("Otonom öğrenme tamamlandı ve rapor iletildi.")

if __name__ == "__main__":
    run_evening_audit()
