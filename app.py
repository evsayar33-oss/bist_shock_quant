import streamlit as st
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime

st.set_page_config(
    page_title="BIST Quant Momentum & Exit Terminal",
    layout="wide",
    page_icon="⚡"
)

st.markdown("""
<style>
    .metric-card {
        background-color: #1E222D;
        border-radius: 10px;
        padding: 15px;
        border-left: 5px solid #2962FF;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

AI_STATE_FILE = "shock_ai_state.json"
GECMIS_DOSYA = "gecmis_veri.csv"
LEDGER_FILE = "backtest_ledger.csv"

def load_ai_state():
    if os.path.exists(AI_STATE_FILE):
        try:
            with open(AI_STATE_FILE, 'r') as f:
                return json.load(f)
        except:
            pass
    return {
        "thresholds": {"min_score": 75.0, "th_vol": 1.5, "th_flow": 2.0},
        "weights": {"vol": 0.35, "flow": 0.35, "range": 0.20, "lambda": 0.10},
        "status": "🛡️ PİYASA ÇÖKÜŞ KALKANLI & HACİM GÜVENLİ MODEL"
    }

def load_data():
    df_scan = pd.DataFrame()
    df_ledger = pd.DataFrame()
    if os.path.exists(GECMIS_DOSYA):
        try:
            df_scan = pd.read_csv(GECMIS_DOSYA)
            if 'tarih' in df_scan.columns:
                df_scan['tarih'] = pd.to_datetime(df_scan['tarih'])
        except:
            pass

    if os.path.exists(LEDGER_FILE):
        try:
            df_ledger = pd.read_csv(LEDGER_FILE)
        except:
            pass

    return df_scan, df_ledger

ai_state = load_ai_state()
df_scan, df_ledger = load_data()

th = ai_state.get('thresholds', {})
w = ai_state.get('weights', {})
min_score = th.get('min_score', 75.0)

st.title("⚡ BIST Quant Momentum & Çıkış Terminali")
st.caption(f"🤖 **Model Durumu:** {ai_state.get('status', 'AKTİF')} | 🎯 **Hedef Baraj:** {min_score:.1f} Puan")

# Piyasa genişliği ve kırmızı oranı kontrolü
market_breadth_text = "NÖTR"
if not df_scan.empty:
    son_tarih = df_scan['tarih'].max()
    today_scan = df_scan[df_scan['tarih'] == son_tarih]
    red_pct = (len(today_scan[today_scan['change_%'] < 0]) / len(today_scan) * 100.0) if len(today_scan) > 0 else 50.0
    if red_pct >= 70:
        market_breadth_text = f"🚨 ÇÖKÜŞ REJİMİ (Hisselerin %{red_pct:.0f}'i Düşüşte)"
    elif red_pct <= 40:
        market_breadth_text = f"🚀 BOĞA REJİMİ (Yükseliş Hakim)"
    else:
        market_breadth_text = f"⚖️ DALGALI REJİM (Kırmızı: %{red_pct:.0f})"

c1, c2, c3, c4 = st.columns(4)

completed_trades = df_ledger[df_ledger['is_completed'] == 1] if not df_ledger.empty and 'is_completed' in df_ledger.columns else pd.DataFrame()
win_rate = 0.0
if not completed_trades.empty and len(completed_trades) > 0:
    wins = completed_trades[completed_trades['return_d5'] > 0]
    win_rate = (len(wins) / len(completed_trades)) * 100.0

c1.metric("🎯 Dinamik Baraj", f"{min_score:.1f} Puan", f"Vol/Flow: %{int(w.get('vol', 0.35)*100)} / %{int(w.get('flow', 0.35)*100)}")
c2.metric("🏆 5G Win Rate", f"%{win_rate:.1f}", f"{len(completed_trades)} Tamamlanmış İşlem")
c3.metric("🌐 Piyasa Rejimi", market_breadth_text)
c4.metric("🛡️ Kalkan", "Hacim Vetosu Aktif", "RVOL < 1.25x Yasak")

st.divider()

# YAN MENÜ: HİSSE SORGULAMA
st.sidebar.header("🔍 BIST Hisse Röntgeni")
search_ticker = st.sidebar.text_input("Hisse Kodu (Örn: THYAO, KARSN):").upper().strip()

if search_ticker and not df_scan.empty:
    h_data = df_scan[df_scan['ticker'] == search_ticker]
    if not h_data.empty:
        last_row = h_data.sort_values(by='tarih', ascending=False).iloc[0]
        st.sidebar.subheader(f"#{search_ticker} Analizi")
        st.sidebar.metric("Quant Güven Skoru", f"{last_row['shock_score']:.1f}", last_row.get('stars', '⭐⭐⭐⭐'))
        st.sidebar.write(f"**Son Fiyat:** {last_row['close']:.2f} TL ({last_row['change_%']:+.2f}%)")
        st.sidebar.write(f"**Giriş Durumu:** {last_row.get('entry_status', 'NORMAL')}")
        st.sidebar.write(f"**Göreli Hacim (RVOL):** {last_row.get('rvol', 1.0):.2f}x")
        st.sidebar.write(f"**Alıcı Akış Baskısı:** {last_row.get('z_flow', 0.0):+.2f}σ")
        st.sidebar.info(f"💰 {last_row.get('allocation', 'Standart Risk')}")
    else:
        st.sidebar.warning("Hisse son tarama kayıtlarında bulunamadı.")

tab1, tab2, tab3 = st.tabs(["🚀 Günün Giriş Liderleri", "🛡️ Açık Pozisyonlar & Çıkışlar", "🧪 Canlı Backtest Defteri"])

with tab1:
    st.subheader("🎯 Bugünün Yüksek Güvenli BIST Şok Girişleri")
    if not df_scan.empty:
        son_tarih = df_scan['tarih'].max()
        df_today = df_scan[df_scan['tarih'] == son_tarih].copy()
        top_candidates = df_today[df_today['shock_score'] >= min_score].sort_values(by='shock_score', ascending=False)

        if not top_candidates.empty:
            disp_cols = ['ticker', 'shock_score', 'stars', 'close', 'change_%', 'rvol', 'entry_status', 'allocation']
            col_map = {
                'ticker': 'Hisse',
                'shock_score': 'Güven Skoru',
                'stars': 'Yıldız',
                'close': 'Fiyat (TL)',
                'change_%': 'Günlük %',
                'rvol': 'RVOL (Hacim Çarpanı)',
                'entry_status': 'Bölge',
                'allocation': 'Önerilen Kasa'
            }
            st.dataframe(
                top_candidates[disp_cols].rename(columns=col_map),
                column_config={
                    "Güven Skoru": st.column_config.ProgressColumn("Güven Skoru", min_value=0, max_value=100, format="%.1f"),
                    "Fiyat (TL)": st.column_config.NumberColumn("Fiyat (TL)", format="%.2f TL"),
                    "Günlük %": st.column_config.NumberColumn("Günlük %", format="%+0.2f%%"),
                    "RVOL (Hacim Çarpanı)": st.column_config.NumberColumn("Hacim Çarpanı", format="%.2fx"),
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.info(f"ℹ️ Bugün {min_score:.1f} puan barajını aşan bir şok hissesi tespit edilemedi.")
    else:
        st.info("Henüz tarama verisi bulunmuyor.")

with tab2:
    st.subheader("🛡️ Açık Pozisyonlar & Çıkış / Kâr Al Motoru")
    st.markdown("*Kademeli kâr koruma, maliyet stopu ve dinamik çıkış alarmları.*")

    if not df_ledger.empty and 'is_completed' in df_ledger.columns:
        open_pos = df_ledger[df_ledger['is_completed'] == 0].copy()
        
        if not open_pos.empty:
            if not df_scan.empty:
                son_tarih = df_scan['tarih'].max()
                current_map = dict(zip(df_scan[df_scan['tarih'] == son_tarih]['ticker'], df_scan[df_scan['tarih'] == son_tarih]['close']))
            else:
                current_map = {}

            pos_cards = []
            for idx, r in open_pos.iterrows():
                tk = r['ticker']
                curr_p = current_map.get(tk, r['entry_price'])
                entry_p = float(r['entry_price'])
                pnl = ((curr_p - entry_p) / entry_p) * 100.0

                action = "🟢 TAŞIMAYA DEVAM ET"
                if pnl <= -3.0:
                    action = "🚨 STOP-LOSS / ACİL ÇIKIŞ"
                elif pnl >= 9.0:
                    action = "💰 TAVAN KİLİTLEDİ (Yarısını Sat)"
                elif pnl >= 4.0:
                    action = "🔒 STOP MALİYETE ÇEKİLDİ (Risksiz)"

                pos_cards.append({
                    "Hisse": tk,
                    "Giriş Tarihi": r['date'],
                    "Giriş Fiyatı": f"{entry_p:.2f} TL",
                    "Güncel Fiyat": f"{curr_p:.2f} TL",
                    "Kâr/Zarar (%)": pnl,
                    "Aksiyon Sinyali": action
                })

            df_pos_show = pd.DataFrame(pos_cards)
            st.dataframe(
                df_pos_show,
                column_config={
                    "Kâr/Zarar (%)": st.column_config.NumberColumn("Kâr/Zarar (%)", format="%+0.2f%%")
                },
                use_container_width=True,
                hide_index=True
            )
        else:
            st.success("✅ Şu an takip edilen açık pozisyon yok.")
    else:
        st.info("Kayıt defterinde henüz açık işlem bulunmuyor.")

with tab3:
    st.subheader("🧪 Şeffaf BIST Backtest Defteri & Model Karnesi")
    if not df_ledger.empty:
        st.dataframe(
            df_ledger.sort_values(by='date', ascending=False),
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("Kayıt defteri henüz boş.")
