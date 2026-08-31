import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="BIST Synchronous Shock Terminal", layout="wide", page_icon="⚡")

st.title("⚡ BIST Senkronize Şok & Faz Değişim Terminali")
st.markdown("*Son 1 aylık normalinden sapan; Hacim, Mum Menzili (ATR) ve Likidite Boşluğunun **aynı gün eşzamanlı patladığı (Day-1)** hisseler.*")

def load_data():
    if os.path.exists("gecmis_veri.csv"):
        try:
            df = pd.read_csv("gecmis_veri.csv")
            if 'tarih' in df.columns:
                df['tarih'] = pd.to_datetime(df['tarih'])
            return df
        except Exception as e:
            st.error(f"Dosya okuma hatası: {e}")
            return pd.DataFrame()
    return pd.DataFrame()

df_gecmis = load_data()

if not df_gecmis.empty:
    son_tarih = df_gecmis['tarih'].max()
    df = df_gecmis[df_gecmis['tarih'] == son_tarih].copy()
    
    st.caption(f"🗓️ Son Tarama: **{son_tarih.strftime('%Y-%m-%d')}** | 📊 Taranan Hisse: **{len(df)}**")

    required_cols = ['shock_score', 'score_diff', 'z_vol', 'z_range', 'z_lambda', 'z_flow', 'value_traded', 'change_%', 'shock_count']
    for c in required_cols:
        if c not in df.columns:
            df[c] = 0.0
        else:
            df[c] = pd.to_numeric(df[c], errors='coerce').fillna(0).round(2)

    if 'regime' not in df.columns:
        df['regime'] = 'NÖTR'

    # SIDEBAR
    st.sidebar.header("🔍 Hisse Şok Analizi")
    search_ticker = st.sidebar.text_input("Hisse Kodu (Örn: THYAO):").upper()
    
    if search_ticker:
        h_data = df[df['ticker'] == search_ticker]
        if not h_data.empty:
            score = float(h_data['shock_score'].iloc[0])
            diff = float(h_data['score_diff'].iloc[0])
            regime = h_data['regime'].iloc[0]
            zv = float(h_data['z_vol'].iloc[0])
            zr = float(h_data['z_range'].iloc[0])
            zl = float(h_data['z_lambda'].iloc[0])
            sc = int(h_data['shock_count'].iloc[0])
            
            st.sidebar.metric(f"{search_ticker} Şok Skoru", f"{score:.1f}", f"{diff:+.1f}")
            st.sidebar.write(f"**Durum:** {regime}")
            st.sidebar.write(f"**Eşzamanlı Şok Sayısı:** {sc}/4 Gösterge")
            st.sidebar.write(f"**Hacim Sapması:** {zv:+.2f}σ")
            st.sidebar.write(f"**Menzil (ATR) Sapması:** {zr:+.2f}σ")
            st.sidebar.write(f"**Likidite Boşluğu:** {zl:+.2f}σ")
            
            st.sidebar.write("📈 Son 30 Günlük Şok Trendi:")
            trend = df_gecmis[df_gecmis['ticker'] == search_ticker][['tarih', 'shock_score']].sort_values('tarih')
            if not trend.empty:
                trend.set_index('tarih', inplace=True)
                st.sidebar.line_chart(trend['shock_score'])
        else:
            st.sidebar.warning("Hisse bulunamadı.")

    # 1. ANA TABLO: SENKRONİZE ŞOK LİDERLERİ
    st.subheader("🚀 Eşzamanlı Şok Patlama Liderleri (Top 20)")
    st.markdown("*Hacim, Mum Boyu ve Alıcı Akışının aynı anda 1 aylık normalini katladığı taze patlamalar.*")
    
    top_candidates = df[df['shock_score'] > 0.0].sort_values(by='shock_score', ascending=False).head(20)
    
    display_cols = ['ticker', 'shock_score', 'score_diff', 'regime', 'shock_count', 'z_vol', 'z_range', 'z_lambda', 'change_%', 'close']
    display_cols = [c for c in display_cols if c in df.columns]
    
    col_names = {
        'ticker': 'Hisse',
        'shock_score': 'Şok Skoru',
        'score_diff': 'İvme Farkı',
        'regime': 'Şok Rejimi',
        'shock_count': 'Senkron Gösterge',
        'z_vol': 'Hacim Şoku (Z)',
        'z_range': 'Menzil/ATR Şoku (Z)',
        'z_lambda': 'Likidite Şoku (Z)',
        'change_%': 'Günlük %',
        'close': 'Fiyat (TL)'
    }
    
    if not top_candidates.empty:
        st.dataframe(
            top_candidates[display_cols].rename(columns=col_names),
            column_config={
                "Şok Skoru": st.column_config.ProgressColumn("Şok Skoru", min_value=0, max_value=100, format="%.1f"),
                "Senkron Gösterge": st.column_config.NumberColumn("Senkron Gösterge", format="%d/4"),
                "Hacim Şoku (Z)": st.column_config.NumberColumn("Hacim Şoku (Z)", format="%+.2fσ"),
                "Menzil/ATR Şoku (Z)": st.column_config.NumberColumn("Menzil/ATR Şoku (Z)", format="%+.2fσ"),
                "Likidite Şoku (Z)": st.column_config.NumberColumn("Likidite Şoku (Z)", format="%+.2fσ"),
                "Günlük %": st.column_config.NumberColumn("Günlük %", format="%+0.2f%%"),
                "Fiyat (TL)": st.column_config.NumberColumn("Fiyat (TL)", format="%.2f TL"),
                "İvme Farkı": st.column_config.NumberColumn("İvme Farkı", format="%+0.1f")
            },
            use_container_width=True,
            hide_index=True
        )
    else:
        st.info("ℹ️ Bugün kriterleri karşılayan eşzamanlı bir şok patlaması bulunamadı.")

    st.divider()

    # 2. DİSKALİFİYE EDİLENLER: DÜŞEN BIÇAK TUZAKLARI
    st.subheader("🪤 Düşen Bıçak Tuzakları (Uzak Dur)")
    st.markdown("*Ağır düşüş trendinde tepki veren sahte şoklar.*")
    
    traps = df[df['regime'].str.contains('DÜŞEN BIÇAK|DUMP', na=False)].sort_values(by='change_%', ascending=True).head(15)
    if not traps.empty:
        st.dataframe(
            traps[display_cols].rename(columns=col_names),
            column_config={
                "Şok Skoru": st.column_config.NumberColumn("Şok Skoru", format="%.1f"),
                "Senkron Gösterge": st.column_config.NumberColumn("Senkron Gösterge", format="%d/4"),
                "Hacim Şoku (Z)": st.column_config.NumberColumn("Hacim Şoku (Z)", format="%+.2fσ"),
                "Menzil/ATR Şoku (Z)": st.column_config.NumberColumn("Menzil/ATR Şoku (Z)", format="%+.2fσ"),
                "Likidite Şoku (Z)": st.column_config.NumberColumn("Likidite Şoku (Z)", format="%+.2fσ"),
                "Günlük %": st.column_config.NumberColumn("Günlük %", format="%+0.2f%%"),
                "Fiyat (TL)": st.column_config.NumberColumn("Fiyat (TL)", format="%.2f TL"),
                "İvme Farkı": st.column_config.NumberColumn("İvme Farkı", format="%+0.1f")
            },
            use_container_width=True,
            hide_index=True
        )

else:
    st.info("🕒 Sistem başlatılıyor... Lütfen GitHub Actions üzerinden 'Run workflow' yapınız.")
