import numpy as np
import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

GECMIS_DOSYA = "gecmis_veri.csv"

def gecmis_veriyi_yukle():
    if os.path.exists(GECMIS_DOSYA):
        try:
            df = pd.read_csv(GECMIS_DOSYA)
            if 'tarih' in df.columns:
                df['tarih'] = pd.to_datetime(df['tarih'])
            return df
        except: 
            return pd.DataFrame()
    return pd.DataFrame()

def calculate_shock_scores(df, df_gecmis):
    if df.empty: 
        return df

    scored_data = []

    for idx, row in df.iterrows():
        item = row.to_dict()
        
        close = float(item.get('close', 0.0))
        open_p = float(item.get('open', close))
        high = float(item.get('high', close))
        low = float(item.get('low', close))
        change = float(item.get('change_%', 0.0))
        value_traded = float(item.get('value_traded', 0.0))
        rvol = float(item.get('rvol', 1.0))
        atr = float(item.get('atr', 1.0))
        perf_1m = float(item.get('perf_1m', 0.0))
        perf_3m = float(item.get('perf_3m', 0.0))

        # =========================================================================
        # 1. ÇOK BOYUTLU ŞOK VEKTÖRLERİ (1 AYLIK ORTALAMADAN SAPMALAR)
        # =========================================================================
        
        # A. HACİM ŞOKU (Z_Vol): 1 Aylık Ortalamadan Sapma
        z_vol = float((rvol - 1.0) * 2.2)
        z_vol = round(min(max(z_vol, -2.0), 6.0), 2)

        # B. MENZİL ŞOKU (Z_Range): Günlük Bar Boyunun 14 Günlük ATR'ye Oranı
        today_range = high - low
        safe_atr = max(atr, 0.01)
        range_expansion_ratio = today_range / safe_atr
        z_range = float((range_expansion_ratio - 1.0) * 2.5)
        z_range = round(min(max(z_range, -2.0), 6.0), 2)

        # C. LİKİDİTE BOŞLUĞU ŞOKU (Kyle's Lambda): Birim Paraya Düşen Fiyat Hareketi
        raw_lambda = (abs(change) / ((value_traded / 10000000.0) + 1e-9)) if value_traded > 0 else 0.0
        z_lambda = round(min(float(np.log1p(raw_lambda) * 2.0), 5.0), 2)

        # D. ALICI BASKISI & EMİR AKIŞ ŞOKU (CLV & Directional Efficiency)
        if today_range > 0:
            clv = ((close - low) - (high - close)) / today_range
            body_eff = (close - open_p) / today_range
        else:
            clv = 0.0
            body_eff = 0.0
        aggressor_flow = (max(clv, 0.0) * 0.55) + (max(body_eff, 0.0) * 0.45)
        z_flow = round(float(aggressor_flow * 4.0), 2)

        # =========================================================================
        # 2. SENKRONİZASYON (TÜM GÖSTERGELER AYNI ANDA PATLADI MI?)
        # =========================================================================
        # Kaç tane gösterge aynı anda +1.5 standart sapmanın üzerine çıktı?
        shock_count = 0
        if z_vol >= 1.5: shock_count += 1
        if z_range >= 1.5: shock_count += 1
        if z_lambda >= 1.2: shock_count += 1
        if z_flow >= 2.0: shock_count += 1

        # Senkronizasyon Çarpanı (Birlikte patlayanlara devasa ödül)
        concordance_multiplier = 1.0 + (shock_count * 0.25) # 4'ü de patlarsa 2.0x çarpan

        # =========================================================================
        # 3. ŞOK ÖNCESİ DİNLENME KONTROLÜ (DAY-1 IGNITION FILTER)
        # =========================================================================
        # Hisse son 1 ayda zaten %35 gitmişse, bu günkü şok bir tükeniştir (Climax).
        # Ama son 1 ayda sakin kalmışsa (%0 - %20), bu gerçek bir 1. GÜN ŞOKUDUR!
        is_fresh_shock = (perf_1m <= 22.0) and (perf_3m >= -10.0)
        is_downtrend_knife = (perf_3m < -25.0) # Düşen bıçak kalkanı

        item['z_vol'] = z_vol
        item['z_range'] = z_range
        item['z_lambda'] = z_lambda
        item['z_flow'] = z_flow
        item['shock_count'] = shock_count
        item['concordance_mult'] = concordance_multiplier
        item['is_fresh_shock'] = is_fresh_shock
        item['is_downtrend_knife'] = is_downtrend_knife
        scored_data.append(item)

    res_df = pd.DataFrame(scored_data)
    if res_df.empty: 
        return res_df

    # Yüzdelik Dilim Normalizasyonu
    res_df['pct_vol'] = res_df['z_vol'].rank(pct=True) * 100.0
    res_df['pct_range'] = res_df['z_range'].rank(pct=True) * 100.0
    res_df['pct_lambda'] = res_df['z_lambda'].rank(pct=True) * 100.0
    res_df['pct_flow'] = res_df['z_flow'].rank(pct=True) * 100.0

    # BİLEŞİK SENKRONİZE ŞOK SKORU (0 - 100)
    raw_score = (
        res_df['pct_vol'] * 0.30 +
        res_df['pct_range'] * 0.30 +
        res_df['pct_flow'] * 0.25 +
        res_df['pct_lambda'] * 0.15
    ) * (res_df['concordance_mult'] / 1.5)

    raw_score = np.clip(np.round(raw_score, 1), 0.0, 99.5)

    # Düşen Bıçakları ve Negatif Fiyatları Sıfırla
    res_df['shock_score'] = np.where(
        (res_df['change_%'] > 0) & (~res_df['is_downtrend_knife']),
        raw_score,
        0.0
    )

    # Rejim Tespiti
    conditions = [
        res_df['is_downtrend_knife'],
        (res_df['shock_score'] >= 75.0) & (res_df['shock_count'] >= 3) & (res_df['is_fresh_shock']),
        (res_df['shock_score'] >= 55.0) & (res_df['shock_count'] >= 2),
        (res_df['change_%'] < -2.0) & (res_df['z_vol'] >= 1.5)
    ]
    choices = [
        "🪤 DÜŞEN BIÇAK TUZAĞI",
        "⚡ SENKRONİZE ŞOK PATLAMASI (DAY-1)",
        "🚀 KISMİ HACİM & MENZİL İVMESİ",
        "🚨 KURUMSAL BOŞALTIM (DUMP)"
    ]
    res_df['regime'] = np.select(conditions, choices, default="NÖTR REJİM")

    drop_cols = ['pct_vol', 'pct_range', 'pct_lambda', 'pct_flow', 'concordance_mult', 'is_downtrend_knife']
    res_df = res_df.drop(columns=[col for col in drop_cols if col in res_df.columns])

    # Düne Göre Skor Farkı
    res_df['score_diff'] = 0.0
    if not df_gecmis.empty and 'shock_score' in df_gecmis.columns:
        son_tarih = df_gecmis['tarih'].max()
        df_son = df_gecmis[df_gecmis['tarih'] == son_tarih]
        eski_map = dict(zip(df_son['ticker'], df_son['shock_score']))
        res_df['score_diff'] = np.round(res_df['shock_score'] - res_df['ticker'].map(eski_map).fillna(res_df['shock_score']), 1)

    return res_df.sort_values(by='shock_score', ascending=False).reset_index(drop=True)
