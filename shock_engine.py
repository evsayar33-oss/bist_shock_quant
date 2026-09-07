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

def calculate_shock_scores(df, df_gecmis, dynamic_thresholds=None, dynamic_weights=None):
    if df.empty: 
        return df

    if dynamic_thresholds is None:
        dynamic_thresholds = {"th_vol": 1.5, "th_range": 1.5, "th_flow": 2.0, "th_lambda": 1.2}
    if dynamic_weights is None:
        dynamic_weights = {"vol": 0.30, "range": 0.30, "flow": 0.25, "lambda": 0.15}

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
        vwap = float(item.get('vwap', 0.0))
        ema20 = float(item.get('ema20', 0.0))
        sma50 = float(item.get('sma50', 0.0))
        bb_upper = float(item.get('bb_upper', 0.0))
        bb_lower = float(item.get('bb_lower', 0.0))
        bb_basis = float(item.get('bb_basis', close))

        # 1. VWAP KONTROLÜ
        is_below_vwap = False
        if vwap > 0 and close < (vwap * 0.990):
            is_below_vwap = True

        # 2. Z-SKORLAR
        z_vol = round(min(max(float((rvol - 1.0) * 2.2), -2.0), 6.0), 2)
        today_range = high - low
        safe_atr = max(atr, 0.01)
        z_range = round(min(max(float(((today_range / safe_atr) - 1.0) * 2.5), -2.0), 6.0), 2)

        liquidity_damping = min(value_traded / 12000000.0, 1.0) if value_traded > 0 else 0.0
        raw_lambda = ((abs(change) / ((value_traded / 10000000.0) + 1e-9)) * liquidity_damping) if value_traded > 0 else 0.0
        z_lambda = round(min(float(np.log1p(raw_lambda) * 2.0), 5.0), 2)

        if today_range > 0:
            clv = ((close - low) - (high - close)) / today_range
            body_eff = (close - open_p) / today_range
        else:
            clv = 0.0
            body_eff = 0.0
        aggressor_flow = (max(clv, 0.0) * 0.55) + (max(body_eff, 0.0) * 0.45)
        z_flow = round(float(aggressor_flow * 4.0), 2)

        # 3. YENİ BOYUTLAR: TREND TABANI & BOLLINGER SIKIŞMASI
        is_above_trend = (close >= ema20) and (close >= sma50 if sma50 > 0 else True)
        
        safe_basis = max(bb_basis, 0.01)
        bb_width = ((bb_upper - bb_lower) / safe_basis) * 100.0 if (bb_upper > bb_lower) else 20.0
        is_squeezed = (bb_width <= 14.0) and (bb_width > 0.0) # Enerjisi sıkışmış, yeni patlayan tahta

        # 4. GİRİŞ MARJI (SWEET SPOT) DEĞERLENDİRMESİ
        entry_bonus = 0.0
        if 2.0 <= change <= 5.2:
            entry_bonus = 6.0
            entry_status = "🎯 İDEAL GİRİŞ BÖLGESİ"
        elif change >= 7.5:
            entry_bonus = -8.0
            entry_status = "⚠️ GEÇ KALINDI (Tepeden Alım Riski)"
        else:
            entry_status = "NORMAL GİRİŞ"

        # Senkronizasyon Kontrolü
        shock_count = 0
        if z_vol >= dynamic_thresholds.get('th_vol', 1.5): shock_count += 1
        if z_range >= dynamic_thresholds.get('th_range', 1.5): shock_count += 1
        if z_lambda >= dynamic_thresholds.get('th_lambda', 1.2): shock_count += 1
        if z_flow >= dynamic_thresholds.get('th_flow', 2.0): shock_count += 1

        concordance_multiplier = 1.0 + (shock_count * 0.25)
        is_fresh_shock = (perf_1m <= 22.0) and (perf_3m >= -10.0)
        is_downtrend_knife = (perf_3m < -25.0) and (z_vol < 2.5)
        is_bottom_reversal = (perf_3m < -25.0) and (z_vol >= 2.5) and (z_flow >= 1.8)

        item['z_vol'] = z_vol
        item['z_range'] = z_range
        item['z_lambda'] = z_lambda
        item['z_flow'] = z_flow
        item['shock_count'] = shock_count
        item['concordance_mult'] = concordance_multiplier
        item['is_above_trend'] = is_above_trend
        item['is_squeezed'] = is_squeezed
        item['entry_bonus'] = entry_bonus
        item['entry_status'] = entry_status
        item['is_fresh_shock'] = is_fresh_shock
        item['is_downtrend_knife'] = is_downtrend_knife
        item['is_bottom_reversal'] = is_bottom_reversal
        item['is_below_vwap'] = is_below_vwap
        scored_data.append(item)

    res_df = pd.DataFrame(scored_data)
    if res_df.empty: 
        return res_df

    res_df['pct_vol'] = res_df['z_vol'].rank(pct=True) * 100.0
    res_df['pct_range'] = res_df['z_range'].rank(pct=True) * 100.0
    res_df['pct_lambda'] = res_df['z_lambda'].rank(pct=True) * 100.0
    res_df['pct_flow'] = res_df['z_flow'].rank(pct=True) * 100.0

    w_v = dynamic_weights.get('vol', 0.30)
    w_r = dynamic_weights.get('range', 0.30)
    w_f = dynamic_weights.get('flow', 0.25)
    w_l = dynamic_weights.get('lambda', 0.15)

    base_score = (
        res_df['pct_vol'] * w_v +
        res_df['pct_range'] * w_r +
        res_df['pct_flow'] * w_f +
        res_df['pct_lambda'] * w_l
    ) * (res_df['concordance_mult'] / 1.5)

    # NİHAİ GÜVEN SKORU: Baz Puan + Trend Bonusu + Sıkışma Bonusu + Giriş Marjı Bonusu
    trend_boost = np.where(res_df['is_above_trend'], 6.0, -5.0)
    squeeze_boost = np.where(res_df['is_squeezed'], 7.0, 0.0)
    
    raw_confidence = base_score + trend_boost + squeeze_boost + res_df['entry_bonus']
    final_score = np.clip(np.round(raw_confidence, 1), 0.0, 99.5)

    res_df['shock_score'] = np.where(
        (res_df['change_%'] > 0) & (~res_df['is_downtrend_knife']) & (~res_df['is_below_vwap']),
        final_score,
        0.0
    )
    res_df['confidence_score'] = res_df['shock_score']

    # KASA DAĞILIMI (POSITION SIZING) & YILDIZ BELİRLEME
    def assign_allocation(row):
        score = row['shock_score']
        chg = row['change_%']
        if score >= 85.0 and chg <= 6.5:
            return "⭐⭐⭐⭐⭐", "Portföyün %15 - %20'si (Yüksek Güven)"
        elif score >= 75.0:
            return "⭐⭐⭐⭐", "Portföyün %8 - %12'si (Dengeli Güven)"
        elif score >= 65.0:
            return "⭐⭐⭐", "Portföyün %3 - %5'i (Deneme / Küçük Kasa)"
        else:
            return "⭐", "İşlem Açma (Yetersiz Güven)"

    stars_alloc = [assign_allocation(r) for _, r in res_df.iterrows()]
    res_df['stars'] = [sa[0] for sa in stars_alloc]
    res_df['allocation'] = [sa[1] for sa in stars_alloc]

    # Rejim Tespiti
    conditions = [
        res_df['is_below_vwap'],
        res_df['is_downtrend_knife'],
        res_df['is_bottom_reversal'],
        (res_df['shock_score'] >= 75.0) & (res_df['is_squeezed']),
        (res_df['shock_score'] >= 75.0) & (res_df['shock_count'] >= 3),
        (res_df['shock_score'] >= 55.0)
    ]
    choices = [
        "🚨 VWAP ALTI (SABAH TUZAĞI)",
        "🪤 DÜŞEN BIÇAK TUZAĞI",
        "⚡ DİPTEN ŞOK DÖNÜŞÜ (REVERSAL)",
        "💎 SIKIŞMADAN HAFTALIK SWING KOPUŞU",
        "⚡ SENKRONİZE ŞOK PATLAMASI (DAY-1)",
        "🚀 KISMİ HACİM & MENZİL İVMESİ"
    ]
    res_df['regime'] = np.select(conditions, choices, default="NÖTR REJİM")

    drop_cols = ['pct_vol', 'pct_range', 'pct_lambda', 'pct_flow', 'concordance_mult', 'is_downtrend_knife', 'is_bottom_reversal', 'is_below_vwap', 'entry_bonus']
    res_df = res_df.drop(columns=[col for col in drop_cols if col in res_df.columns])

    res_df['score_diff'] = 0.0
    if not df_gecmis.empty and 'shock_score' in df_gecmis.columns:
        son_tarih = df_gecmis['tarih'].max()
        df_son = df_gecmis[df_gecmis['tarih'] == son_tarih]
        eski_map = dict(zip(df_son['ticker'], df_son['shock_score']))
        res_df['score_diff'] = np.round(res_df['shock_score'] - res_df['ticker'].map(eski_map).fillna(res_df['shock_score']), 1)

    return res_df.sort_values(by='shock_score', ascending=False).reset_index(drop=True)
