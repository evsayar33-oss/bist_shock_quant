import pandas as pd
import numpy as np
import os
import json
from scipy.stats import spearmanr

SIGNAL_LOG_FILE = "shock_signals_log.csv"
AI_STATE_FILE = "shock_ai_state.json"

DEFAULT_WEIGHTS = {"vol": 0.30, "range": 0.30, "flow": 0.25, "lambda": 0.15}

def load_signal_history():
    if os.path.exists(SIGNAL_LOG_FILE):
        try:
            df = pd.read_csv(SIGNAL_LOG_FILE)
            df['tarih'] = pd.to_datetime(df['tarih'])
            return df
        except:
            return pd.DataFrame()
    return pd.DataFrame()

def log_shock_signals(top_df):
    if top_df.empty: return
    signals = top_df.head(10)[['ticker', 'close', 'shock_score', 'z_vol', 'z_range', 'z_flow', 'z_lambda', 'tarih']].copy()
    signals['realized_3d'] = np.nan
    signals['realized_5d'] = np.nan
    
    history_df = load_signal_history()
    if not history_df.empty:
        today_val = pd.Timestamp.now().normalize()
        history_df = history_df[history_df['tarih'] != today_val]
        updated = pd.concat([history_df, signals], ignore_index=True)
    else:
        updated = signals
    updated.to_csv(SIGNAL_LOG_FILE, index=False)

def update_realized_shock_returns(current_market_df):
    history_df = load_signal_history()
    if history_df.empty or current_market_df.empty:
        return history_df
    
    price_map = dict(zip(current_market_df['ticker'], current_market_df['close']))
    today = pd.Timestamp.now().normalize()
    
    for idx, row in history_df.iterrows():
        sig_date = pd.to_datetime(row['tarih'])
        days_passed = (today - sig_date).days
        ticker = row['ticker']
        entry_price = float(row['close'])
        
        if ticker in price_map and entry_price > 0:
            current_price = price_map[ticker]
            gain = ((current_price - entry_price) / entry_price) * 100.0
            
            if days_passed >= 3 and pd.isna(row['realized_3d']):
                history_df.at[idx, 'realized_3d'] = round(gain, 2)
            if days_passed >= 5 and pd.isna(row['realized_5d']):
                history_df.at[idx, 'realized_5d'] = round(gain, 2)
                
    history_df.to_csv(SIGNAL_LOG_FILE, index=False)
    return history_df

def compute_dynamic_market_thresholds(df):
    """SABİT EŞİKLERİ YOK EDEN FONKSİYON: O günkü piyasanın en uç %10'luk dilimini otomatik eşik yapar."""
    if df.empty:
        return {"th_vol": 1.5, "th_range": 1.5, "th_flow": 2.0, "th_lambda": 1.2}
    
    # 85. Persentil: O günkü piyasaya göre dinamik şok barajı
    th_vol = float(np.percentile(df['z_vol'], 85)) if 'z_vol' in df.columns else 1.5
    th_range = float(np.percentile(df['z_range'], 85)) if 'z_range' in df.columns else 1.5
    th_flow = float(np.percentile(df['z_flow'], 85)) if 'z_flow' in df.columns else 2.0
    th_lambda = float(np.percentile(df['z_lambda'], 85)) if 'z_lambda' in df.columns else 1.2
    
    return {
        "th_vol": round(max(th_vol, 0.8), 2),
        "th_range": round(max(th_range, 0.8), 2),
        "th_flow": round(max(th_flow, 1.0), 2),
        "th_lambda": round(max(th_lambda, 0.5), 2)
    }

def calibrate_adaptive_weights():
    """ÖZ-ÖĞRENME: Geçmiş şokların getirisinden ağırlıkları optimize eder."""
    history_df = load_signal_history()
    valid = history_df.dropna(subset=['realized_3d']) if not history_df.empty else pd.DataFrame()
    
    if len(valid) < 15:
        return DEFAULT_WEIGHTS, "🕒 ÖĞRENME EVRESİNDE (Örneklem Bekleniyor)"
    
    factors = ['z_vol', 'z_range', 'z_flow', 'z_lambda']
    ic_scores = {}
    y = valid['realized_3d'].values
    
    for f in factors:
        x = valid[f].values
        if np.std(x) > 0 and np.std(y) > 0:
            corr, _ = spearmanr(x, y)
            ic_scores[f] = max(corr if not np.isnan(corr) else 0.05, 0.05)
        else:
            ic_scores[f] = 0.05
            
    total = sum(ic_scores.values())
    raw_weights = {
        "vol": ic_scores['z_vol'] / total,
        "range": ic_scores['z_range'] / total,
        "flow": ic_scores['z_flow'] / total,
        "lambda": ic_scores['z_lambda'] / total
    }
    
    # Bayesian Shrinkage (Aşırı dalgalanmayı engeller)
    final_w = {k: round(0.50 * DEFAULT_WEIGHTS[k] + 0.50 * raw_weights[k], 2) for k in DEFAULT_WEIGHTS}
    w_sum = sum(final_w.values())
    final_w = {k: round(v / w_sum, 2) for k, v in final_w.items()}
    
    status = f"🧠 AI ÖZ-ÖĞRENME AKTİF (Eğitilen Sinyal: {len(valid)})"
    return final_w, status
