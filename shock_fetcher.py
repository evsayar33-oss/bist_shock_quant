import requests
import pandas as pd
import numpy as np
from datetime import datetime

def get_bist_raw_data():
    """TradingView üzerinden mikroyapı, hacim, menzil ve VWAP verilerini çeker."""
    url = "https://scanner.tradingview.com/turkey/scan"
    payload = {
        "filter": [
            {"left": "type", "operation": "equal", "right": "stock"},
            {"left": "Value.Traded", "operation": "greater", "right": 8000000}
        ],
        "columns": [
            "name", "close", "open", "high", "low", "volume", "change", "Value.Traded",
            "relative_volume_10d_calc",
            "average_true_range_14",
            "Perf.1M",
            "Perf.3M",
            "Volatility.D",
            "VWAP" # KURUMSAL VWAP VERİSİ EKLENDİ
        ],
        "sort": {"sortBy": "Value.Traded", "sortOrder": "desc"},
        "range": [0, 300]
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "application/json",
        "Referer": "https://www.tradingview.com/"
    }
    
    try:
        response = requests.post(url, json=payload, headers=headers, timeout=15)
        data = response.json()
        rows = []
        for item in data.get("data", []):
            d = item["d"]
            rows.append({
                "ticker": d[0],
                "close": float(d[1]) if d[1] is not None else 0.0,
                "open": float(d[2]) if d[2] is not None else 0.0,
                "high": float(d[3]) if d[3] is not None else 0.0,
                "low": float(d[4]) if d[4] is not None else 0.0,
                "volume": float(d[5]) if d[5] is not None else 0.0,
                "change_%": float(d[6]) if d[6] is not None else 0.0,
                "value_traded": float(d[7]) if d[7] is not None else 0.0,
                "rvol": float(d[8]) if len(d) > 8 and d[8] is not None else 1.0,
                "atr": float(d[9]) if len(d) > 9 and d[9] is not None else 1.0,
                "perf_1m": float(d[10]) if len(d) > 10 and d[10] is not None else 0.0,
                "perf_3m": float(d[11]) if len(d) > 11 and d[11] is not None else 0.0,
                "volatility": float(d[12]) if len(d) > 12 and d[12] is not None else 2.0,
                "vwap": float(d[13]) if len(d) > 13 and d[13] is not None else 0.0
            })
        return pd.DataFrame(rows)
    except Exception as e:
        print(f"Piyasa Verisi Hatası: {e}")
        return pd.DataFrame()

def fetch_all_data():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Çok boyutlu şok ve VWAP verileri toplanıyor...")
    df = get_bist_raw_data()
    if df.empty:
        return df
    df['tarih'] = pd.Timestamp.now().normalize()
    return df
