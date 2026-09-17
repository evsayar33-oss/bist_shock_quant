import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from shock_engine import (
    DEFAULT_META_WEIGHTS,
    DEFAULT_THRESHOLDS,
    DEFAULT_WEIGHTS,
    REGIME_META_TEMPLATES,
    classify_bist_regime,
)

SIGNAL_LOG_FILE = "shock_signals_log.csv"
AI_STATE_FILE = "shock_ai_state.json"
GECMIS_DOSYA = "gecmis_veri.csv"
LEDGER_FILE = "backtest_ledger.csv"

META_VERSION = 1
DEFAULT_RESILIENCE_WEIGHT = 0.70
REGIMES = tuple(REGIME_META_TEMPLATES.keys())
REGIME_MIN_SCORES = {
    "CRASH": 88.0,
    "STRESS": 83.0,
    "ROTATION": 77.0,
    "EXPANSION": 74.0,
    "QUIET": 72.0,
    "NORMAL": 75.0,
}


def _safe_float(value, default=0.0):
    try:
        value = float(value)
        return default if not np.isfinite(value) else value
    except Exception:
        return default


def _normalize_weights(weights, fallback):
    out = {k: max(_safe_float((weights or {}).get(k), fallback[k]), 0.0) for k in fallback}
    total = sum(out.values())
    if total <= 0:
        return dict(fallback)
    return {k: out[k] / total for k in out}


def load_ai_state():
    if not os.path.exists(AI_STATE_FILE):
        return {}
    try:
        with open(AI_STATE_FILE, "r", encoding="utf-8") as f:
            state = json.load(f)
        return state if isinstance(state, dict) else {}
    except Exception:
        return {}


def save_ai_state(state):
    with open(AI_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=4, ensure_ascii=False)


def _default_profile(regime="NORMAL"):
    regime = str(regime).upper()
    return {
        "weights": dict(REGIME_META_TEMPLATES.get(regime, DEFAULT_META_WEIGHTS)),
        "min_score": REGIME_MIN_SCORES.get(regime, 75.0),
        "version": META_VERSION,
        "regime": regime,
    }


def _ensure_meta_state(state):
    meta = state.setdefault("meta_engine", {})
    meta.setdefault("version", META_VERSION)
    meta.setdefault("regime_profiles", {})
    meta.setdefault("stable_profiles", {})
    meta.setdefault("shadow", {})
    meta.setdefault("last_validation", {})
    meta.setdefault("promotion_count", 0)
    meta.setdefault("rollback_count", 0)
    meta.setdefault("last_runtime", {})
    return meta


def load_signal_history():
    if not os.path.exists(SIGNAL_LOG_FILE):
        return pd.DataFrame()
    try:
        df = pd.read_csv(SIGNAL_LOG_FILE)
        if "tarih" in df.columns:
            df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _ensure_columns(df, columns):
    for col in columns:
        if col not in df.columns:
            df[col] = np.nan
    return df


def log_shock_signals(top_df, regime_snapshot=None, shadow_df=None):
    """Log active/union candidates so the learner gets an auditable training set."""
    if top_df is None or top_df.empty:
        return

    regime_snapshot = regime_snapshot or {}
    shadow_df = shadow_df if shadow_df is not None else pd.DataFrame()

    cols = [
        "ticker", "close", "shock_score", "watch_score", "meta_score", "risk_adjusted_score",
        "z_vol", "z_range", "z_flow", "z_lambda", "resilience_score", "excess_return",
        "rel_1m_pct", "rel_3m_pct", "trend_persistence", "flow_score", "activity_score",
        "liquidity_score", "non_price_score", "overnight_risk", "current_positive",
        "directional_flow_ok", "crash_resilient", "crash_survivor", "extended_but_supported",
        "meta_regime", "meta_regime_confidence", "meta_selection", "tarih",
    ]

    active = top_df.copy()
    if "tarih" not in active.columns:
        active["tarih"] = pd.Timestamp.now().normalize()
    active["model_variant"] = "active"
    active["signal_score"] = pd.to_numeric(active.get("shock_score", 0.0), errors="coerce")

    frames = [active]
    if not shadow_df.empty:
        shadow = shadow_df.copy()
        if "tarih" not in shadow.columns:
            shadow["tarih"] = pd.Timestamp.now().normalize()
        shadow["model_variant"] = "shadow"
        shadow["signal_score"] = pd.to_numeric(
            shadow.get("watch_score", shadow.get("meta_score", 0.0)), errors="coerce"
        )
        frames.append(shadow)

    signals = pd.concat(frames, ignore_index=True, sort=False)
    signals = signals.sort_values("signal_score", ascending=False).head(15)
    signals = _ensure_columns(signals, cols)
    signals = signals[cols + ["model_variant", "signal_score"]].copy()
    signals["regime_label"] = str(regime_snapshot.get("label", "NORMAL"))
    signals["regime_confidence"] = _safe_float(regime_snapshot.get("confidence"), 0.35)
    signals["realized_3d"] = np.nan
    signals["realized_5d"] = np.nan

    history_df = load_signal_history()
    today_val = pd.Timestamp.now().normalize()
    if not history_df.empty:
        history_df = _ensure_columns(history_df, signals.columns.tolist())
        history_df = history_df[
            pd.to_datetime(history_df["tarih"], errors="coerce").dt.normalize() != today_val
        ]
        updated = pd.concat([history_df, signals], ignore_index=True, sort=False)
    else:
        updated = signals

    updated.to_csv(SIGNAL_LOG_FILE, index=False)


def _trading_days_from_history(today):
    dates = []
    if os.path.exists(GECMIS_DOSYA):
        try:
            h = pd.read_csv(GECMIS_DOSYA, usecols=["tarih"])
            dates = pd.to_datetime(h["tarih"], errors="coerce").dropna().dt.normalize().drop_duplicates().sort_values().tolist()
        except Exception:
            dates = []
    today = pd.Timestamp(today).normalize()
    if today not in dates:
        dates.append(today)
    return sorted(set(dates))


def _trading_days_passed(sig_date, today):
    sig_date = pd.Timestamp(sig_date).normalize()
    today = pd.Timestamp(today).normalize()
    if today <= sig_date:
        return 0
    dates = _trading_days_from_history(today)
    dates = [d for d in dates if sig_date <= d <= today]
    if len(dates) >= 2:
        return max(len(dates) - 1, 0)
    return int(np.busday_count(sig_date.date(), today.date()))


def update_realized_shock_returns(current_market_df):
    history_df = load_signal_history()
    if history_df.empty or current_market_df is None or current_market_df.empty:
        return history_df

    price_map = dict(zip(current_market_df["ticker"], current_market_df["close"]))
    today = pd.Timestamp.now().normalize()
    history_df = _ensure_columns(history_df, ["realized_3d", "realized_5d"])

    for idx, row in history_df.iterrows():
        sig_date = pd.to_datetime(row.get("tarih"), errors="coerce")
        if pd.isna(sig_date):
            continue
        ticker = row.get("ticker")
        entry_price = _safe_float(row.get("close"), 0.0)
        if ticker not in price_map or entry_price <= 0:
            continue

        days_passed = _trading_days_passed(sig_date, today)
        current_price = _safe_float(price_map[ticker], 0.0)
        if current_price <= 0:
            continue

        gain = ((current_price - entry_price) / entry_price) * 100.0
        if days_passed >= 3 and pd.isna(history_df.at[idx, "realized_3d"]):
            history_df.at[idx, "realized_3d"] = round(gain, 2)
        if days_passed >= 5 and pd.isna(history_df.at[idx, "realized_5d"]):
            history_df.at[idx, "realized_5d"] = round(gain, 2)

    history_df.to_csv(SIGNAL_LOG_FILE, index=False)
    return history_df


def compute_dynamic_market_thresholds(df):
    if df is None or df.empty:
        return dict(DEFAULT_THRESHOLDS)

    def percentile(column, fallback):
        if column not in df.columns:
            return fallback
        s = pd.to_numeric(df[column], errors="coerce").dropna()
        return float(np.percentile(s, 85)) if not s.empty else fallback

    return {
        "th_vol": round(max(percentile("z_vol", 1.5), 1.2), 2),
        "th_range": round(max(percentile("z_range", 1.5), 1.0), 2),
        "th_flow": round(max(percentile("z_flow", 2.0), 1.5), 2),
        "th_lambda": round(max(percentile("z_lambda", 1.2), 0.5), 2),
    }


def calibrate_adaptive_weights():
    """Legacy factor calibration retained for backward compatibility."""
    history_df = load_signal_history()
    if history_df.empty or "realized_3d" not in history_df.columns:
        return dict(DEFAULT_WEIGHTS), "🕒 ÖĞRENME EVRESİNDE (Örneklem Bekleniyor)"

    valid = history_df.dropna(subset=["realized_3d"]).copy()
    if len(valid) < 20:
        return dict(DEFAULT_WEIGHTS), f"🕒 ÖĞRENME EVRESİNDE ({len(valid)}/20)"

    factors = ["z_vol", "z_flow", "z_range", "z_lambda"]
    y = pd.to_numeric(valid["realized_3d"], errors="coerce").values
    ic_scores = {}
    for factor in factors:
        x = pd.to_numeric(valid.get(factor, np.nan), errors="coerce").values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 10 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
            corr, _ = spearmanr(x[mask], y[mask])
            ic_scores[factor] = max(float(corr) if np.isfinite(corr) else 0.0, 0.0)
        else:
            ic_scores[factor] = 0.05

    total = sum(ic_scores.values())
    if total <= 0:
        return dict(DEFAULT_WEIGHTS), "🕒 GEÇERLİ IC YOK"

    raw = {
        "vol": ic_scores["z_vol"] / total,
        "flow": ic_scores["z_flow"] / total,
        "range": ic_scores["z_range"] / total,
        "lambda": ic_scores["z_lambda"] / total,
    }
    final = {k: 0.70 * DEFAULT_WEIGHTS[k] + 0.30 * raw[k] for k in DEFAULT_WEIGHTS}
    final = _normalize_weights(final, DEFAULT_WEIGHTS)
    return {k: round(v, 4) for k, v in final.items()}, f"🧠 LEGACY FACTOR LEARNING (N={len(valid)})"


def calibrate_resilience_weight(default=DEFAULT_RESILIENCE_WEIGHT):
    history_df = load_signal_history()
    required = ["realized_3d", "resilience_score"]
    if history_df.empty or any(c not in history_df.columns for c in required):
        return float(default), "🛡️ RESILIENCE AĞIRLIĞI: VARSAYILAN"

    valid = history_df.dropna(subset=required).copy()
    if len(valid) < 25:
        return float(default), f"🛡️ RESILIENCE AĞIRLIĞI: ÖRNEKLEM BEKLENİYOR ({len(valid)}/25)"

    x = pd.to_numeric(valid["resilience_score"], errors="coerce").values
    y = pd.to_numeric(valid["realized_3d"], errors="coerce").values
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 25 or np.std(x[mask]) == 0 or np.std(y[mask]) == 0:
        return float(default), "🛡️ RESILIENCE AĞIRLIĞI: VARSAYILAN"

    corr, _ = spearmanr(x[mask], y[mask])
    corr = float(corr) if np.isfinite(corr) else 0.0
    weight = float(np.clip(0.68 + 0.12 * np.clip(corr, -0.5, 0.5), 0.55, 0.80))
    return round(weight, 3), f"🛡️ RESILIENCE AĞIRLIĞI: {weight:.3f} (Spearman {corr:+.3f})"


def _prepare_historical_frame(df_history):
    if df_history is None or df_history.empty:
        return pd.DataFrame()

    df = df_history.copy()
    df["tarih"] = pd.to_datetime(df.get("tarih"), errors="coerce").dt.normalize()
    df = df.dropna(subset=["tarih", "ticker", "close"]).copy()
    df = df.sort_values(["ticker", "tarih"]).reset_index(drop=True)

    numeric_defaults = {
        "change_%": 0.0, "z_vol": 0.0, "z_range": 0.0, "z_flow": 0.0,
        "z_lambda": 0.0, "rvol": 1.0, "value_traded": 0.0,
        "perf_1m": 0.0, "perf_3m": 0.0, "volatility": 2.0,
        "high": np.nan, "low": np.nan, "close": np.nan,
    }
    for col, default in numeric_defaults.items():
        if col not in df.columns:
            df[col] = default
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(default)

    # Forward returns are labels only; they are never fed into a live score.
    df["future_ret_3d"] = df.groupby("ticker")["close"].shift(-3)
    df["future_ret_5d"] = df.groupby("ticker")["close"].shift(-5)
    df["future_ret_3d"] = (df["future_ret_3d"] / df["close"] - 1.0) * 100.0
    df["future_ret_5d"] = (df["future_ret_5d"] / df["close"] - 1.0) * 100.0

    def pct(col):
        return df.groupby("tarih")[col].rank(pct=True, method="average").fillna(0.5) * 100.0

    df["f_flow"] = pct("z_flow")
    df["f_activity"] = pct("rvol") * 0.60 + pct("z_vol") * 0.40
    df["f_liquidity"] = pct("value_traded")
    df["f_event"] = pct("z_range") * 0.35 + pct("z_lambda") * 0.30 + pct("z_vol") * 0.20 + pct("rvol") * 0.15

    high = df["high"].fillna(df["close"])
    low = df["low"].fillna(df["close"])
    close = df["close"]
    day_range = (high - low).replace(0.0, np.nan)
    close_location = (((close - low) / day_range) * 100.0).replace([np.inf, -np.inf], np.nan).fillna(50.0).clip(0.0, 100.0)

    rel_daily = pct("change_%")
    rel_1m = pct("perf_1m")
    rel_3m = pct("perf_3m")
    rel_rvol = pct("rvol")
    rel_flow = pct("z_flow")
    excess = df["change_%"] - df.groupby("tarih")["change_%"].transform("median")
    excess_pct = excess.groupby(df["tarih"]).rank(pct=True, method="average").fillna(0.5) * 100.0
    trend_persistence = rel_1m * 0.45 + rel_3m * 0.55

    df["f_resilience"] = (
        rel_daily * 0.30 + excess_pct * 0.15 + trend_persistence * 0.10
        + rel_rvol * 0.10 + rel_flow * 0.10 + close_location * 0.10 + rel_3m * 0.15
    )

    df["risk_proxy"] = np.clip(
        pct("volatility") * 0.25
        + pct("z_range") * 0.20
        + (100.0 - df["f_liquidity"]) * 0.25
        + (100.0 - df["f_flow"]) * 0.20
        + (100.0 - pct("rvol")) * 0.10,
        0.0,
        100.0,
    )

    regime_rows = []
    for date, group in df.groupby("tarih", sort=True):
        snapshot = classify_bist_regime(group)
        regime_rows.append((date, snapshot["label"], snapshot["confidence"]))
    regime_map = {d: (r, c) for d, r, c in regime_rows}
    df["regime_label"] = df["tarih"].map(lambda x: regime_map.get(x, ("NORMAL", 0.35))[0])
    df["regime_confidence"] = df["tarih"].map(lambda x: regime_map.get(x, ("NORMAL", 0.35))[1])
    return df


def prepare_historical_frame(df_history):
    return _prepare_historical_frame(df_history)


def score_profile_frame(df, profile):
    if df is None or df.empty:
        return pd.Series(dtype=float)
    weights = _normalize_weights(profile.get("weights", {}), DEFAULT_META_WEIGHTS)
    score = (
        df["f_event"] * weights["event"]
        + df["f_flow"] * weights["flow"]
        + df["f_activity"] * weights["activity"]
        + df["f_liquidity"] * weights["liquidity"]
        + df["f_resilience"] * weights["resilience"]
    )
    score = score - np.clip((df["risk_proxy"] - 65.0) * 0.18, 0.0, 10.0)
    score = score.clip(0.0, 99.5)
    return score


def evaluate_profile(df, profile, min_samples=20, target="future_ret_3d"):
    if df is None or df.empty:
        return {"n": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_return": 0.0, "p10": 0.0, "median": 0.0, "score": -999.0}

    data = df.copy()
    data["profile_score"] = score_profile_frame(data, profile)
    threshold = _safe_float(profile.get("min_score"), 75.0)
    selected = data[
        (data["profile_score"] >= threshold)
        & (data["change_%"] > 0.0)
        & (data["value_traded"] >= 25_000_000.0)
        & (data[target].notna())
    ].copy()
    ret = pd.to_numeric(selected[target], errors="coerce").dropna()
    if len(ret) < min_samples:
        return {"n": int(len(ret)), "win_rate": float((ret > 0).mean() * 100.0) if len(ret) else 0.0, "profit_factor": 0.0, "avg_return": float(ret.mean()) if len(ret) else 0.0, "p10": float(ret.quantile(0.10)) if len(ret) else 0.0, "median": float(ret.median()) if len(ret) else 0.0, "score": -999.0}

    gains = ret[ret > 0].sum()
    losses = abs(ret[ret < 0].sum())
    pf = float(gains / losses) if losses > 0 else 5.0
    wr = float((ret > 0).mean() * 100.0)
    avg = float(ret.mean())
    p10 = float(ret.quantile(0.10))
    median = float(ret.median())
    sample_factor = min(len(ret) / 40.0, 1.0)
    metric_score = (avg * 0.45 + np.log1p(pf) * 2.4 + wr * 0.018 + p10 * 0.08) * sample_factor
    return {
        "n": int(len(ret)),
        "win_rate": wr,
        "profit_factor": pf,
        "avg_return": avg,
        "p10": p10,
        "median": median,
        "score": float(metric_score),
    }


def optimize_profile_threshold(df, profile, target="future_ret_3d"):
    if df is None or df.empty:
        return _safe_float(profile.get("min_score"), 75.0)
    best = None
    for threshold in np.arange(68.0, 93.0, 2.0):
        candidate = dict(profile)
        candidate["min_score"] = float(threshold)
        metrics = evaluate_profile(df, candidate, min_samples=15, target=target)
        if metrics["n"] < 15:
            continue
        item = {"threshold": float(threshold), **metrics}
        if best is None or item["score"] > best["score"]:
            best = item
    return round(best["threshold"], 1) if best else _safe_float(profile.get("min_score"), 75.0)


def learn_meta_candidate(training_df, current_regime):
    regime = str(current_regime).upper()
    base = _default_profile(regime)
    if training_df is None or training_df.empty:
        base["learning_status"] = "NO_DATA"
        return base

    regime_df = training_df[
        (training_df["regime_label"] == regime) & training_df["future_ret_3d"].notna()
    ].copy()
    if len(regime_df) < 120:
        regime_df = training_df[training_df["future_ret_3d"].notna()].copy()

    factor_cols = {
        "event": "f_event",
        "flow": "f_flow",
        "activity": "f_activity",
        "liquidity": "f_liquidity",
        "resilience": "f_resilience",
    }
    y = pd.to_numeric(regime_df["future_ret_3d"], errors="coerce").values
    edges = {}
    for family, col in factor_cols.items():
        x = pd.to_numeric(regime_df[col], errors="coerce").values
        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() >= 40 and np.std(x[mask]) > 0 and np.std(y[mask]) > 0:
            corr, _ = spearmanr(x[mask], y[mask])
            corr = float(corr) if np.isfinite(corr) else 0.0
        else:
            corr = 0.0
        edges[family] = max(corr, 0.02)

    total = sum(edges.values())
    learned = {k: edges[k] / total for k in edges}
    n = len(regime_df)
    learned_blend = 0.15 if n < 250 else (0.25 if n < 600 else 0.35)

    final = {
        k: (1.0 - learned_blend) * base["weights"][k] + learned_blend * learned[k]
        for k in DEFAULT_META_WEIGHTS
    }
    final = _normalize_weights(final, base["weights"])
    base["weights"] = {k: round(v, 4) for k, v in final.items()}
    optimized_threshold = optimize_profile_threshold(regime_df, base, target="future_ret_3d")
    base["min_score"] = max(float(optimized_threshold), REGIME_MIN_SCORES.get(regime, 75.0))
    base["training_samples"] = int(n)
    base["learned_edges"] = {k: round(v, 4) for k, v in edges.items()}
    base["learning_status"] = "LEARNED"
    return base


def split_training_validation(history_df, validation_days=3, forward_days=3):
    df = prepare_historical_frame(history_df)
    if df.empty:
        return pd.DataFrame(), pd.DataFrame(), []

    dates = sorted(df["tarih"].dropna().unique())
    if len(dates) <= validation_days + forward_days + 2:
        return df.iloc[0:0].copy(), df.iloc[0:0].copy(), dates

    validation_start_index = len(dates) - forward_days - validation_days
    train_end_index = max(validation_start_index - forward_days, 0)
    train_dates = dates[:train_end_index]
    validation_dates = dates[validation_start_index:]
    train = df[df["tarih"].isin(train_dates)].copy()
    validation = df[df["tarih"].isin(validation_dates)].copy()
    validation = validation[validation["future_ret_5d"].notna()].copy()
    return train, validation, validation_dates


def build_runtime_meta_profile(regime_snapshot, state=None):
    state = state if isinstance(state, dict) else load_ai_state()
    meta = _ensure_meta_state(state)
    regime = str(regime_snapshot.get("label", "NORMAL")).upper()
    confidence = float(np.clip(_safe_float(regime_snapshot.get("confidence"), 0.35), 0.0, 1.0))

    stored = meta.get("regime_profiles", {}).get(regime)
    if not isinstance(stored, dict):
        stored = _default_profile(regime)

    template = REGIME_META_TEMPLATES.get(regime, DEFAULT_META_WEIGHTS)
    learned = _normalize_weights(stored.get("weights", {}), template)
    blend = 0.25 + 0.55 * confidence
    runtime_weights = {
        k: (1.0 - blend) * learned[k] + blend * template[k]
        for k in DEFAULT_META_WEIGHTS
    }
    runtime_weights = _normalize_weights(runtime_weights, template)

    min_score = _safe_float(stored.get("min_score"), REGIME_MIN_SCORES.get(regime, 75.0))
    meta["last_runtime"] = {
        "regime": regime,
        "confidence": round(confidence, 3),
        "weights": {k: round(v, 4) for k, v in runtime_weights.items()},
        "min_score": round(min_score, 1),
    }

    return {
        "version": META_VERSION,
        "regime": regime,
        "confidence": round(confidence, 3),
        "weights": {k: round(v, 4) for k, v in runtime_weights.items()},
        "min_score": round(min_score, 1),
        "source": "REGIME_PROFILE+LIVE_REGIME_BLEND",
    }
