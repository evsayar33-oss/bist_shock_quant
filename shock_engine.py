import os
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

GECMIS_DOSYA = "gecmis_veri.csv"

DEFAULT_THRESHOLDS = {
    "th_vol": 1.5,
    "th_range": 1.5,
    "th_flow": 2.0,
    "th_lambda": 1.2,
}

DEFAULT_WEIGHTS = {
    "vol": 0.35,
    "flow": 0.35,
    "range": 0.20,
    "lambda": 0.10,
}

DEFAULT_META_WEIGHTS = {
    "event": 0.20,
    "flow": 0.25,
    "activity": 0.20,
    "liquidity": 0.15,
    "resilience": 0.20,
}

REGIME_META_TEMPLATES = {
    "CRASH": {
        "event": 0.14,
        "flow": 0.24,
        "activity": 0.14,
        "liquidity": 0.12,
        "resilience": 0.36,
    },
    "STRESS": {
        "event": 0.16,
        "flow": 0.25,
        "activity": 0.16,
        "liquidity": 0.13,
        "resilience": 0.30,
    },
    "ROTATION": {
        "event": 0.16,
        "flow": 0.28,
        "activity": 0.18,
        "liquidity": 0.15,
        "resilience": 0.23,
    },
    "EXPANSION": {
        "event": 0.18,
        "flow": 0.31,
        "activity": 0.23,
        "liquidity": 0.15,
        "resilience": 0.13,
    },
    "QUIET": {
        "event": 0.14,
        "flow": 0.24,
        "activity": 0.22,
        "liquidity": 0.20,
        "resilience": 0.20,
    },
    "NORMAL": DEFAULT_META_WEIGHTS,
}


def gecmis_veriyi_yukle():
    if not os.path.exists(GECMIS_DOSYA):
        return pd.DataFrame()
    try:
        df = pd.read_csv(GECMIS_DOSYA)
        if "tarih" in df.columns:
            df["tarih"] = pd.to_datetime(df["tarih"], errors="coerce")
        return df
    except Exception:
        return pd.DataFrame()


def _num(value, default=0.0):
    try:
        value = float(value)
        return default if not np.isfinite(value) else value
    except Exception:
        return default


def _pct_rank(series):
    s = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan)
    if s.notna().sum() <= 1:
        return pd.Series(50.0, index=series.index)
    return s.rank(pct=True, method="average").fillna(0.5) * 100.0


def _normalize_weights(weights, fallback):
    out = {}
    for key in fallback:
        value = _num((weights or {}).get(key), fallback[key])
        out[key] = max(value, 0.0)
    total = sum(out.values())
    if total <= 0:
        return dict(fallback)
    return {key: out[key] / total for key in out}


def classify_bist_regime(df):
    """Cross-sectional regime classifier. Uses only current scan data."""
    if df is None or df.empty or "change_%" not in df.columns:
        return {
            "label": "NORMAL",
            "confidence": 0.35,
            "is_crashing": False,
            "red_ratio": 0.50,
            "mean_change": 0.0,
            "median_change": 0.0,
            "dispersion": 0.0,
            "breadth_balance": 0.0,
        }

    changes = pd.to_numeric(df["change_%"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if changes.empty:
        return {
            "label": "NORMAL",
            "confidence": 0.35,
            "is_crashing": False,
            "red_ratio": 0.50,
            "mean_change": 0.0,
            "median_change": 0.0,
            "dispersion": 0.0,
            "breadth_balance": 0.0,
        }

    red_ratio = float((changes < 0).mean())
    green_ratio = float((changes > 0).mean())
    mean_change = float(changes.mean())
    median_change = float(changes.median())
    dispersion = float(changes.std(ddof=0)) if len(changes) > 1 else 0.0
    breadth_balance = green_ratio - red_ratio

    # Rejim sınırları kasıtlı olarak geniş tutuluyor; tek metrikle rejim atlanmaması için
    # birden fazla kanıtın aynı yöne gitmesi aranıyor.
    if red_ratio >= 0.70 or mean_change <= -1.50 or median_change <= -1.00:
        label = "CRASH"
    elif red_ratio >= 0.60 or median_change <= -0.45:
        label = "STRESS"
    elif green_ratio >= 0.62 and mean_change >= 0.55 and median_change >= 0.30:
        label = "EXPANSION"
    elif dispersion >= max(2.25, abs(mean_change) * 1.8 + 1.0) and 0.35 <= red_ratio <= 0.65:
        label = "ROTATION"
    elif dispersion <= 1.20 and abs(median_change) <= 0.35:
        label = "QUIET"
    else:
        label = "NORMAL"

    distance_terms = []
    if label in ("CRASH", "STRESS"):
        distance_terms.extend([
            red_ratio / 0.85,
            abs(min(median_change, 0.0)) / 1.80,
            abs(min(mean_change, 0.0)) / 2.50,
        ])
    elif label == "EXPANSION":
        distance_terms.extend([
            green_ratio / 0.85,
            max(mean_change, 0.0) / 1.80,
            max(median_change, 0.0) / 1.20,
        ])
    elif label == "ROTATION":
        distance_terms.extend([
            dispersion / 3.50,
            (1.0 - abs(red_ratio - 0.50) / 0.50),
        ])
    else:
        distance_terms.extend([
            1.0 - min(abs(median_change) / 1.5, 1.0),
            1.0 - min(abs(breadth_balance) / 1.0, 1.0),
        ])

    confidence = float(np.clip(np.mean(distance_terms) if distance_terms else 0.35, 0.20, 0.98))

    return {
        "label": label,
        "confidence": round(confidence, 3),
        "is_crashing": label == "CRASH",
        "red_ratio": round(red_ratio, 4),
        "mean_change": round(mean_change, 3),
        "median_change": round(median_change, 3),
        "dispersion": round(dispersion, 3),
        "breadth_balance": round(breadth_balance, 4),
    }


def calculate_shock_scores(
    df,
    df_gecmis,
    dynamic_thresholds=None,
    dynamic_weights=None,
    market_is_crashing=False,
    regime_snapshot=None,
    meta_profile=None,
):
    """
    Adaptive BIST Meta-Engine scoring layer.

    Mimari:
      1) Ham shock/event faktörleri
      2) Cross-sectional resilience
      3) Non-price-dominant meta factors: event / flow / activity / liquidity
      4) Rejim koşullu ağırlıklandırma
      5) Overnight risk proxy + kalite kontrolü
      6) Fiyat değişimi ana skor değil; yalnızca mevcut yönün son doğrulamasıdır.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    dynamic_thresholds = {**DEFAULT_THRESHOLDS, **(dynamic_thresholds or {})}
    dynamic_weights = {**DEFAULT_WEIGHTS, **(dynamic_weights or {})}

    if regime_snapshot is None:
        regime_snapshot = classify_bist_regime(df)

    regime_label = str(regime_snapshot.get("label", "NORMAL")).upper()
    regime_confidence = float(np.clip(_num(regime_snapshot.get("confidence"), 0.35), 0.0, 1.0))

    template = REGIME_META_TEMPLATES.get(regime_label, REGIME_META_TEMPLATES["NORMAL"])
    supplied_meta = (meta_profile or {}).get("weights", {}) if isinstance(meta_profile, dict) else {}
    learned_meta = _normalize_weights(supplied_meta, DEFAULT_META_WEIGHTS)

    # Rejim şablonu canlı koşulları; öğrenilmiş profil ise geçmiş doğrulanmış davranışı taşır.
    regime_blend = 0.45 + 0.40 * regime_confidence
    meta_weights = {
        key: ((1.0 - regime_blend) * learned_meta[key]) + (regime_blend * template.get(key, DEFAULT_META_WEIGHTS[key]))
        for key in DEFAULT_META_WEIGHTS
    }
    meta_weights = _normalize_weights(meta_weights, DEFAULT_META_WEIGHTS)

    scored_data = []
    for _, row in df.iterrows():
        item = row.to_dict()

        close = _num(item.get("close"), 0.0)
        open_p = _num(item.get("open"), close)
        high = _num(item.get("high"), close)
        low = _num(item.get("low"), close)
        change = _num(item.get("change_%"), 0.0)
        value_traded = _num(item.get("value_traded"), 0.0)
        rvol = max(_num(item.get("rvol"), 1.0), 0.0)
        atr = max(_num(item.get("atr"), 1.0), 0.01)
        perf_1m = _num(item.get("perf_1m"), 0.0)
        perf_3m = _num(item.get("perf_3m"), 0.0)
        volatility = max(_num(item.get("volatility"), 2.0), 0.0)

        z_vol = np.clip((rvol - 1.0) * 2.2, -2.0, 6.0)
        today_range = max(high - low, 0.0)
        z_range = np.clip(((today_range / atr) - 1.0) * 2.5, -2.0, 6.0)

        liquidity_damping = min(value_traded / 25_000_000.0, 1.0) if value_traded > 0 else 0.0
        raw_lambda = (
            abs(change) / ((value_traded / 10_000_000.0) + 1e-9) * liquidity_damping
            if value_traded > 0
            else 0.0
        )
        z_lambda = np.clip(np.log1p(max(raw_lambda, 0.0)) * 2.0, 0.0, 5.0)

        if today_range > 0:
            clv = ((close - low) - (high - close)) / today_range
            body_eff = (close - open_p) / today_range
        else:
            clv = 0.0
            body_eff = 0.0

        aggressor_flow = (max(clv, 0.0) * 0.55) + (max(body_eff, 0.0) * 0.45)
        z_flow = np.clip(aggressor_flow * 4.0, 0.0, 6.0)

        entry_bonus = 0.0
        if 2.0 <= change <= 5.5:
            entry_bonus = 4.0
            entry_status = "🎯 UYGUN GİRİŞ BÖLGESİ"
        elif change > 6.5:
            if perf_1m > 0 and perf_3m > 0 and rvol >= 1.0:
                entry_bonus = 1.0
                entry_status = "🔥 UZUN VADELİ DESTEK VAR"
            else:
                entry_bonus = -4.0
                entry_status = "⚠️ UZAMIŞ HAREKET"
        else:
            entry_status = "NORMAL GİRİŞ"

        is_volume_fake = (rvol < 0.65) and (z_flow < 0.8)
        is_illiquid = value_traded < 25_000_000.0

        shock_count = sum([
            z_vol >= _num(dynamic_thresholds.get("th_vol"), 1.5),
            z_range >= _num(dynamic_thresholds.get("th_range"), 1.5),
            z_flow >= _num(dynamic_thresholds.get("th_flow"), 2.0),
            z_lambda >= _num(dynamic_thresholds.get("th_lambda"), 1.2),
        ])
        concordance_bonus = min(shock_count * 4.0, 16.0)

        is_downtrend_knife = (perf_3m < -25.0) and (z_vol < 2.5) and (change <= 0.0)

        item.update({
            "z_vol": round(float(z_vol), 2),
            "z_range": round(float(z_range), 2),
            "z_lambda": round(float(z_lambda), 2),
            "z_flow": round(float(z_flow), 2),
            "shock_count": int(shock_count),
            "concordance_bonus": round(float(concordance_bonus), 2),
            "entry_bonus": round(float(entry_bonus), 2),
            "entry_status": entry_status,
            "is_downtrend_knife": is_downtrend_knife,
            "is_volume_fake": is_volume_fake,
            "is_illiquid": is_illiquid,
            "regime_label": regime_label,
            "regime_confidence": round(regime_confidence, 3),
        })
        scored_data.append(item)

    res_df = pd.DataFrame(scored_data)
    if res_df.empty:
        return res_df

    # ------------------------------------------------------------
    # Cross-sectional market-relative features
    # ------------------------------------------------------------
    change_s = pd.to_numeric(res_df["change_%"], errors="coerce").fillna(0.0)
    perf_1m_s = pd.to_numeric(res_df.get("perf_1m", 0.0), errors="coerce").fillna(0.0)
    perf_3m_s = pd.to_numeric(res_df.get("perf_3m", 0.0), errors="coerce").fillna(0.0)
    rvol_s = pd.to_numeric(res_df.get("rvol", 1.0), errors="coerce").fillna(1.0)
    flow_s = pd.to_numeric(res_df["z_flow"], errors="coerce").fillna(0.0)
    value_s = pd.to_numeric(res_df.get("value_traded", 0.0), errors="coerce").fillna(0.0)
    volatility_s = pd.to_numeric(res_df.get("volatility", 2.0), errors="coerce").fillna(2.0)

    market_median_change = float(change_s.median())
    market_mean_change = float(change_s.mean())
    market_red_ratio = float((change_s < 0).mean()) if len(change_s) else 0.5

    res_df["market_median_change"] = round(market_median_change, 3)
    res_df["market_mean_change"] = round(market_mean_change, 3)
    res_df["market_red_ratio"] = round(market_red_ratio, 4)
    res_df["excess_return"] = (change_s - market_median_change).round(3)
    res_df["excess_vs_mean"] = (change_s - market_mean_change).round(3)

    rel_daily_pct = _pct_rank(change_s)
    excess_pct = _pct_rank(res_df["excess_return"])
    rel_1m_pct = _pct_rank(perf_1m_s)
    rel_3m_pct = _pct_rank(perf_3m_s)
    rel_rvol_pct = _pct_rank(rvol_s)
    rel_flow_pct = _pct_rank(flow_s)
    liquidity_pct = _pct_rank(value_s)
    volatility_pct = _pct_rank(volatility_s)

    high_s = pd.to_numeric(res_df["high"], errors="coerce").fillna(0.0)
    low_s = pd.to_numeric(res_df["low"], errors="coerce").fillna(0.0)
    close_s = pd.to_numeric(res_df["close"], errors="coerce").fillna(0.0)
    range_s = (high_s - low_s).replace(0.0, np.nan)
    close_location = (((close_s - low_s) / range_s) * 100.0).replace([np.inf, -np.inf], np.nan).fillna(50.0).clip(0.0, 100.0)

    trend_persistence = (rel_1m_pct * 0.45) + (rel_3m_pct * 0.55)

    resilience_score = (
        rel_daily_pct * 0.30
        + excess_pct * 0.15
        + trend_persistence * 0.10
        + rel_rvol_pct * 0.10
        + rel_flow_pct * 0.10
        + close_location * 0.10
        + rel_3m_pct * 0.15
    )

    # ------------------------------------------------------------
    # Meta factor families
    # ------------------------------------------------------------
    w_v = _num(dynamic_weights.get("vol"), 0.35)
    w_f = _num(dynamic_weights.get("flow"), 0.35)
    w_r = _num(dynamic_weights.get("range"), 0.20)
    w_l = _num(dynamic_weights.get("lambda"), 0.10)
    legacy_weights = _normalize_weights(
        {"vol": w_v, "flow": w_f, "range": w_r, "lambda": w_l},
        DEFAULT_WEIGHTS,
    )

    pct_vol = _pct_rank(res_df["z_vol"])
    pct_range = _pct_rank(res_df["z_range"])
    pct_lambda = _pct_rank(res_df["z_lambda"])
    pct_flow = _pct_rank(res_df["z_flow"])

    legacy_base = (
        pct_vol * legacy_weights["vol"]
        + pct_flow * legacy_weights["flow"]
        + pct_range * legacy_weights["range"]
        + pct_lambda * legacy_weights["lambda"]
        + res_df["concordance_bonus"]
        + res_df["entry_bonus"]
    )
    legacy_base -= np.where(res_df["is_volume_fake"], 8.0, 0.0)
    legacy_base -= np.where(res_df["is_illiquid"], 10.0, 0.0)

    event_score = (
        pct_range * 0.35
        + pct_lambda * 0.30
        + pct_vol * 0.20
        + rel_rvol_pct * 0.15
    )
    flow_score = rel_flow_pct
    activity_score = rel_rvol_pct * 0.60 + pct_vol * 0.40
    liquidity_score = liquidity_pct
    resilience_family = resilience_score
    non_price_score = (
        event_score * 0.30
        + flow_score * 0.30
        + activity_score * 0.20
        + liquidity_score * 0.20
    )

    # Overnight risk proxy: doğrudan fiyat tahmini değil; oynaklık, sığlık ve akış kalitesinin
    # birleşiminden oluşan koruyucu bir risk katmanıdır.
    overnight_risk = np.clip(
        volatility_pct * 0.25
        + pct_range * 0.20
        + (100.0 - liquidity_score) * 0.25
        + (100.0 - flow_score) * 0.20
        + (100.0 - rel_rvol_pct) * 0.10,
        0.0,
        100.0,
    )

    meta_score = (
        event_score * meta_weights["event"]
        + flow_score * meta_weights["flow"]
        + activity_score * meta_weights["activity"]
        + liquidity_score * meta_weights["liquidity"]
        + resilience_family * meta_weights["resilience"]
    )

    # Çok yüksek riskte skoru kontrollü düşür; tamamen veto etmiyoruz.
    risk_adjusted_score = meta_score - np.clip((overnight_risk - 65.0) * 0.18, 0.0, 10.0)
    risk_adjusted_score += np.where(res_df["is_volume_fake"], -6.0, 0.0)
    risk_adjusted_score += np.where(res_df["is_illiquid"], -6.0, 0.0)
    risk_adjusted_score += np.where(res_df["is_downtrend_knife"], -15.0, 0.0)

    # Crash/stress içinde relative resilience'in adaptif etkisi güçlenir.
    if regime_label == "CRASH":
        risk_adjusted_score += np.where(res_df["excess_return"] > 0.0, 5.0, -5.0)
    elif regime_label == "STRESS":
        risk_adjusted_score += np.where(res_df["excess_return"] > 0.0, 3.0, -2.0)

    risk_adjusted_score = np.clip(risk_adjusted_score + np.clip(legacy_base - 50.0, -12.0, 12.0) * 0.10, 0.0, 99.5)

    # Fiyat yönü ana puanı üretmez; yalnızca şu anda pozitif olup olmadığını son kapı olarak belirtir.
    directional_flow_ok = flow_score >= 45.0
    current_positive = change_s > 0.0
    valid_long_candidate = current_positive & directional_flow_ok & (~res_df["is_downtrend_knife"])

    # Crash gününde sadece piyasadan daha iyi + yeterli kalite; normalde daha geniş.
    crash_resilient = (
        current_positive
        & (res_df["excess_return"] > 0.0)
        & (resilience_score >= 55.0)
        & (liquidity_score >= 35.0)
    )
    crash_leader = crash_resilient & (resilience_score >= 70.0) & (rel_daily_pct >= 70.0)

    extended_but_supported = (
        (change_s > 6.0)
        & (perf_1m_s > 0.0)
        & (perf_3m_s > 0.0)
        & (rvol_s >= 1.0)
        & (res_df["excess_return"] > 0.0)
    )

    res_df["rel_daily_pct"] = rel_daily_pct.round(1)
    res_df["rel_1m_pct"] = rel_1m_pct.round(1)
    res_df["rel_3m_pct"] = rel_3m_pct.round(1)
    res_df["rel_rvol_pct"] = rel_rvol_pct.round(1)
    res_df["rel_flow_pct"] = rel_flow_pct.round(1)
    res_df["liquidity_pct"] = liquidity_score.round(1)
    res_df["trend_persistence"] = trend_persistence.round(1)
    res_df["close_location"] = close_location.round(1)
    res_df["resilience_score"] = resilience_score.round(1)
    res_df["event_score"] = event_score.round(1)
    res_df["flow_score"] = flow_score.round(1)
    res_df["activity_score"] = activity_score.round(1)
    res_df["liquidity_score"] = liquidity_score.round(1)
    res_df["non_price_score"] = non_price_score.round(1)
    res_df["overnight_risk"] = overnight_risk.round(1)
    res_df["legacy_score"] = np.clip(np.round(legacy_base, 1), 0.0, 99.5)
    res_df["meta_score"] = np.clip(np.round(meta_score, 1), 0.0, 99.5)
    res_df["risk_adjusted_score"] = np.clip(np.round(risk_adjusted_score, 1), 0.0, 99.5)
    res_df["crash_resilient"] = crash_resilient
    res_df["crash_survivor"] = crash_leader
    res_df["extended_but_supported"] = extended_but_supported
    res_df["directional_flow_ok"] = directional_flow_ok
    res_df["current_positive"] = current_positive
    res_df["meta_regime"] = regime_label
    res_df["meta_regime_confidence"] = round(regime_confidence, 3)
    res_df["meta_weight_event"] = round(meta_weights["event"], 4)
    res_df["meta_weight_flow"] = round(meta_weights["flow"], 4)
    res_df["meta_weight_activity"] = round(meta_weights["activity"], 4)
    res_df["meta_weight_liquidity"] = round(meta_weights["liquidity"], 4)
    res_df["meta_weight_resilience"] = round(meta_weights["resilience"], 4)

    # min score runtime profilinden gelir; crash/stress için güvenlik primi uygulanır.
    profile_min = _num((meta_profile or {}).get("min_score"), 75.0) if isinstance(meta_profile, dict) else 75.0
    regime_surcharge = {"CRASH": 5.0, "STRESS": 3.0, "ROTATION": 1.0, "QUIET": -2.0, "EXPANSION": 0.0, "NORMAL": 0.0}.get(regime_label, 0.0)
    effective_min = float(np.clip(profile_min + regime_surcharge, 65.0, 94.0))

    # Ana üretim skoru: non-price/meta engine. Fiyat, yalnızca son yön kapısıdır.
    final_score = np.clip(np.round(risk_adjusted_score, 1), 0.0, 99.5)
    res_df["shock_score"] = np.where(valid_long_candidate, final_score, 0.0)
    res_df["watch_score"] = final_score
    res_df["confidence_score"] = final_score
    res_df["effective_min_score"] = round(effective_min, 1)

    def assign_allocation(row):
        score = _num(row.get("shock_score"), 0.0)
        resilience = _num(row.get("resilience_score"), 0.0)
        risk = _num(row.get("overnight_risk"), 100.0)
        leader = bool(row.get("crash_survivor", False))
        resilient = bool(row.get("crash_resilient", False))

        if score < effective_min:
            return "⭐", "İşlem Açma (Meta Eşiği Altı)"
        if risk >= 82.0:
            return "⭐", "İşlem Açma (Yüksek Overnight Risk)"

        if regime_label == "CRASH":
            if leader and score >= max(88.0, effective_min + 4.0):
                return "⭐⭐⭐⭐⭐", "Kontrollü Pozisyon (%8 - %12) / Crash Leader"
            if resilient and score >= effective_min:
                return "⭐⭐⭐⭐", "Kontrollü Pozisyon (%4 - %8) / Crash Survivor"
            return "⭐⭐⭐", "Küçük Kademe (%2 - %4) / Seçici"

        if score >= effective_min + 8.0 and resilience >= 70.0 and risk < 62.0:
            return "⭐⭐⭐⭐⭐", "Yüksek Güven (%8 - %12)"
        if score >= effective_min and resilience >= 55.0 and risk < 72.0:
            return "⭐⭐⭐⭐", "Dengeli Pozisyon (%4 - %8)"
        return "⭐⭐⭐", "Küçük Kademe (%1 - %3)"

    allocations = [assign_allocation(r) for _, r in res_df.iterrows()]
    res_df["stars"] = [x[0] for x in allocations]
    res_df["allocation"] = [x[1] for x in allocations]
    res_df["meta_selection"] = np.where(
        res_df["shock_score"] >= effective_min,
        "SELECTED",
        np.where(res_df["watch_score"] >= effective_min - 5.0, "WATCH", "REJECTED"),
    )

    drop_cols = [
        "pct_vol",
        "pct_range",
        "pct_lambda",
        "pct_flow",
        "concordance_bonus",
        "is_downtrend_knife",
        "entry_bonus",
        "is_volume_fake",
        "is_illiquid",
    ]
    res_df = res_df.drop(columns=[c for c in drop_cols if c in res_df.columns])

    return res_df.sort_values(
        by=["shock_score", "risk_adjusted_score", "resilience_score", "excess_return"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
