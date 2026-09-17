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


def gecmis_veriyi_yukle():
    if not os.path.exists(GECMIS_DOSYA):
        return pd.DataFrame()

    try:
        df = pd.read_csv(GECMIS_DOSYA)

        if "tarih" in df.columns:
            df["tarih"] = pd.to_datetime(
                df["tarih"],
                errors="coerce"
            )

        return df

    except Exception:
        return pd.DataFrame()


def _num(value, default=0.0):
    try:
        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def _pct_rank(series):
    s = pd.to_numeric(
        series,
        errors="coerce"
    ).replace(
        [np.inf, -np.inf],
        np.nan
    )

    if s.notna().sum() <= 1:
        return pd.Series(
            50.0,
            index=series.index
        )

    return (
        s.rank(
            pct=True,
            method="average"
        )
        .fillna(0.5)
        * 100.0
    )


def calculate_shock_scores(
    df,
    df_gecmis,
    dynamic_thresholds=None,
    dynamic_weights=None,
    market_is_crashing=False
):
    """
    BIST Shock Engine V2

    Yeni ana mimari:

    1. Klasik shock faktörleri korunur.
    2. Aynı gün taranan tüm hisseler arasında relative-strength ölçülür.
    3. Crash rejiminde relative resilience ana faktör haline gelir.
    4. RVOL artık tek başına zorunlu crash filtresi değildir.
    5. Günlük +6% üzeri hareket otomatik olarak reddedilmez.
    6. Eski günlük OHLC tabanlı sahte VWAP vetosu kaldırılmıştır.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    dynamic_thresholds = {
        **DEFAULT_THRESHOLDS,
        **(dynamic_thresholds or {})
    }

    dynamic_weights = {
        **DEFAULT_WEIGHTS,
        **(dynamic_weights or {})
    }

    scored_data = []

    for _, row in df.iterrows():

        item = row.to_dict()

        close = _num(
            item.get("close"),
            0.0
        )

        open_p = _num(
            item.get("open"),
            close
        )

        high = _num(
            item.get("high"),
            close
        )

        low = _num(
            item.get("low"),
            close
        )

        change = _num(
            item.get("change_%"),
            0.0
        )

        value_traded = _num(
            item.get("value_traded"),
            0.0
        )

        rvol = max(
            _num(item.get("rvol"), 1.0),
            0.0
        )

        atr = max(
            _num(item.get("atr"), 1.0),
            0.01
        )

        perf_1m = _num(
            item.get("perf_1m"),
            0.0
        )

        perf_3m = _num(
            item.get("perf_3m"),
            0.0
        )

        # ============================================================
        # 1. KLASİK SHOCK FAKTÖRLERİ
        # ============================================================

        z_vol = np.clip(
            (rvol - 1.0) * 2.2,
            -2.0,
            6.0
        )

        today_range = max(
            high - low,
            0.0
        )

        z_range = np.clip(
            ((today_range / atr) - 1.0) * 2.5,
            -2.0,
            6.0
        )

        liquidity_damping = (
            min(
                value_traded / 25_000_000.0,
                1.0
            )
            if value_traded > 0
            else 0.0
        )

        raw_lambda = (
            abs(change)
            / ((value_traded / 10_000_000.0) + 1e-9)
            * liquidity_damping
            if value_traded > 0
            else 0.0
        )

        z_lambda = np.clip(
            np.log1p(
                max(raw_lambda, 0.0)
            ) * 2.0,
            0.0,
            5.0
        )

        if today_range > 0:

            clv = (
                ((close - low) - (high - close))
                / today_range
            )

            body_eff = (
                (close - open_p)
                / today_range
            )

        else:

            clv = 0.0
            body_eff = 0.0

        aggressor_flow = (
            max(clv, 0.0) * 0.55
            +
            max(body_eff, 0.0) * 0.45
        )

        z_flow = np.clip(
            aggressor_flow * 4.0,
            0.0,
            6.0
        )

        z_vol = round(
            float(z_vol),
            2
        )

        z_range = round(
            float(z_range),
            2
        )

        z_lambda = round(
            float(z_lambda),
            2
        )

        z_flow = round(
            float(z_flow),
            2
        )

        # ============================================================
        # 2. GİRİŞ BÖLGESİ
        # ============================================================

        if 2.0 <= change <= 5.5:

            entry_bonus = 6.0
            entry_status = "🎯 İDEAL GİRİŞ BÖLGESİ"

        elif change > 6.5:

            # Artık büyük hareket otomatik olarak çöpe atılmıyor.
            if (
                perf_1m > 0
                and perf_3m > 0
                and rvol >= 1.0
            ):
                entry_bonus = 2.0
                entry_status = (
                    "🔥 GÜÇLÜ HAREKET / DESTEKLİ"
                )

            else:
                entry_bonus = -5.0
                entry_status = (
                    "⚠️ UZAMIŞ HAREKET / "
                    "TAZE GİRİŞ RİSKİ"
                )

        else:

            entry_bonus = 0.0
            entry_status = "NORMAL GİRİŞ"

        # ============================================================
        # 3. HACİM / LİKİDİTE
        # ============================================================

        # Artık RVOL < 1.25 doğrudan sahte hareket değildir.
        is_volume_fake = (
            (rvol < 0.65)
            and
            (z_flow < 0.8)
        )

        is_illiquid = (
            value_traded < 25_000_000.0
        )

        # ============================================================
        # 4. SHOCK CONCORDANCE
        # ============================================================

        shock_count = sum(
            [
                z_vol >= _num(
                    dynamic_thresholds.get(
                        "th_vol"
                    ),
                    1.5
                ),

                z_range >= _num(
                    dynamic_thresholds.get(
                        "th_range"
                    ),
                    1.5
                ),

                z_flow >= _num(
                    dynamic_thresholds.get(
                        "th_flow"
                    ),
                    2.0
                ),

                z_lambda >= _num(
                    dynamic_thresholds.get(
                        "th_lambda"
                    ),
                    1.2
                ),
            ]
        )

        # Eski 1.25 / 1.50 çarpanları skorları 99.5'e yapıştırıyordu.
        concordance_bonus = min(
            shock_count * 4.0,
            16.0
        )

        is_fresh_shock = (
            perf_1m <= 22.0
            and
            perf_3m >= -10.0
        )

        is_downtrend_knife = (
            perf_3m < -25.0
            and
            z_vol < 2.5
            and
            change <= 0.0
        )

        item.update(
            {
                "z_vol": z_vol,
                "z_range": z_range,
                "z_lambda": z_lambda,
                "z_flow": z_flow,
                "shock_count": shock_count,
                "concordance_bonus": concordance_bonus,
                "entry_bonus": entry_bonus,
                "entry_status": entry_status,
                "is_fresh_shock": is_fresh_shock,
                "is_downtrend_knife": is_downtrend_knife,
                "is_volume_fake": is_volume_fake,
                "is_illiquid": is_illiquid,
            }
        )

        scored_data.append(item)

    res_df = pd.DataFrame(
        scored_data
    )

    if res_df.empty:
        return res_df

    # ================================================================
    # 5. MARKET-RELATIVE RESILIENCE
    # ================================================================

    change_s = pd.to_numeric(
        res_df["change_%"],
        errors="coerce"
    ).fillna(0.0)

    perf_1m_s = pd.to_numeric(
        res_df.get(
            "perf_1m",
            0.0
        ),
        errors="coerce"
    ).fillna(0.0)

    perf_3m_s = pd.to_numeric(
        res_df.get(
            "perf_3m",
            0.0
        ),
        errors="coerce"
    ).fillna(0.0)

    rvol_s = pd.to_numeric(
        res_df.get(
            "rvol",
            1.0
        ),
        errors="coerce"
    ).fillna(1.0)

    flow_s = pd.to_numeric(
        res_df["z_flow"],
        errors="coerce"
    ).fillna(0.0)

    market_median_change = float(
        change_s.median()
    )

    market_mean_change = float(
        change_s.mean()
    )

    market_red_ratio = (
        float(
            (change_s < 0).mean()
        )
        if len(change_s)
        else 0.5
    )

    res_df["market_median_change"] = round(
        market_median_change,
        3
    )

    res_df["market_mean_change"] = round(
        market_mean_change,
        3
    )

    res_df["market_red_ratio"] = round(
        market_red_ratio,
        4
    )

    # Hissenin piyasa medyanından ayrışması.
    res_df["excess_return"] = (
        change_s
        -
        market_median_change
    ).round(3)

    res_df["excess_vs_mean"] = (
        change_s
        -
        market_mean_change
    ).round(3)

    # ================================================================
    # 6. CROSS-SECTIONAL PERCENTILES
    # ================================================================

    rel_daily_pct = _pct_rank(
        change_s
    )

    excess_pct = _pct_rank(
        res_df["excess_return"]
    )

    rel_1m_pct = _pct_rank(
        perf_1m_s
    )

    rel_3m_pct = _pct_rank(
        perf_3m_s
    )

    rel_rvol_pct = _pct_rank(
        rvol_s
    )

    rel_flow_pct = _pct_rank(
        flow_s
    )

    high_s = pd.to_numeric(
        res_df["high"],
        errors="coerce"
    ).fillna(0.0)

    low_s = pd.to_numeric(
        res_df["low"],
        errors="coerce"
    ).fillna(0.0)

    close_s = pd.to_numeric(
        res_df["close"],
        errors="coerce"
    ).fillna(0.0)

    range_s = (
        high_s - low_s
    ).replace(
        0.0,
        np.nan
    )

    close_location = (
        (
            (close_s - low_s)
            /
            range_s
        )
        * 100.0
    ).replace(
        [np.inf, -np.inf],
        np.nan
    ).fillna(50.0)

    close_location = close_location.clip(
        0.0,
        100.0
    )

    # Geçmiş trend devamlılığı.
    trend_persistence = (
        rel_1m_pct * 0.45
        +
        rel_3m_pct * 0.55
    )

    # ================================================================
    # 7. RESILIENCE SCORE
    # ================================================================

    resilience_score = (

        rel_daily_pct * 0.35

        +

        excess_pct * 0.15

        +

        trend_persistence * 0.10

        +

        rel_rvol_pct * 0.10

        +

        rel_flow_pct * 0.10

        +

        close_location * 0.10

        +

        rel_3m_pct * 0.10
    )

    res_df["rel_daily_pct"] = (
        rel_daily_pct.round(1)
    )

    res_df["rel_1m_pct"] = (
        rel_1m_pct.round(1)
    )

    res_df["rel_3m_pct"] = (
        rel_3m_pct.round(1)
    )

    res_df["rel_rvol_pct"] = (
        rel_rvol_pct.round(1)
    )

    res_df["rel_flow_pct"] = (
        rel_flow_pct.round(1)
    )

    res_df["trend_persistence"] = (
        trend_persistence.round(1)
    )

    res_df["close_location"] = (
        close_location.round(1)
    )

    res_df["resilience_score"] = (
        resilience_score.round(1)
    )

    # ================================================================
    # 8. CRASH SURVIVOR SINIFLANDIRMASI
    # ================================================================

    # Piyasa düşerken pozitif kalan ve piyasa medyanını aşan hisseler.
    crash_resilient = (
        (change_s > 0.0)
        &
        (
            res_df["excess_return"]
            > 0.0
        )
        &
        (
            resilience_score
            >= 60.0
        )
        &
        (
            res_df["value_traded"]
            >= 25_000_000.0
        )
    )

    # Daha güçlü crash lideri.
    crash_leader = (
        crash_resilient
        &
        (
            resilience_score
            >= 70.0
        )
        &
        (
            rel_daily_pct
            >= 70.0
        )
    )

    res_df["crash_resilient"] = (
        crash_resilient
    )

    res_df["crash_survivor"] = (
        crash_leader
    )

    # Büyük hareket + pozitif geçmiş trend + piyasa üstü performans.
    extended_but_supported = (
        (change_s > 6.0)
        &
        (perf_1m_s > 0.0)
        &
        (perf_3m_s > 0.0)
        &
        (rvol_s >= 1.0)
        &
        (
            res_df["excess_return"]
            > 0.0
        )
    )

    res_df["extended_but_supported"] = (
        extended_but_supported
    )

    # ================================================================
    # 9. BASE SHOCK SCORE
    # ================================================================

    w_v = _num(
        dynamic_weights.get(
            "vol"
        ),
        0.35
    )

    w_f = _num(
        dynamic_weights.get(
            "flow"
        ),
        0.35
    )

    w_r = _num(
        dynamic_weights.get(
            "range"
        ),
        0.20
    )

    w_l = _num(
        dynamic_weights.get(
            "lambda"
        ),
        0.10
    )

    weight_sum = max(
        w_v + w_f + w_r + w_l,
        1e-9
    )

    w_v, w_f, w_r, w_l = [
        x / weight_sum
        for x in (
            w_v,
            w_f,
            w_r,
            w_l
        )
    ]

    pct_vol = _pct_rank(
        res_df["z_vol"]
    )

    pct_range = _pct_rank(
        res_df["z_range"]
    )

    pct_lambda = _pct_rank(
        res_df["z_lambda"]
    )

    pct_flow = _pct_rank(
        res_df["z_flow"]
    )

    base_score = (

        pct_vol * w_v

        +

        pct_flow * w_f

        +

        pct_range * w_r

        +

        pct_lambda * w_l

        +

        res_df["concordance_bonus"]
    )

    base_score = (
        base_score
        -
        np.where(
            res_df["is_volume_fake"],
            8.0,
            0.0
        )
    )

    base_score = (
        base_score
        -
        np.where(
            res_df["is_illiquid"],
            10.0,
            0.0
        )
    )

    base_score = (
        base_score
        +
        res_df["entry_bonus"]
    )

    # ================================================================
    # 10. REGIME BLENDING
    # ================================================================

    learned_resilience_weight = _num(
        dynamic_weights.get(
            "resilience_weight"
        ),
        0.70
    )

    learned_resilience_weight = float(
        np.clip(
            learned_resilience_weight,
            0.55,
            0.80
        )
    )

    if market_is_crashing:

        # Crash gününde resilience ana motor.
        raw_confidence = (

            base_score
            *
            (
                1.0
                -
                learned_resilience_weight
            )

            +

            resilience_score
            *
            learned_resilience_weight
        )

        # Piyasa üstü hisselere küçük bonus.
        raw_confidence += np.where(
            res_df["excess_return"] > 0.0,
            4.0,
            -8.0
        )

        # Pozitif kapanışa bonus.
        raw_confidence += np.where(
            change_s > 0.0,
            3.0,
            -18.0
        )

        # Likidite cezası var, fakat tamamen veto yok.
        raw_confidence -= np.where(
            res_df["is_illiquid"],
            12.0,
            0.0
        )

        raw_confidence -= np.where(
            res_df["is_volume_fake"],
            8.0,
            0.0
        )

    else:

        # Normal rejimde klasik shock motoru daha etkili.
        raw_confidence = (
            base_score * 0.75
            +
            resilience_score * 0.25
        )

    final_score = np.clip(
        np.round(
            raw_confidence,
            1
        ),
        0.0,
        99.5
    )

    # ================================================================
    # 11. FINAL SIGNAL
    # ================================================================

    valid_long_candidate = (
        (change_s > 0.0)
        &
        (~res_df["is_downtrend_knife"])
    )

    # Eski günlük OHLC -> VWAP hesabı artık sinyali veto etmiyor.
    res_df["shock_score"] = np.where(
        valid_long_candidate,
        final_score,
        0.0
    )

    res_df["confidence_score"] = (
        res_df["shock_score"]
    )

    # ================================================================
    # 12. POSITION ALLOCATION
    # ================================================================

    def assign_allocation(row):

        score = _num(
            row.get(
                "shock_score"
            ),
            0.0
        )

        resilience = _num(
            row.get(
                "resilience_score"
            ),
            0.0
        )

        resilient = bool(
            row.get(
                "crash_resilient",
                False
            )
        )

        leader = bool(
            row.get(
                "crash_survivor",
                False
            )
        )

        if market_is_crashing:

            if (
                leader
                and
                score >= 88.0
            ):
                return (
                    "⭐⭐⭐⭐⭐",
                    "Portföyün %10 - %15'i "
                    "(Crash Leader)"
                )

            if (
                resilient
                and
                score >= 82.0
            ):
                return (
                    "⭐⭐⭐⭐",
                    "Portföyün %6 - %10'u "
                    "(Crash Survivor)"
                )

            if (
                resilient
                and
                score >= 75.0
            ):
                return (
                    "⭐⭐⭐",
                    "Portföyün %2 - %5'i "
                    "(İzleme / Kademeli)"
                )

            return (
                "⭐",
                "İşlem Açma "
                "(Crash Risk / Yetersiz Dayanıklılık)"
            )

        if (
            score >= 85.0
            and
            resilience >= 75.0
        ):
            return (
                "⭐⭐⭐⭐⭐",
                "Portföyün %10 - %15'i "
                "(Yüksek Göreli Güç)"
            )

        if (
            score >= 75.0
            and
            resilience >= 60.0
        ):
            return (
                "⭐⭐⭐⭐",
                "Portföyün %6 - %10'u "
                "(Dengeli Güven)"
            )

        if score >= 65.0:
            return (
                "⭐⭐⭐",
                "Portföyün %2 - %5'i "
                "(Küçük Deneme)"
            )

        return (
            "⭐",
            "İşlem Açma "
            "(Yetersiz Güven)"
        )

    allocations = [
        assign_allocation(row)
        for _, row
        in res_df.iterrows()
    ]

    res_df["stars"] = [
        x[0]
        for x in allocations
    ]

    res_df["allocation"] = [
        x[1]
        for x in allocations
    ]

    # Yardımcı kolonları kaldır.
    # Relative strength teşhis kolonları korunur.
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

    res_df = res_df.drop(
        columns=[
            c
            for c in drop_cols
            if c in res_df.columns
        ]
    )

    return res_df.sort_values(
        by=[
            "shock_score",
            "resilience_score",
            "excess_return"
        ],
        ascending=[
            False,
            False,
            False
        ]
    ).reset_index(
        drop=True
    )
