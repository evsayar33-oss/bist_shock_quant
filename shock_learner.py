import json
import os

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


SIGNAL_LOG_FILE = "shock_signals_log.csv"
AI_STATE_FILE = "shock_ai_state.json"


DEFAULT_WEIGHTS = {
    "vol": 0.35,
    "flow": 0.35,
    "range": 0.20,
    "lambda": 0.10,
}

DEFAULT_RESILIENCE_WEIGHT = 0.70


def load_signal_history():

    if not os.path.exists(
        SIGNAL_LOG_FILE
    ):
        return pd.DataFrame()

    try:

        df = pd.read_csv(
            SIGNAL_LOG_FILE
        )

        if "tarih" in df.columns:
            df["tarih"] = pd.to_datetime(
                df["tarih"],
                errors="coerce"
            )

        return df

    except Exception:
        return pd.DataFrame()


def _ensure_columns(
    df,
    columns
):

    for col in columns:

        if col not in df.columns:
            df[col] = np.nan

    return df


def _safe_float(
    value,
    default=0.0
):

    try:

        value = float(value)

        if not np.isfinite(value):
            return default

        return value

    except Exception:
        return default


def log_shock_signals(
    top_df
):

    if top_df is None or top_df.empty:
        return

    cols = [

        "ticker",
        "close",
        "shock_score",

        "z_vol",
        "z_range",
        "z_flow",
        "z_lambda",

        "resilience_score",
        "excess_return",

        "rel_1m_pct",
        "rel_3m_pct",
        "trend_persistence",

        "crash_resilient",
        "crash_survivor",

        "tarih",
    ]

    signals = top_df.head(
        10
    ).copy()

    signals = _ensure_columns(
        signals,
        cols
    )

    signals = signals[
        cols
    ].copy()

    signals["realized_3d"] = np.nan
    signals["realized_5d"] = np.nan

    history_df = load_signal_history()

    today_val = (
        pd.Timestamp.now()
        .normalize()
    )

    if not history_df.empty:

        history_df = _ensure_columns(
            history_df,
            signals.columns.tolist()
        )

        history_df = history_df[
            history_df["tarih"]
            !=
            today_val
        ]

        updated = pd.concat(
            [
                history_df,
                signals
            ],
            ignore_index=True
        )

    else:

        updated = signals

    updated.to_csv(
        SIGNAL_LOG_FILE,
        index=False
    )


def update_realized_shock_returns(
    current_market_df
):

    history_df = load_signal_history()

    if (
        history_df.empty
        or
        current_market_df is None
        or
        current_market_df.empty
    ):
        return history_df

    price_map = dict(
        zip(
            current_market_df[
                "ticker"
            ],
            current_market_df[
                "close"
            ]
        )
    )

    today = (
        pd.Timestamp.now()
        .normalize()
    )

    history_df = _ensure_columns(
        history_df,
        [
            "realized_3d",
            "realized_5d"
        ]
    )

    for idx, row in history_df.iterrows():

        sig_date = pd.to_datetime(
            row.get("tarih"),
            errors="coerce"
        )

        if pd.isna(sig_date):
            continue

        days_passed = (
            today
            -
            sig_date.normalize()
        ).days

        ticker = row.get(
            "ticker"
        )

        entry_price = _safe_float(
            row.get("close"),
            0.0
        )

        if (
            ticker in price_map
            and
            entry_price > 0
        ):

            current_price = _safe_float(
                price_map[ticker],
                0.0
            )

            gain = (
                (
                    current_price
                    -
                    entry_price
                )
                /
                entry_price
            ) * 100.0

            if (
                days_passed >= 3
                and
                pd.isna(
                    history_df.at[
                        idx,
                        "realized_3d"
                    ]
                )
            ):

                history_df.at[
                    idx,
                    "realized_3d"
                ] = round(
                    gain,
                    2
                )

            if (
                days_passed >= 5
                and
                pd.isna(
                    history_df.at[
                        idx,
                        "realized_5d"
                    ]
                )
            ):

                history_df.at[
                    idx,
                    "realized_5d"
                ] = round(
                    gain,
                    2
                )

    history_df.to_csv(
        SIGNAL_LOG_FILE,
        index=False
    )

    return history_df


def compute_dynamic_market_thresholds(
    df
):

    if df is None or df.empty:

        return {
            "th_vol": 1.5,
            "th_range": 1.5,
            "th_flow": 2.0,
            "th_lambda": 1.2,
        }

    def percentile(
        column,
        fallback
    ):

        if column not in df.columns:
            return fallback

        s = pd.to_numeric(
            df[column],
            errors="coerce"
        ).dropna()

        if s.empty:
            return fallback

        return float(
            np.percentile(
                s,
                85
            )
        )

    th_vol = percentile(
        "z_vol",
        1.5
    )

    th_range = percentile(
        "z_range",
        1.5
    )

    th_flow = percentile(
        "z_flow",
        2.0
    )

    th_lambda = percentile(
        "z_lambda",
        1.2
    )

    return {

        "th_vol": round(
            max(th_vol, 1.2),
            2
        ),

        "th_range": round(
            max(th_range, 1.0),
            2
        ),

        "th_flow": round(
            max(th_flow, 1.5),
            2
        ),

        "th_lambda": round(
            max(th_lambda, 0.5),
            2
        ),
    }


def calibrate_adaptive_weights():

    history_df = load_signal_history()

    if (
        history_df.empty
        or
        "realized_3d"
        not in history_df.columns
    ):

        return (
            DEFAULT_WEIGHTS.copy(),
            "🕒 ÖĞRENME EVRESİNDE "
            "(Örneklem Bekleniyor)"
        )

    valid = history_df.dropna(
        subset=[
            "realized_3d"
        ]
    ).copy()

    if len(valid) < 15:

        return (
            DEFAULT_WEIGHTS.copy(),
            "🕒 ÖĞRENME EVRESİNDE "
            "(Örneklem Bekleniyor)"
        )

    factors = [
        "z_vol",
        "z_flow",
        "z_range",
        "z_lambda"
    ]

    ic_scores = {}

    y = pd.to_numeric(
        valid["realized_3d"],
        errors="coerce"
    ).values

    for factor in factors:

        if factor not in valid.columns:

            ic_scores[
                factor
            ] = 0.05

            continue

        x = pd.to_numeric(
            valid[factor],
            errors="coerce"
        ).values

        mask = (
            np.isfinite(x)
            &
            np.isfinite(y)
        )

        if (
            mask.sum() >= 8
            and
            np.std(x[mask]) > 0
            and
            np.std(y[mask]) > 0
        ):

            corr, _ = spearmanr(
                x[mask],
                y[mask]
            )

            ic_scores[
                factor
            ] = max(
                float(corr)
                if np.isfinite(corr)
                else 0.05,
                0.05
            )

        else:

            ic_scores[
                factor
            ] = 0.05

    total = sum(
        ic_scores.values()
    )

    if total <= 0:

        return (
            DEFAULT_WEIGHTS.copy(),
            "🕒 ÖĞRENME EVRESİNDE "
            "(Geçerli IC Yok)"
        )

    raw_weights = {

        "vol":
            ic_scores["z_vol"]
            /
            total,

        "flow":
            ic_scores["z_flow"]
            /
            total,

        "range":
            ic_scores["z_range"]
            /
            total,

        "lambda":
            ic_scores["z_lambda"]
            /
            total,
    }

    # Bayesian shrinkage.
    final_w = {

        k:
        round(
            0.60
            *
            DEFAULT_WEIGHTS[k]
            +
            0.40
            *
            raw_weights[k],
            4
        )

        for k in DEFAULT_WEIGHTS
    }

    w_sum = sum(
        final_w.values()
    )

    final_w = {
        k:
        round(
            v / w_sum,
            4
        )

        for k, v
        in final_w.items()
    }

    status = (
        "🧠 AI ÖZ-ÖĞRENME AKTİF "
        f"(Eğitilen Sinyal: {len(valid)})"
    )

    return (
        final_w,
        status
    )


def calibrate_resilience_weight(
    default=DEFAULT_RESILIENCE_WEIGHT
):
    """
    Relative resilience'in gerçekleşen 3 günlük getiriyle ilişkisini ölçer.

    Adaptasyon sınırlıdır.
    Böylece az sayıdaki işlem bütün crash mimarisini bozamaz.
    """

    history_df = load_signal_history()

    required = [
        "realized_3d",
        "resilience_score"
    ]

    if (
        history_df.empty
        or
        any(
            c not in history_df.columns
            for c in required
        )
    ):

        return (
            float(default),
            "🛡️ RESILIENCE AĞIRLIĞI: "
            "VARSAYILAN"
        )

    valid = history_df.dropna(
        subset=required
    ).copy()

    if len(valid) < 20:

        return (
            float(default),
            "🛡️ RESILIENCE AĞIRLIĞI: "
            f"ÖRNEKLEM BEKLENİYOR "
            f"({len(valid)}/20)"
        )

    x = pd.to_numeric(
        valid["resilience_score"],
        errors="coerce"
    ).values

    y = pd.to_numeric(
        valid["realized_3d"],
        errors="coerce"
    ).values

    mask = (
        np.isfinite(x)
        &
        np.isfinite(y)
    )

    if (
        mask.sum() < 20
        or
        np.std(x[mask]) == 0
        or
        np.std(y[mask]) == 0
    ):

        return (
            float(default),
            "🛡️ RESILIENCE AĞIRLIĞI: "
            "VARSAYILAN"
        )

    corr, _ = spearmanr(
        x[mask],
        y[mask]
    )

    corr = (
        float(corr)
        if np.isfinite(corr)
        else 0.0
    )

    # Merkez ağırlık 0.68.
    # Adaptasyon sadece 0.55 - 0.80 arasında.
    weight = (
        0.68
        +
        (
            0.12
            *
            float(
                np.clip(
                    corr,
                    -0.5,
                    0.5
                )
            )
        )
    )

    weight = float(
        np.clip(
            weight,
            0.55,
            0.80
        )
    )

    return (
        round(weight, 3),
        "🛡️ RESILIENCE AĞIRLIĞI: "
        f"{weight:.3f} "
        f"(Spearman {corr:+.3f})"
    )
