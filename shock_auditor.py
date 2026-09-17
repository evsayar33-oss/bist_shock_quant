import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from shock_fetcher import (
    get_bist_raw_data
)


AI_STATE_FILE = "shock_ai_state.json"
LEDGER_FILE = "backtest_ledger.csv"

MIN_BACKTEST_SAMPLES = 25


def send_telegram_audit(
    message
):

    token = os.environ.get(
        "TELEGRAM_TOKEN"
    )

    chat_id = os.environ.get(
        "CHAT_ID"
    )

    if not token or not chat_id:
        return

    url = (
        "https://api.telegram.org/"
        "bot"
        f"{token}"
        "/sendMessage"
    )

    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:

        res = requests.post(
            url,
            json=payload,
            timeout=15
        )

        if not res.json().get(
            "ok"
        ):

            payload.pop(
                "parse_mode",
                None
            )

            requests.post(
                url,
                json=payload,
                timeout=15
            )

    except Exception as exc:

        print(
            f"Hata: {exc}"
        )


def update_ledger_returns(
    df_close
):

    if not os.path.exists(
        LEDGER_FILE
    ):
        return pd.DataFrame()

    try:

        df_ledger = pd.read_csv(
            LEDGER_FILE
        )

    except Exception:

        return pd.DataFrame()

    if df_ledger.empty:
        return df_ledger

    required = [

        "price_d1",
        "price_d3",
        "price_d5",

        "return_d1",
        "return_d3",
        "return_d5",

        "is_completed",
    ]

    for col in required:

        if col not in df_ledger.columns:

            if col == "is_completed":

                df_ledger[
                    col
                ] = 0

            else:

                df_ledger[
                    col
                ] = np.nan

    close_map = dict(
        zip(
            df_close[
                "ticker"
            ],
            df_close[
                "close"
            ]
        )
    )

    bugun = (
        datetime.now().date()
    )

    for idx, row in (
        df_ledger.iterrows()
    ):

        ticker = row.get(
            "ticker"
        )

        if ticker not in close_map:
            continue

        try:

            curr_p = float(
                close_map[ticker]
            )

            entry_p = float(
                row["entry_price"]
            )

            row_date = (
                datetime.strptime(
                    str(row["date"]),
                    "%Y-%m-%d"
                ).date()
            )

        except Exception:

            continue

        if entry_p <= 0:
            continue

        days_passed = (
            bugun
            -
            row_date
        ).days

        ret = (
            (
                curr_p
                -
                entry_p
            )
            /
            entry_p
        ) * 100.0

        if (
            days_passed >= 1
            and
            pd.isna(
                df_ledger.at[
                    idx,
                    "price_d1"
                ]
            )
        ):

            df_ledger.at[
                idx,
                "price_d1"
            ] = curr_p

            df_ledger.at[
                idx,
                "return_d1"
            ] = round(
                ret,
                2
            )

        if (
            days_passed >= 3
            and
            pd.isna(
                df_ledger.at[
                    idx,
                    "price_d3"
                ]
            )
        ):

            df_ledger.at[
                idx,
                "price_d3"
            ] = curr_p

            df_ledger.at[
                idx,
                "return_d3"
            ] = round(
                ret,
                2
            )

        if (
            days_passed >= 5
            and
            pd.isna(
                df_ledger.at[
                    idx,
                    "price_d5"
                ]
            )
        ):

            df_ledger.at[
                idx,
                "price_d5"
            ] = curr_p

            df_ledger.at[
                idx,
                "return_d5"
            ] = round(
                ret,
                2
            )

            df_ledger.at[
                idx,
                "is_completed"
            ] = 1

    df_ledger.to_csv(
        LEDGER_FILE,
        index=False
    )

    return df_ledger


def _audit_metric(
    trades
):

    if trades.empty:

        return {
            "n": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_return": 0.0,
            "score": -999.0,
        }

    ret = pd.to_numeric(
        trades[
            "return_d5"
        ],
        errors="coerce"
    ).dropna()

    if ret.empty:

        return {
            "n": 0,
            "win_rate": 0.0,
            "profit_factor": 0.0,
            "avg_return": 0.0,
            "score": -999.0,
        }

    gains = ret[
        ret > 0
    ]

    losses = ret[
        ret < 0
    ].abs()

    win_rate = float(
        (
            ret > 0
        ).mean()
        *
        100.0
    )

    if losses.sum() > 0:

        profit_factor = float(
            gains.sum()
            /
            losses.sum()
        )

    else:

        profit_factor = 5.0

    avg_return = float(
        ret.mean()
    )

    # Küçük örneklemin modeli
    # aşırı etkilemesini engelle.
    sample_factor = min(
        len(ret) / 30.0,
        1.0
    )

    score = (
        avg_return * 0.40
        +
        np.log1p(
            profit_factor
        ) * 2.0
        +
        win_rate * 0.02
    ) * sample_factor

    return {

        "n":
            int(len(ret)),

        "win_rate":
            win_rate,

        "profit_factor":
            profit_factor,

        "avg_return":
            avg_return,

        "score":
            score,
    }


def run_threshold_audit(
    completed_trades
):

    """
    Üretimde gerçekten kullanılan
    initial_score üzerinden audit.

    Eski sistemde auditor farklı bir
    z-score formülü kullanarak üretim
    skorunu taklit etmeye çalışıyordu.

    Bu yanlıştı.

    Artık gerçek initial_score
    doğrudan test ediliyor.
    """

    if (
        completed_trades is None
        or
        completed_trades.empty
    ):
        return None

    if (
        "initial_score"
        not in completed_trades.columns
        or
        "return_d5"
        not in completed_trades.columns
    ):
        return None

    data = (
        completed_trades.copy()
    )

    data[
        "initial_score"
    ] = pd.to_numeric(
        data[
            "initial_score"
        ],
        errors="coerce"
    )

    data[
        "return_d5"
    ] = pd.to_numeric(
        data[
            "return_d5"
        ],
        errors="coerce"
    )

    data = data.dropna(
        subset=[
            "initial_score",
            "return_d5"
        ]
    )

    if data.empty:
        return None

    best = None

    # Gerçek production score
    # 65 - 95 arasında threshold audit.
    for threshold in np.arange(
        65.0,
        96.0,
        2.0
    ):

        selected = data[
            data[
                "initial_score"
            ]
            >=
            threshold
        ]

        metrics = _audit_metric(
            selected
        )

        # Çok küçük sample
        # dikkate alınmaz.
        if metrics["n"] < 8:
            continue

        candidate = {

            "min_score":
                float(threshold),

            **metrics
        }

        if (
            best is None
            or
            candidate["score"]
            >
            best["score"]
        ):

            best = candidate

    return best


def run_evening_audit():

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        "BIST Kapanış Denetimi Başlatılıyor..."
    )

    try:

        df_close = (
            get_bist_raw_data()
        )

    except Exception:

        df_close = (
            pd.DataFrame()
        )

    if df_close.empty:
        return

    # ================================================================
    # 1. LEDGER GETİRİLERİNİ GÜNCELLE
    # ================================================================

    df_ledger = (
        update_ledger_returns(
            df_close
        )
    )

    completed = (

        df_ledger[
            df_ledger[
                "is_completed"
            ]
            == 1
        ]

        if (
            not df_ledger.empty
            and
            "is_completed"
            in
            df_ledger.columns
        )

        else
        pd.DataFrame()
    )

    completed_count = len(
        completed
    )

    # ================================================================
    # 2. GERÇEK SCORE AUDIT
    # ================================================================

    best = None

    if (
        completed_count
        >=
        MIN_BACKTEST_SAMPLES
    ):

        best = (
            run_threshold_audit(
                completed
            )
        )

    backtest_msg = ""

    if best is not None:

        try:

            if os.path.exists(
                AI_STATE_FILE
            ):

                with open(
                    AI_STATE_FILE,
                    "r"
                ) as f:

                    state = json.load(
                        f
                    )

            else:

                state = {}

        except Exception:

            state = {}

        state.setdefault(
            "thresholds",
            {}
        )

        state.setdefault(
            "weights",
            {}
        )

        state[
            "thresholds"
        ][
            "min_score"
        ] = best[
            "min_score"
        ]

        # Mevcut production ağırlıklarını
        # koru. Auditor burada yeni
        # sahte ağırlıklar üretmiyor.
        state[
            "weights"
        ].setdefault(
            "vol",
            0.35
        )

        state[
            "weights"
        ].setdefault(
            "flow",
            0.35
        )

        state[
            "weights"
        ].setdefault(
            "range",
            0.20
        )

        state[
            "weights"
        ].setdefault(
            "lambda",
            0.10
        )

        state.setdefault(
            "resilience_weight",
            0.70
        )

        state[
            "status"
        ] = (
            "🧪 GERÇEK SCORE "
            "AUDIT AKTİF "
            f"(N={best['n']} "
            f"| WR=%"
            f"{best['win_rate']:.1f} "
            f"| PF="
            f"{best['profit_factor']:.2f})"
        )

        with open(
            AI_STATE_FILE,
            "w"
        ) as f:

            json.dump(
                state,
                f,
                indent=4,
                ensure_ascii=False
            )

        backtest_msg = (

            "🧪 <b>GERÇEK SCORE "
            "BACKTEST AUDITI:</b>\n"

            f"• <i>"
            f"{completed_count} "
            f"tamamlanmış işlem "
            f"incelendi.</i>\n"

            f"• Önerilen minimum skor: "
            f"<b>"
            f"{best['min_score']:.1f}"
            f"</b>\n"

            f"• Örneklem: "
            f"<b>{best['n']}</b> "
            f"| Win Rate: "
            f"<b>%"
            f"{best['win_rate']:.1f}"
            f"</b>\n"

            f"• Profit Factor: "
            f"<b>"
            f"{best['profit_factor']:.2f}"
            f"</b> | "

            f"Ortalama 5G: "
            f"<b>%"
            f"{best['avg_return']:+.2f}"
            f"</b>\n"
        )

    else:

        kalan = max(
            MIN_BACKTEST_SAMPLES
            -
            completed_count,
            0
        )

        backtest_msg = (

            "⏳ <b>BIST BACKTEST "
            "DEFTER İLERLEMESİ:</b>\n"

            f"• <i>Tamamlanmış: "
            f"<b>"
            f"{completed_count}"
            f" / "
            f"{MIN_BACKTEST_SAMPLES}"
            f"</b></i>\n"

            f"• <i>Threshold audit "
            f"için yaklaşık "
            f"{kalan} işlem daha "
            f"gerekiyor.</i>\n"
        )

    # ================================================================
    # 3. RAPOR
    # ================================================================

    rep = (
        "🔬 <b>BIST GÖRELI GÜÇ / "
        "CRASH SHIELD "
        "DENETİM RAPORU</b>\n"
    )

    rep += (
        f"🗓 <i>"
        f"{datetime.now().strftime('%Y-%m-%d')}"
        f" | Seans Kapanışı</i>\n"
    )

    rep += (
        "━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    rep += backtest_msg

    send_telegram_audit(
        rep
    )

    print(
        "BIST Denetim raporu iletildi."
    )


if __name__ == "__main__":
    run_evening_audit()
