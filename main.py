import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from shock_fetcher import fetch_all_data

from shock_engine import (
    calculate_shock_scores,
    gecmis_veriyi_yukle,
    GECMIS_DOSYA
)

from shock_learner import (
    update_realized_shock_returns,
    compute_dynamic_market_thresholds,
    calibrate_adaptive_weights,
    calibrate_resilience_weight,
    AI_STATE_FILE
)


LEDGER_FILE = "backtest_ledger.csv"


def send_telegram_message(
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

    max_len = 3800
    messages = []

    if len(message) > max_len:

        parts = message.split(
            "\n\n"
        )

        current_msg = ""

        for part in parts:

            if (
                len(current_msg)
                +
                len(part)
                +
                2
                <
                max_len
            ):

                current_msg += (
                    part
                    +
                    "\n\n"
                )

            else:

                if current_msg.strip():
                    messages.append(
                        current_msg.strip()
                    )

                current_msg = (
                    part
                    +
                    "\n\n"
                )

        if current_msg.strip():
            messages.append(
                current_msg.strip()
            )

    else:

        messages = [
            message
        ]

    for msg in messages:

        payload = {
            "chat_id": chat_id,
            "text": msg,
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
                f"Telegram hatası: {exc}"
            )


def assess_bist_market_regime(
    df_market
):

    """
    Cross-sectional BIST crash detector.

    Ortalama + medyan + düşen oranı
    birlikte kullanılır.
    """

    if (
        df_market is None
        or
        df_market.empty
    ):
        return (
            False,
            "NÖTR"
        )

    changes = pd.to_numeric(
        df_market["change_%"],
        errors="coerce"
    ).fillna(0.0)

    total_stocks = len(
        changes
    )

    red_stocks = int(
        (
            changes < 0
        ).sum()
    )

    red_ratio = (
        red_stocks
        /
        total_stocks
        if total_stocks
        else 0.5
    )

    avg_market_change = float(
        changes.mean()
    )

    median_market_change = float(
        changes.median()
    )

    # Crash tespiti:
    #
    # - Hisselerin %70'i kırmızı
    # - veya ortalama <= -1.5
    # - veya medyan <= -1.0
    #
    # Median eklenmesi birkaç büyük hissenin
    # ortalamayı bozmasını engeller.

    if (
        red_ratio >= 0.70
        or
        avg_market_change <= -1.5
        or
        median_market_change <= -1.0
    ):

        regime_status = (
            "🚨 PİYASA ÇÖKÜŞ REJİMİ "
            f"(Düşen: "
            f"%{red_ratio * 100:.0f} "
            f"| Ort: "
            f"%{avg_market_change:+.2f} "
            f"| Medyan: "
            f"%{median_market_change:+.2f})"
        )

        return (
            True,
            regime_status
        )

    if (
        red_ratio <= 0.40
        and
        avg_market_change >= 0.8
    ):

        regime_status = (
            "🚀 BOĞA REJİMİ "
            "(Yükselen Hisseler Hakim "
            f"| Ort: "
            f"%{avg_market_change:+.2f})"
        )

        return (
            False,
            regime_status
        )

    regime_status = (
        "⚖️ NÖTR / DALGALI PİYASA "
        f"(Kırmızı: "
        f"%{red_ratio * 100:.0f} "
        f"| Ort: "
        f"%{avg_market_change:+.2f} "
        f"| Medyan: "
        f"%{median_market_change:+.2f})"
    )

    return (
        False,
        regime_status
    )


def check_bist_earnings_risk(
    ticker
):

    try:

        t = yf.Ticker(
            f"{ticker}.IS"
        )

        cal = t.calendar

        if cal is None:
            return (
                False,
                ""
            )

        ed = None

        if isinstance(
            cal,
            dict
        ):

            ed = cal.get(
                "Earnings Date"
            )

        elif hasattr(
            cal,
            "get"
        ):

            ed = cal.get(
                "Earnings Date"
            )

        elif (
            hasattr(
                cal,
                "loc"
            )
            and
            "Earnings Date"
            in cal.index
        ):

            ed = cal.loc[
                "Earnings Date"
            ].values

        if ed is not None:

            if not isinstance(
                ed,
                (
                    list,
                    np.ndarray,
                    tuple
                )
            ):

                ed = [ed]

            for d in ed:

                if pd.notna(d):

                    d_date = (
                        d.date()
                        if hasattr(
                            d,
                            "date"
                        )
                        else
                        pd.to_datetime(
                            d
                        ).date()
                    )

                    days_diff = (
                        d_date
                        -
                        datetime.now().date()
                    ).days

                    if (
                        0
                        <=
                        days_diff
                        <=
                        5
                    ):

                        return (
                            True,
                            f"{d_date.strftime('%d.%m')} "
                            f"({days_diff} Gün Kaldı)"
                        )

    except Exception:
        pass

    return (
        False,
        ""
    )


def generate_exit_signals(
    df_current
):

    if not os.path.exists(
        LEDGER_FILE
    ):
        return ""

    try:

        df_ledger = pd.read_csv(
            LEDGER_FILE
        )

    except Exception:

        return ""

    if (
        df_ledger.empty
        or
        "is_completed"
        not in df_ledger.columns
    ):
        return ""

    open_positions = (
        df_ledger[
            df_ledger[
                "is_completed"
            ]
            == 0
        ]
    )

    if open_positions.empty:
        return ""

    close_map = dict(
        zip(
            df_current[
                "ticker"
            ],
            df_current[
                "close"
            ]
        )
    )

    bugun = (
        datetime.now().date()
    )

    signals = []

    for _, row in (
        open_positions.iterrows()
    ):

        ticker = row[
            "ticker"
        ]

        if ticker not in close_map:
            continue

        curr_p = float(
            close_map[ticker]
        )

        entry_p = float(
            row["entry_price"]
        )

        row_date = datetime.strptime(
            str(row["date"]),
            "%Y-%m-%d"
        ).date()

        days_held = (
            bugun
            -
            row_date
        ).days

        pnl = (
            (
                curr_p
                -
                entry_p
            )
            /
            entry_p
        ) * 100.0

        if pnl <= -3.0:

            signals.append(
                f"🚨 <b>#{ticker} "
                f"STOP-LOSS "
                f"(ACİL ÇIKIŞ)!</b>\n"
                f"  ↳ <i>Giriş: "
                f"{entry_p:.2f} TL | "
                f"Güncel: "
                f"{curr_p:.2f} TL | "
                f"Zarar: "
                f"<b>%{pnl:+.2f}</b>\n"
                f"  🛑 Stop kırıldı, "
                f"zararı kes ve çık!</i>"
            )

        elif pnl >= 9.0:

            signals.append(
                f"💰 <b>#{ticker} "
                f"KÂR AL!</b>\n"
                f"  ↳ <i>Giriş: "
                f"{entry_p:.2f} TL | "
                f"Güncel: "
                f"{curr_p:.2f} TL | "
                f"Kâr: "
                f"<b>%{pnl:+.2f}</b>\n"
                f"  🎯 Pozisyonun "
                f"%50'sini sat, "
                f"kalanın stopunu "
                f"maliyete çek!</i>"
            )

        elif pnl >= 4.0:

            signals.append(
                f"🔒 <b>#{ticker} "
                f"KÂR KORUMA / "
                f"MALİYET STOPU</b>\n"
                f"  ↳ <i>Fiyat: "
                f"{curr_p:.2f} TL | "
                f"Kâr: "
                f"<b>%{pnl:+.2f}</b>\n"
                f"  🛡️ Stop maliyete "
                f"({entry_p:.2f} TL) "
                f"çekildi.</i>"
            )

        elif days_held >= 5:

            signals.append(
                f"⏰ <b>#{ticker} "
                f"1 HAFTALIK "
                f"VADE DOLDU</b>\n"
                f"  ↳ <i>Kapanış: "
                f"{curr_p:.2f} TL | "
                f"Net: "
                f"<b>%{pnl:+.2f}</b></i>"
            )

        else:

            signals.append(
                f"🟢 <b>#{ticker} "
                f"TAŞIMAYA DEVAM ET</b> "
                f"({days_held}. Gün)\n"
                f"  ↳ <i>Fiyat: "
                f"{curr_p:.2f} TL | "
                f"Durum: "
                f"<b>%{pnl:+.2f}</b></i>"
            )

    if not signals:
        return ""

    return (
        "🛡️ <b>BIST AÇIK POZİSYONLAR "
        "& ÇIKIŞ ALARMLARI "
        "(Exit Engine):</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        +
        "\n\n".join(
            signals
        )
        +
        "\n━━━━━━━━━━━━━━━━━━━━\n\n"
    )


def record_clean_ledger_entries(
    df_scored,
    market_is_crashing=False
):

    """
    Üretim backtest ledger.

    Crash gününde:

    RVOL >= 1.3 artık zorunlu değildir.

    Bunun yerine:

    - pozitif günlük getiri
    - piyasa medyanının üzerinde performans
    - resilience >= 60
    - likidite >= 25M
    - score >= 88

    kullanılır.
    """

    if (
        df_scored is None
        or
        df_scored.empty
    ):
        return

    bugun_str = (
        datetime.now()
        .strftime("%Y-%m-%d")
    )

    min_entry_score = (
        88.0
        if market_is_crashing
        else
        75.0
    )

    candidates = (
        df_scored[
            df_scored[
                "shock_score"
            ]
            >=
            min_entry_score
        ]
        .copy()
    )

    new_rows = []

    for _, r in (
        candidates.iterrows()
    ):

        entry_status = str(
            r.get(
                "entry_status",
                ""
            )
        )

        change = float(
            r.get(
                "change_%",
                0.0
            )
        )

        rvol_val = float(
            r.get(
                "rvol",
                1.0
            )
        )

        val_traded = float(
            r.get(
                "value_traded",
                0.0
            )
        )

        resilience = float(
            r.get(
                "resilience_score",
                0.0
            )
        )

        excess_return = float(
            r.get(
                "excess_return",
                0.0
            )
        )

        crash_resilient = bool(
            r.get(
                "crash_resilient",
                False
            )
        )

        if "BİLANÇO" in entry_status:
            continue

        if (
            val_traded
            <
            25_000_000.0
        ):
            continue

        if market_is_crashing:

            # Crash gününün esas filtresi.
            if change <= 0.0:
                continue

            if excess_return <= 0.0:
                continue

            if resilience < 60.0:
                continue

        else:

            if rvol_val < 1.3:
                continue

            if (
                change > 8.0
                and
                not bool(
                    r.get(
                        "extended_but_supported",
                        False
                    )
                )
            ):
                continue

        new_rows.append(
            {

                "date":
                    bugun_str,

                "ticker":
                    r["ticker"],

                "entry_price":
                    r["close"],

                "z_vol":
                    r.get(
                        "z_vol",
                        0.0
                    ),

                "z_range":
                    r.get(
                        "z_range",
                        0.0
                    ),

                "z_flow":
                    r.get(
                        "z_flow",
                        0.0
                    ),

                "z_lambda":
                    r.get(
                        "z_lambda",
                        0.0
                    ),

                "resilience_score":
                    resilience,

                "excess_return":
                    excess_return,

                "rel_1m_pct":
                    r.get(
                        "rel_1m_pct",
                        50.0
                    ),

                "rel_3m_pct":
                    r.get(
                        "rel_3m_pct",
                        50.0
                    ),

                "trend_persistence":
                    r.get(
                        "trend_persistence",
                        50.0
                    ),

                "crash_resilient":
                    crash_resilient,

                "crash_survivor":
                    bool(
                        r.get(
                            "crash_survivor",
                            False
                        )
                    ),

                "entry_status":
                    r.get(
                        "entry_status",
                        "NORMAL"
                    ),

                "initial_score":
                    r.get(
                        "shock_score",
                        0.0
                    ),

                "price_d1":
                    np.nan,

                "price_d3":
                    np.nan,

                "price_d5":
                    np.nan,

                "return_d1":
                    np.nan,

                "return_d3":
                    np.nan,

                "return_d5":
                    np.nan,

                "is_completed":
                    0,
            }
        )

    if not new_rows:
        return

    df_new = pd.DataFrame(
        new_rows
    )

    if os.path.exists(
        LEDGER_FILE
    ):

        try:

            df_old = pd.read_csv(
                LEDGER_FILE
            )

            if (
                "date"
                in
                df_old.columns
                and
                "ticker"
                in
                df_old.columns
            ):

                df_old = df_old[
                    ~(
                        (
                            df_old[
                                "date"
                            ]
                            ==
                            bugun_str
                        )
                        &
                        (
                            df_old[
                                "ticker"
                            ]
                            .isin(
                                df_new[
                                    "ticker"
                                ]
                            )
                        )
                    )
                ]

            df_final = pd.concat(
                [
                    df_old,
                    df_new
                ],
                ignore_index=True
            )

        except Exception:

            df_final = df_new

    else:

        df_final = df_new

    df_final.to_csv(
        LEDGER_FILE,
        index=False
    )


def format_shock_report(
    df_scored,
    exit_signals_text,
    market_regime_text,
    market_is_crashing=False,
    min_score=75.0
):

    effective_min = (
        88.0
        if market_is_crashing
        else
        min_score
    )

    shocks = (
        df_scored[
            df_scored[
                "shock_score"
            ]
            >=
            effective_min
        ]
        .sort_values(
            by=[
                "shock_score",
                "resilience_score"
            ],
            ascending=[
                False,
                False
            ]
        )
    )

    msg = (
        exit_signals_text
        or
        ""
    )

    msg += (
        f"⚡ <b>BIST GÖRELI GÜÇ / "
        f"ŞOK LİSTESİ "
        f"({effective_min:.1f}+)</b>\n"
    )

    msg += (
        f"🗓 <i>"
        f"{datetime.now().strftime('%Y-%m-%d')}"
        f" | Seans Raporu</i>\n"
    )

    msg += (
        f"🌐 <b>Piyasa Rejimi:</b> "
        f"<i>{market_regime_text}</i>\n"
    )

    msg += (
        "━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    if market_is_crashing:

        msg += (
            "⚠️ <b>CRASH SHIELD AKTİF:</b> "
            "Sistem artık yalnızca yüksek "
            "hacmi değil, piyasa medyanına "
            "göre direnç ve göreli gücü "
            "de ölçüyor.\n\n"
        )

    if shocks.empty:

        # Crash gününde hiçbir giriş
        # eşiği geçmese bile en dirençli
        # adayları teşhis için göster.
        if (
            market_is_crashing
            and
            "crash_resilient"
            in
            df_scored.columns
        ):

            diagnostic = (
                df_scored[
                    (
                        df_scored[
                            "change_%"
                        ]
                        > 0
                    )
                    &
                    (
                        df_scored[
                            "excess_return"
                        ]
                        > 0
                    )
                    &
                    (
                        df_scored[
                            "value_traded"
                        ]
                        >=
                        25_000_000
                    )
                ]
                .sort_values(
                    "resilience_score",
                    ascending=False
                )
                .head(5)
            )

            if not diagnostic.empty:

                msg += (
                    "🔎 <b>GİRİŞ BARAJINI "
                    "AŞAMAYAN CRASH "
                    "DİRENÇLİLER:</b>\n"
                )

                for _, row in (
                    diagnostic.iterrows()
                ):

                    msg += (
                        f"• #{row['ticker']} | "
                        f"%{row['change_%']:+.2f} | "
                        f"Dayanıklılık "
                        f"{row['resilience_score']:.1f} | "
                        f"Piyasa Üstü "
                        f"{row['excess_return']:+.2f} puan\n"
                    )

                msg += "\n"

        msg += (
            "🛡️ <i>Otomatik giriş "
            "kriterlerini geçen yeterli "
            "aday bulunamadı.</i>"
        )

        return msg

    for _, row in (
        shocks.head(10).iterrows()
    ):

        msg += (
            f"🚀 <b>#{row['ticker']}</b> "
            f"── "
            f"<b>{row['shock_score']:.1f} "
            f"Puan</b> "
            f"({row['stars']})\n"
        )

        msg += (
            f"• <b>Fiyat:</b> "
            f"{row['close']:.2f} TL | "
            f"<b>Değişim:</b> "
            f"%{row['change_%']:+.2f}\n"
        )

        msg += (
            f"• <b>Dayanıklılık:</b> "
            f"<b>"
            f"{row.get('resilience_score', 0.0):.1f}"
            f"</b> | "
            f"<b>Piyasa Üstü:</b> "
            f"%"
            f"{row.get('excess_return', 0.0):+.2f}\n"
        )

        msg += (
            f"• <b>1A / 3A Göreli:</b> "
            f"{row.get('rel_1m_pct', 50.0):.0f} / "
            f"{row.get('rel_3m_pct', 50.0):.0f} "
            f"persentil | "
            f"<b>RVOL:</b> "
            f"{row.get('rvol', 1.0):.2f}x\n"
        )

        msg += (
            f"• <b>Giriş:</b> "
            f"<i>{row['entry_status']}</i>\n"
        )

        msg += (
            f"💰 <b>KASA:</b> "
            f"<b>{row['allocation']}</b>\n\n"
        )

    msg += (
        "━━━━━━━━━━━━━━━━━━━━\n"
    )

    msg += (
        f"🎯 <i>Toplam "
        f"{len(shocks)} "
        f"otomatik giriş adayı "
        f"tespit edildi.</i>"
    )

    return msg


def main():

    print(
        f"[{datetime.now().strftime('%H:%M:%S')}] "
        "=== BIST Scanner & Crash Shield V2 Başlıyor ==="
    )

    df_current = fetch_all_data()

    if df_current.empty:

        print(
            "Hata: BIST verisi temin edilemedi."
        )

        return

    # ================================================================
    # 1. MARKET REGIME
    # ================================================================

    (
        market_is_crashing,
        regime_status_text
    ) = assess_bist_market_regime(
        df_current
    )

    print(
        f"Piyasa Durumu: "
        f"{regime_status_text}"
    )

    # ================================================================
    # 2. ÖNCE GEÇMİŞ GETİRİLERİ GÜNCELLE
    # ================================================================

    update_realized_shock_returns(
        df_current
    )

    # ================================================================
    # 3. İLK CROSS-SECTIONAL SKOR
    # ================================================================

    df_temp = calculate_shock_scores(
        df_current,
        pd.DataFrame(),
        market_is_crashing=
        market_is_crashing
    )

    # ================================================================
    # 4. DİNAMİK ÖĞRENME
    # ================================================================

    dynamic_thresholds = (
        compute_dynamic_market_thresholds(
            df_temp
        )
    )

    (
        dynamic_weights,
        ai_status
    ) = calibrate_adaptive_weights()

    (
        resilience_weight,
        resilience_status
    ) = calibrate_resilience_weight()

    saved_min_score = 75.0

    if os.path.exists(
        AI_STATE_FILE
    ):

        try:

            with open(
                AI_STATE_FILE,
                "r"
            ) as f:

                saved_state = json.load(
                    f
                )

            saved_min_score = float(
                saved_state
                .get(
                    "thresholds",
                    {}
                )
                .get(
                    "min_score",
                    75.0
                )
            )

            if isinstance(
                saved_state.get(
                    "weights"
                ),
                dict
            ):

                for key in (
                    "vol",
                    "flow",
                    "range",
                    "lambda"
                ):

                    if (
                        key
                        in
                        saved_state[
                            "weights"
                        ]
                    ):

                        dynamic_weights[
                            key
                        ] = float(
                            saved_state[
                                "weights"
                            ][key]
                        )

        except Exception:
            pass

    dynamic_weights[
        "resilience_weight"
    ] = resilience_weight

    dynamic_thresholds[
        "min_score"
    ] = saved_min_score

    # ================================================================
    # 5. AI STATE
    # ================================================================

    state_payload = {

        "thresholds":
            dynamic_thresholds,

        "weights":
            dynamic_weights,

        "resilience_weight":
            resilience_weight,

        "status":
            f"{ai_status} | "
            f"{resilience_status}",
    }

    with open(
        AI_STATE_FILE,
        "w"
    ) as f:

        json.dump(
            state_payload,
            f,
            indent=4,
            ensure_ascii=False
        )

    # ================================================================
    # 6. FINAL SCORE
    # ================================================================

    df_gecmis = (
        gecmis_veriyi_yukle()
    )

    df_scored = calculate_shock_scores(
        df_current,
        df_gecmis,
        dynamic_thresholds,
        dynamic_weights,
        market_is_crashing=
        market_is_crashing
    )

    if df_scored.empty:
        return

    # ================================================================
    # 7. BİLANÇO KALKANI
    # ================================================================

    print(
        "🔍 BIST bilanço takvimi taranıyor..."
    )

    for idx, row in (
        df_scored.head(20).iterrows()
    ):

        if (
            row["shock_score"]
            >=
            65.0
        ):

            (
                has_earnings,
                e_date
            ) = check_bist_earnings_risk(
                row["ticker"]
            )

            if has_earnings:

                print(
                    f"⚠️ "
                    f"{row['ticker']} "
                    f"için BIST bilanço "
                    f"riski: {e_date}"
                )

                df_scored.at[
                    idx,
                    "entry_status"
                ] = (
                    f"🚨 BİLANÇO RİSKİ "
                    f"({e_date})"
                )

                df_scored.at[
                    idx,
                    "allocation"
                ] = (
                    "İşlem Açma "
                    "(%0 - Bilanço Kumarı)"
                )

                df_scored.at[
                    idx,
                    "stars"
                ] = "⚠️"

                df_scored.at[
                    idx,
                    "shock_score"
                ] = max(
                    saved_min_score
                    - 10.0,
                    0.0
                )

    # ================================================================
    # 8. EXIT ENGINE
    # ================================================================

    exit_signals_text = (
        generate_exit_signals(
            df_current
        )
    )

    # ================================================================
    # 9. BACKTEST LEDGER
    # ================================================================

    record_clean_ledger_entries(
        df_scored,
        market_is_crashing=
        market_is_crashing
    )

    # ================================================================
    # 10. GEÇMİŞ VERİ
    # ================================================================

    if (
        not df_gecmis.empty
        and
        "tarih"
        in
        df_gecmis.columns
    ):

        bugun = (
            pd.Timestamp.now()
            .normalize()
        )

        df_gecmis = df_gecmis[
            df_gecmis[
                "tarih"
            ]
            !=
            bugun
        ]

        df_yeni_gecmis = pd.concat(
            [
                df_gecmis,
                df_scored
            ],
            ignore_index=True
        )

    else:

        df_yeni_gecmis = (
            df_scored.copy()
        )

    if "tarih" not in df_yeni_gecmis.columns:

        df_yeni_gecmis[
            "tarih"
        ] = (
            pd.Timestamp.now()
            .normalize()
        )

    df_yeni_gecmis[
        "tarih"
    ] = pd.to_datetime(
        df_yeni_gecmis[
            "tarih"
        ],
        errors="coerce"
    )

    limit_tarih = (
        pd.Timestamp.now()
        .normalize()
        -
        pd.Timedelta(
            days=30
        )
    )

    df_yeni_gecmis = (
        df_yeni_gecmis[
            df_yeni_gecmis[
                "tarih"
            ]
            >=
            limit_tarih
        ]
    )

    df_yeni_gecmis.to_csv(
        GECMIS_DOSYA,
        index=False
    )

    # ================================================================
    # 11. TELEGRAM RAPOR
    # ================================================================

    telegram_msg = (
        format_shock_report(
            df_scored,
            exit_signals_text,
            regime_status_text,
            market_is_crashing=
            market_is_crashing,
            min_score=
            saved_min_score
        )
    )

    send_telegram_message(
        telegram_msg
    )

    print(
        "BIST Göreli Güç / "
        "Crash Shield raporu "
        "başarıyla tamamlandı."
    )


if __name__ == "__main__":
    main()
