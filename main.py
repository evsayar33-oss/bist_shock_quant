import json
import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from shock_fetcher import fetch_all_data
from shock_engine import calculate_shock_scores, gecmis_veriyi_yukle, GECMIS_DOSYA, classify_bist_regime
from shock_learner import (
    update_realized_shock_returns,
    compute_dynamic_market_thresholds,
    calibrate_adaptive_weights,
    calibrate_resilience_weight,
    build_runtime_meta_profile,
    load_ai_state,
    save_ai_state,
    log_shock_signals,
    AI_STATE_FILE,
)

LEDGER_FILE = "backtest_ledger.csv"


def send_telegram_message(message):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("CHAT_ID")
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    max_len = 3800
    messages = []
    if len(message) > max_len:
        parts = message.split("\n\n")
        current_msg = ""
        for part in parts:
            if len(current_msg) + len(part) + 2 < max_len:
                current_msg += part + "\n\n"
            else:
                if current_msg.strip():
                    messages.append(current_msg.strip())
                current_msg = part + "\n\n"
        if current_msg.strip():
            messages.append(current_msg.strip())
    else:
        messages = [message]

    for msg in messages:
        payload = {
            "chat_id": chat_id,
            "text": msg,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }
        try:
            res = requests.post(url, json=payload, timeout=15)
            if not res.json().get("ok"):
                payload.pop("parse_mode", None)
                requests.post(url, json=payload, timeout=15)
        except Exception as exc:
            print(f"Telegram hatası: {exc}")


def assess_bist_market_regime(df_market):
    snapshot = classify_bist_regime(df_market)
    label = snapshot["label"]
    confidence = snapshot["confidence"]
    red_ratio = snapshot["red_ratio"]
    mean_change = snapshot["mean_change"]
    median_change = snapshot["median_change"]
    dispersion = snapshot["dispersion"]

    labels = {
        "CRASH": "🚨 PİYASA ÇÖKÜŞ REJİMİ",
        "STRESS": "⚠️ PİYASA STRES REJİMİ",
        "ROTATION": "🔄 ROTASYON REJİMİ",
        "EXPANSION": "🚀 GENİŞLEME / GÜÇLENME REJİMİ",
        "QUIET": "🧊 DÜŞÜK VOLATİLİTE / SESSİZ REJİM",
        "NORMAL": "⚖️ NORMAL / DALGALI REJİM",
    }
    status = (
        f"{labels.get(label, label)} "
        f"(Güven %{confidence * 100:.0f} | Düşen %{red_ratio * 100:.0f} | "
        f"Ort %{mean_change:+.2f} | Medyan %{median_change:+.2f} | Disp %{dispersion:.2f})"
    )
    return bool(snapshot["is_crashing"]), status, snapshot


def check_bist_earnings_risk(ticker):
    try:
        t = yf.Ticker(f"{ticker}.IS")
        cal = t.calendar
        if cal is None:
            return False, ""

        ed = None
        if isinstance(cal, dict):
            ed = cal.get("Earnings Date")
        elif hasattr(cal, "get"):
            ed = cal.get("Earnings Date")
        elif hasattr(cal, "loc") and "Earnings Date" in cal.index:
            ed = cal.loc["Earnings Date"].values

        if ed is not None:
            if not isinstance(ed, (list, np.ndarray, tuple)):
                ed = [ed]
            for d in ed:
                if pd.notna(d):
                    d_date = d.date() if hasattr(d, "date") else pd.to_datetime(d).date()
                    days_diff = (d_date - datetime.now().date()).days
                    if 0 <= days_diff <= 5:
                        return True, f"{d_date.strftime('%d.%m')} ({days_diff} Gün Kaldı)"
    except Exception:
        pass
    return False, ""


def generate_exit_signals(df_current):
    if not os.path.exists(LEDGER_FILE):
        return ""
    try:
        df_ledger = pd.read_csv(LEDGER_FILE)
    except Exception:
        return ""

    if df_ledger.empty or "is_completed" not in df_ledger.columns:
        return ""

    open_positions = df_ledger[df_ledger["is_completed"] == 0]
    if open_positions.empty:
        return ""

    close_map = dict(zip(df_current["ticker"], df_current["close"]))
    bugun = datetime.now().date()
    signals = []

    for _, row in open_positions.iterrows():
        ticker = row["ticker"]
        if ticker not in close_map:
            continue

        curr_p = float(close_map[ticker])
        entry_p = float(row["entry_price"])
        try:
            row_date = datetime.strptime(str(row["date"]), "%Y-%m-%d").date()
        except Exception:
            continue
        days_held = (bugun - row_date).days
        pnl = ((curr_p - entry_p) / entry_p) * 100.0

        if pnl <= -3.0:
            signals.append(
                f"🚨 <b>#{ticker} STOP-LOSS (ACİL ÇIKIŞ)!</b>\n"
                f"  ↳ <i>Giriş: {entry_p:.2f} TL | Güncel: {curr_p:.2f} TL | "
                f"Zarar: <b>%{pnl:+.2f}</b>\n  🛑 Stop kırıldı, zararı kes ve çık!</i>"
            )
        elif pnl >= 9.0:
            signals.append(
                f"💰 <b>#{ticker} KÂR AL!</b>\n"
                f"  ↳ <i>Giriş: {entry_p:.2f} TL | Güncel: {curr_p:.2f} TL | "
                f"Kâr: <b>%{pnl:+.2f}</b>\n  🎯 Pozisyonun %50'sini sat, kalanın stopunu maliyete çek!</i>"
            )
        elif pnl >= 4.0:
            signals.append(
                f"🔒 <b>#{ticker} KÂR KORUMA / MALİYET STOPU</b>\n"
                f"  ↳ <i>Fiyat: {curr_p:.2f} TL | Kâr: <b>%{pnl:+.2f}</b>\n"
                f"  🛡️ Stop maliyete ({entry_p:.2f} TL) çekildi.</i>"
            )
        elif days_held >= 5:
            signals.append(
                f"⏰ <b>#{ticker} 1 HAFTALIK VADE DOLDU</b>\n"
                f"  ↳ <i>Kapanış: {curr_p:.2f} TL | Net: <b>%{pnl:+.2f}</b></i>"
            )
        else:
            signals.append(
                f"🟢 <b>#{ticker} TAŞIMAYA DEVAM ET</b> ({days_held}. Gün)\n"
                f"  ↳ <i>Fiyat: {curr_p:.2f} TL | Durum: <b>%{pnl:+.2f}</b></i>"
            )

    if not signals:
        return ""

    return (
        "🛡️ <b>BIST AÇIK POZİSYONLAR & ÇIKIŞ ALARMLARI (Exit Engine):</b>\n"
        "━━━━━━━━━━━━━━━━━━━━\n"
        + "\n\n".join(signals)
        + "\n━━━━━━━━━━━━━━━━━━━━\n\n"
    )


def record_clean_ledger_entries(df_scored, market_is_crashing=False):
    """Write only high-quality active candidates to the production ledger."""
    if df_scored is None or df_scored.empty:
        return

    bugun_str = datetime.now().strftime("%Y-%m-%d")
    candidates = df_scored[df_scored["shock_score"] > 0].copy()
    if candidates.empty:
        return

    if "effective_min_score" in candidates.columns:
        candidates = candidates[candidates["shock_score"] >= candidates["effective_min_score"]]
    else:
        minimum = 88.0 if market_is_crashing else 75.0
        candidates = candidates[candidates["shock_score"] >= minimum]

    candidates = candidates.sort_values(
        ["shock_score", "overnight_risk", "resilience_score"],
        ascending=[False, True, False],
    ).head(10)
    new_rows = []

    for _, r in candidates.iterrows():
        entry_status = str(r.get("entry_status", ""))
        value_traded = float(r.get("value_traded", 0.0))
        resilience = float(r.get("resilience_score", 0.0))
        excess_return = float(r.get("excess_return", 0.0))
        overnight_risk = float(r.get("overnight_risk", 100.0))
        score = float(r.get("shock_score", 0.0))
        current_positive = bool(r.get("current_positive", False))

        if "BİLANÇO" in entry_status:
            continue
        if value_traded < 25_000_000.0 or overnight_risk >= 82.0:
            continue
        if not current_positive:
            continue

        if market_is_crashing:
            if excess_return <= 0.0 or resilience < 55.0:
                continue
            if score < 82.0:
                continue

        new_rows.append(
            {
                "date": bugun_str,
                "ticker": r["ticker"],
                "entry_price": r["close"],
                "z_vol": r.get("z_vol", 0.0),
                "z_range": r.get("z_range", 0.0),
                "z_flow": r.get("z_flow", 0.0),
                "z_lambda": r.get("z_lambda", 0.0),
                "resilience_score": resilience,
                "excess_return": excess_return,
                "rel_1m_pct": r.get("rel_1m_pct", 50.0),
                "rel_3m_pct": r.get("rel_3m_pct", 50.0),
                "trend_persistence": r.get("trend_persistence", 50.0),
                "non_price_score": r.get("non_price_score", 50.0),
                "flow_score": r.get("flow_score", 50.0),
                "activity_score": r.get("activity_score", 50.0),
                "liquidity_score": r.get("liquidity_score", 50.0),
                "overnight_risk": overnight_risk,
                "meta_score": r.get("meta_score", score),
                "risk_adjusted_score": r.get("risk_adjusted_score", score),
                "regime": r.get("meta_regime", "NORMAL"),
                "regime_confidence": r.get("meta_regime_confidence", 0.35),
                "crash_resilient": bool(r.get("crash_resilient", False)),
                "crash_survivor": bool(r.get("crash_survivor", False)),
                "entry_status": r.get("entry_status", "NORMAL"),
                "initial_score": score,
                "price_d1": np.nan,
                "price_d3": np.nan,
                "price_d5": np.nan,
                "return_d1": np.nan,
                "return_d3": np.nan,
                "return_d5": np.nan,
                "is_completed": 0,
            }
        )

    if not new_rows:
        return

    df_new = pd.DataFrame(new_rows)
    if os.path.exists(LEDGER_FILE):
        try:
            df_old = pd.read_csv(LEDGER_FILE)
            if "date" in df_old.columns and "ticker" in df_old.columns:
                df_old = df_old[
                    ~((df_old["date"] == bugun_str) & (df_old["ticker"].isin(df_new["ticker"])))
                ]
            df_final = pd.concat([df_old, df_new], ignore_index=True, sort=False)
        except Exception:
            df_final = df_new
    else:
        df_final = df_new

    df_final.to_csv(LEDGER_FILE, index=False)


def format_shock_report(
    df_scored,
    exit_signals_text,
    market_regime_text,
    market_snapshot,
    runtime_profile,
):
    if df_scored is None or df_scored.empty:
        return exit_signals_text or "BIST taraması sonuç üretmedi."

    effective_min = float(df_scored["effective_min_score"].iloc[0]) if "effective_min_score" in df_scored.columns else 75.0
    shocks = df_scored[df_scored["shock_score"] >= effective_min].sort_values(
        by=["shock_score", "overnight_risk", "resilience_score"], ascending=[False, True, False]
    )

    msg = exit_signals_text or ""
    msg += "🧠 <b>ADAPTIVE BIST META-ENGINE</b>\n"
    msg += f"🗓 <i>{datetime.now().strftime('%Y-%m-%d %H:%M')} | Seans Taraması</i>\n"
    msg += f"🌐 <b>Rejim:</b> <i>{market_regime_text}</i>\n"
    msg += f"🎯 <b>Canlı Eşik:</b> {effective_min:.1f} | <b>Profil:</b> {runtime_profile.get('source', 'META')}\n"
    msg += "━━━━━━━━━━━━━━━━━━━━\n\n"

    if market_snapshot.get("is_crashing"):
        msg += (
            "🛡️ <b>CRASH SHIELD:</b> Ana seçim skoru fiyat hareketini kovalamak yerine "
            "olay/akış/aktivite/likidite ve göreli dayanıklılık kombinasyonunu kullanıyor.\n\n"
        )

    if shocks.empty:
        watch = df_scored[df_scored["watch_score"] >= max(effective_min - 5.0, 65.0)].head(5)
        if not watch.empty:
            msg += "🔎 <b>WATCH / HENÜZ GİRİŞ EŞİĞİNDE DEĞİL:</b>\n"
            for _, row in watch.iterrows():
                msg += (
                    f"• #{row['ticker']} | Meta {row['watch_score']:.1f} | "
                    f"Non-price {row.get('non_price_score', 0):.1f} | "
                    f"Risk {row.get('overnight_risk', 100):.1f}\n"
                )
            msg += "\n"
        msg += "🛡️ <i>Otomatik giriş kriterlerini geçen aday bulunamadı.</i>"
        return msg

    for _, row in shocks.head(10).iterrows():
        msg += (
            f"🚀 <b>#{row['ticker']}</b> ── <b>{row['shock_score']:.1f} Puan</b> "
            f"({row['stars']})\n"
        )
        msg += (
            f"• <b>Değişim:</b> %{row['change_%']:+.2f} | "
            f"<b>Piyasa Üstü:</b> %{row.get('excess_return', 0.0):+.2f}\n"
        )
        msg += (
            f"• <b>Non-price:</b> {row.get('non_price_score', 0.0):.1f} | "
            f"<b>Dayanıklılık:</b> {row.get('resilience_score', 0.0):.1f} | "
            f"<b>Akış:</b> {row.get('flow_score', 0.0):.1f}\n"
        )
        msg += (
            f"• <b>Likidite:</b> {row.get('liquidity_score', 0.0):.1f} | "
            f"<b>Overnight Risk:</b> {row.get('overnight_risk', 100.0):.1f} | "
            f"<b>1A/3A Rel:</b> {row.get('rel_1m_pct', 50):.0f}/{row.get('rel_3m_pct', 50):.0f}\n"
        )
        msg += f"• <b>Giriş:</b> <i>{row['entry_status']}</i>\n"
        msg += f"💰 <b>KASA:</b> <b>{row['allocation']}</b>\n\n"

    msg += "━━━━━━━━━━━━━━━━━━━━\n"
    msg += f"🎯 <i>{len(shocks)} aktif giriş adayı. Meta Engine v{runtime_profile.get('version', 1)}.</i>"
    return msg


def main():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] === Adaptive BIST Meta-Engine Başlıyor ===")

    df_current = fetch_all_data()
    if df_current.empty:
        print("Hata: BIST verisi temin edilemedi.")
        return

    market_is_crashing, regime_status_text, market_snapshot = assess_bist_market_regime(df_current)
    print(f"Piyasa Durumu: {regime_status_text}")

    # Önceki sinyallerin gerçekleşen sonuçlarını güncelle.
    update_realized_shock_returns(df_current)

    state = load_ai_state()

    # Legacy factor weights geriye dönük uyumluluk için korunuyor.
    legacy_weights, legacy_status = calibrate_adaptive_weights()
    resilience_weight, resilience_status = calibrate_resilience_weight()

    # Canlı rejime göre runtime meta profilini oluştur.
    runtime_profile = build_runtime_meta_profile(market_snapshot, state)

    # Dinamik event eşikleri güncel cross-section'tan alınır.
    df_temp = calculate_shock_scores(
        df_current,
        pd.DataFrame(),
        dynamic_thresholds=DEFAULT_SAFE_THRESHOLDS(),
        dynamic_weights=legacy_weights,
        market_is_crashing=market_is_crashing,
        regime_snapshot=market_snapshot,
        meta_profile=runtime_profile,
    )
    dynamic_thresholds = compute_dynamic_market_thresholds(df_temp)

    # State içindeki öğrenilmiş legacy ağırlıkları ezmeden güncelle.
    state["thresholds"] = {**state.get("thresholds", {}), **dynamic_thresholds}
    state["weights"] = {**state.get("weights", {}), **legacy_weights}
    state["resilience_weight"] = resilience_weight
    state["status"] = (
        f"🧠 META ENGINE | {market_snapshot['label']} | "
        f"Güven %{market_snapshot['confidence'] * 100:.0f} | "
        f"{legacy_status} | {resilience_status}"
    )
    state.setdefault("meta_engine", {})
    state["meta_engine"]["version"] = 1
    state["meta_engine"]["last_runtime"] = runtime_profile
    save_ai_state(state)

    df_gecmis = gecmis_veriyi_yukle()
    df_scored = calculate_shock_scores(
        df_current,
        df_gecmis,
        dynamic_thresholds=dynamic_thresholds,
        dynamic_weights={**legacy_weights, "resilience_weight": resilience_weight},
        market_is_crashing=market_is_crashing,
        regime_snapshot=market_snapshot,
        meta_profile=runtime_profile,
    )

    if df_scored.empty:
        return

    # Shadow profil varsa aynı gün aynı veride bağımsız skor üretilir; bu sinyal üretmez.
    shadow_df = pd.DataFrame()
    shadow_profile = state.get("meta_engine", {}).get("shadow", {}).get("profile")
    if isinstance(shadow_profile, dict) and shadow_profile.get("weights"):
        shadow_runtime = dict(shadow_profile)
        shadow_runtime["regime"] = market_snapshot["label"]
        shadow_runtime["confidence"] = market_snapshot["confidence"]
        shadow_df = calculate_shock_scores(
            df_current,
            df_gecmis,
            dynamic_thresholds=dynamic_thresholds,
            dynamic_weights={**legacy_weights, "resilience_weight": resilience_weight},
            market_is_crashing=market_is_crashing,
            regime_snapshot=market_snapshot,
            meta_profile=shadow_runtime,
        )

    # Öğrenme veri akışının kritik parçası: her taramada aktif/shadow adaylar kayıt edilir.
    log_shock_signals(
        df_scored.head(10),
        regime_snapshot=market_snapshot,
        shadow_df=shadow_df.head(8) if not shadow_df.empty else None,
    )

    print("🔍 BIST bilanço takvimi taranıyor...")
    for idx, row in df_scored.head(20).iterrows():
        if row["shock_score"] >= row.get("effective_min_score", 75.0):
            has_earnings, e_date = check_bist_earnings_risk(row["ticker"])
            if has_earnings:
                print(f"⚠️ {row['ticker']} için BIST bilanço riski: {e_date}")
                df_scored.at[idx, "entry_status"] = f"🚨 BİLANÇO RİSKİ ({e_date})"
                df_scored.at[idx, "allocation"] = "İşlem Açma (%0 - Bilanço Riski)"
                df_scored.at[idx, "stars"] = "⚠️"
                df_scored.at[idx, "shock_score"] = 0.0

    exit_signals_text = generate_exit_signals(df_current)
    record_clean_ledger_entries(df_scored, market_is_crashing=market_is_crashing)

    # Geçmişi 30 günle sınırlamak yerine daha uzun bir öğrenme penceresi tutuyoruz.
    if not df_gecmis.empty and "tarih" in df_gecmis.columns:
        bugun = pd.Timestamp.now().normalize()
        df_gecmis = df_gecmis[df_gecmis["tarih"] != bugun]
        df_yeni_gecmis = pd.concat([df_gecmis, df_scored], ignore_index=True, sort=False)
    else:
        df_yeni_gecmis = df_scored.copy()

    if "tarih" not in df_yeni_gecmis.columns:
        df_yeni_gecmis["tarih"] = pd.Timestamp.now().normalize()
    df_yeni_gecmis["tarih"] = pd.to_datetime(df_yeni_gecmis["tarih"], errors="coerce")
    limit_tarih = pd.Timestamp.now().normalize() - pd.Timedelta(days=120)
    df_yeni_gecmis = df_yeni_gecmis[df_yeni_gecmis["tarih"] >= limit_tarih]
    df_yeni_gecmis.to_csv(GECMIS_DOSYA, index=False)

    telegram_msg = format_shock_report(
        df_scored,
        exit_signals_text,
        regime_status_text,
        market_snapshot,
        runtime_profile,
    )
    send_telegram_message(telegram_msg)
    print("Adaptive BIST Meta-Engine taraması başarıyla tamamlandı.")


def DEFAULT_SAFE_THRESHOLDS():
    return {"th_vol": 1.5, "th_range": 1.5, "th_flow": 2.0, "th_lambda": 1.2}


if __name__ == "__main__":
    main()
