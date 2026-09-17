import os
from datetime import datetime

import numpy as np
import pandas as pd
import requests

from shock_fetcher import fetch_all_data
from shock_engine import classify_bist_regime
from shock_learner import (
    AI_STATE_FILE,
    GECMIS_DOSYA,
    LEDGER_FILE,
    build_runtime_meta_profile,
    evaluate_profile,
    learn_meta_candidate,
    load_ai_state,
    prepare_historical_frame,
    save_ai_state,
    split_training_validation,
)

MIN_ADAPTIVE_TRAIN_ROWS = 150
MIN_ADAPTIVE_VALIDATION_SAMPLES = 20
PROMOTION_MIN_SAMPLES = 25
ROLLBACK_MIN_SAMPLES = 30


def send_telegram_audit(message):
    token = os.environ.get("TELEGRAM_TOKEN")
    chat_id = os.environ.get("CHAT_ID")
    if not token or not chat_id:
        return

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }
    try:
        res = requests.post(url, json=payload, timeout=15)
        if not res.json().get("ok"):
            payload.pop("parse_mode", None)
            requests.post(url, json=payload, timeout=15)
    except Exception as exc:
        print(f"Telegram audit hatası: {exc}")


def _safe_float(value, default=0.0):
    try:
        value = float(value)
        return default if not np.isfinite(value) else value
    except Exception:
        return default


def _trading_days_passed(row_date, current_date, history_dates):
    row_date = pd.Timestamp(row_date).normalize()
    current_date = pd.Timestamp(current_date).normalize()
    dates = sorted(set(pd.to_datetime(history_dates, errors="coerce").dropna().tolist()))
    dates = [pd.Timestamp(d).normalize() for d in dates if row_date <= pd.Timestamp(d).normalize() <= current_date]
    if current_date not in dates:
        dates.append(current_date)
    if len(dates) >= 2:
        return max(len(sorted(set(dates))) - 1, 0)
    try:
        return int(np.busday_count(row_date.date(), current_date.date()))
    except Exception:
        return 0


def update_ledger_returns(df_close):
    if not os.path.exists(LEDGER_FILE):
        return pd.DataFrame()
    try:
        df_ledger = pd.read_csv(LEDGER_FILE)
    except Exception:
        return pd.DataFrame()
    if df_ledger.empty:
        return df_ledger

    required = [
        "price_d1", "price_d3", "price_d5", "return_d1", "return_d3", "return_d5", "is_completed"
    ]
    for col in required:
        if col not in df_ledger.columns:
            df_ledger[col] = np.nan if col != "is_completed" else 0

    close_map = dict(zip(df_close["ticker"], df_close["close"]))
    current_date = pd.Timestamp.now().normalize()
    history_dates = []
    if os.path.exists(GECMIS_DOSYA):
        try:
            history_dates = pd.read_csv(GECMIS_DOSYA, usecols=["tarih"])["tarih"].tolist()
        except Exception:
            history_dates = []

    for idx, row in df_ledger.iterrows():
        ticker = row.get("ticker")
        if ticker not in close_map:
            continue
        try:
            curr_p = float(close_map[ticker])
            entry_p = float(row["entry_price"])
            row_date = pd.Timestamp(row["date"]).normalize()
        except Exception:
            continue
        if entry_p <= 0:
            continue

        trading_days = _trading_days_passed(row_date, current_date, history_dates)
        ret = ((curr_p - entry_p) / entry_p) * 100.0

        if trading_days >= 1 and pd.isna(df_ledger.at[idx, "price_d1"]):
            df_ledger.at[idx, "price_d1"] = curr_p
            df_ledger.at[idx, "return_d1"] = round(ret, 2)
        if trading_days >= 3 and pd.isna(df_ledger.at[idx, "price_d3"]):
            df_ledger.at[idx, "price_d3"] = curr_p
            df_ledger.at[idx, "return_d3"] = round(ret, 2)
        if trading_days >= 5 and pd.isna(df_ledger.at[idx, "price_d5"]):
            df_ledger.at[idx, "price_d5"] = curr_p
            df_ledger.at[idx, "return_d5"] = round(ret, 2)
            df_ledger.at[idx, "is_completed"] = 1

    df_ledger.to_csv(LEDGER_FILE, index=False)
    return df_ledger


def _audit_metric(trades):
    if trades is None or trades.empty or "return_d5" not in trades.columns:
        return {"n": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_return": 0.0, "score": -999.0}

    ret = pd.to_numeric(trades["return_d5"], errors="coerce").dropna()
    if ret.empty:
        return {"n": 0, "win_rate": 0.0, "profit_factor": 0.0, "avg_return": 0.0, "score": -999.0}

    gains = ret[ret > 0].sum()
    losses = abs(ret[ret < 0].sum())
    win_rate = float((ret > 0).mean() * 100.0)
    profit_factor = float(gains / losses) if losses > 0 else 5.0
    avg_return = float(ret.mean())
    sample_factor = min(len(ret) / 30.0, 1.0)
    score = (avg_return * 0.40 + np.log1p(profit_factor) * 2.0 + win_rate * 0.02) * sample_factor
    return {
        "n": int(len(ret)),
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "avg_return": avg_return,
        "score": score,
    }


def run_threshold_audit(completed_trades):
    """Legacy ledger threshold audit kept as a secondary diagnostic."""
    if completed_trades is None or completed_trades.empty:
        return None
    if "initial_score" not in completed_trades.columns or "return_d5" not in completed_trades.columns:
        return None

    data = completed_trades.copy()
    data["initial_score"] = pd.to_numeric(data["initial_score"], errors="coerce")
    data["return_d5"] = pd.to_numeric(data["return_d5"], errors="coerce")
    data = data.dropna(subset=["initial_score", "return_d5"])
    best = None
    for threshold in np.arange(68.0, 93.0, 2.0):
        selected = data[data["initial_score"] >= threshold]
        metrics = _audit_metric(selected)
        if metrics["n"] < 10:
            continue
        candidate = {"min_score": float(threshold), **metrics}
        if best is None or candidate["score"] > best["score"]:
            best = candidate
    return best


def _profile_key(profile):
    return {
        "weights": {k: round(_safe_float(v), 4) for k, v in profile.get("weights", {}).items()},
        "min_score": round(_safe_float(profile.get("min_score"), 75.0), 1),
        "version": int(_safe_float(profile.get("version"), 1)),
        "regime": str(profile.get("regime", "NORMAL")),
    }


def _promotion_allowed(active_metrics, candidate_metrics):
    if candidate_metrics["n"] < PROMOTION_MIN_SAMPLES:
        return False, "YETERSİZ VALIDATION ÖRNEKLEMİ"

    if active_metrics["n"] < PROMOTION_MIN_SAMPLES:
        ok = (
            candidate_metrics["profit_factor"] >= 1.10
            and candidate_metrics["avg_return"] >= 0.15
            and candidate_metrics["win_rate"] >= 45.0
        )
        return ok, "AKTİF MODEL İÇİN ÖRNEKLEM AZ; DAHA YÜKSEK GÜVEN EŞİĞİ"

    pf_lift = candidate_metrics["profit_factor"] - active_metrics["profit_factor"]
    avg_lift = candidate_metrics["avg_return"] - active_metrics["avg_return"]
    wr_delta = candidate_metrics["win_rate"] - active_metrics["win_rate"]
    p10_delta = candidate_metrics["p10"] - active_metrics["p10"]

    improve = (pf_lift >= 0.10) or (avg_lift >= 0.15)
    safety = (wr_delta >= -3.0) and (p10_delta >= -0.75)
    absolute_floor = candidate_metrics["profit_factor"] >= 1.02 and candidate_metrics["avg_return"] > -0.10
    return bool(improve and safety and absolute_floor), (
        f"PF fark {pf_lift:+.2f} | Ortalama fark {avg_lift:+.2f} | "
        f"WR fark {wr_delta:+.1f}pp | P10 fark {p10_delta:+.2f}"
    )


def _rollback_needed(active_metrics):
    if active_metrics["n"] < ROLLBACK_MIN_SAMPLES:
        return False
    return bool(
        active_metrics["profit_factor"] < 0.80
        or (active_metrics["avg_return"] < -0.75 and active_metrics["win_rate"] < 45.0)
    )


def run_adaptive_meta_audit(current_df, state):
    """Train candidate profile on older data and validate on a recent holdout window."""
    snapshot = classify_bist_regime(current_df)
    regime = snapshot["label"]

    if not os.path.exists(GECMIS_DOSYA):
        return state, {
            "regime": regime,
            "status": "NO_HISTORY",
            "message": "Geçmiş veri yok; shadow model üretilemedi.",
        }

    try:
        hist_raw = pd.read_csv(GECMIS_DOSYA)
    except Exception:
        return state, {"regime": regime, "status": "READ_ERROR", "message": "Geçmiş veri okunamadı."}

    hist = prepare_historical_frame(hist_raw)
    if hist.empty or len(hist) < MIN_ADAPTIVE_TRAIN_ROWS:
        return state, {
            "regime": regime,
            "status": "TRAINING_WAIT",
            "message": f"Meta öğrenme için veri yetersiz: {len(hist)} satır.",
        }

    train, validation, validation_dates = split_training_validation(hist_raw, validation_days=3, forward_days=3)
    if train.empty or validation.empty:
        return state, {"regime": regime, "status": "HOLDOUT_WAIT", "message": "Walk-forward holdout için yeterli tarih yok."}

    candidate = learn_meta_candidate(train, regime)
    meta = state.setdefault("meta_engine", {})
    profiles = meta.setdefault("regime_profiles", {})
    stable = meta.setdefault("stable_profiles", {})

    active = profiles.get(regime)
    if not isinstance(active, dict):
        active = {
            "weights": dict(candidate.get("weights", {})),
            "min_score": float(candidate.get("min_score", 75.0)),
            "version": 1,
            "regime": regime,
            "learning_status": "INITIAL_TEMPLATE",
        }

    active_metrics = evaluate_profile(validation, active, min_samples=10, target="future_ret_3d")
    candidate_metrics = evaluate_profile(validation, candidate, min_samples=10, target="future_ret_3d")

    validation_info = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "regime": regime,
        "regime_confidence": snapshot["confidence"],
        "validation_days": len(validation_dates),
        "active": active_metrics,
        "candidate": candidate_metrics,
        "candidate_profile": _profile_key(candidate),
    }

    promoted = False
    rollback = False
    decision_note = ""

    if candidate_metrics["n"] >= MIN_ADAPTIVE_VALIDATION_SAMPLES:
        allowed, note = _promotion_allowed(active_metrics, candidate_metrics)
        if allowed:
            old_active = _profile_key(active)
            stable[regime] = old_active
            promoted_profile = dict(candidate)
            promoted_profile["version"] = int(_safe_float(active.get("version"), 1)) + 1
            promoted_profile["promoted_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            promoted_profile["promoted_validation"] = candidate_metrics
            profiles[regime] = promoted_profile
            promoted = True
            meta["promotion_count"] = int(meta.get("promotion_count", 0)) + 1
            decision_note = f"SHADOW → ACTIVE | {note}"
        else:
            meta["shadow"] = {
                "profile": _profile_key(candidate),
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "regime": regime,
                "validation": candidate_metrics,
                "decision": note,
            }
            decision_note = f"SHADOW KORUNDU | {note}"
    else:
        meta["shadow"] = {
            "profile": _profile_key(candidate),
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "regime": regime,
            "validation": candidate_metrics,
            "decision": "Örneklem bekleniyor",
        }
        decision_note = "SHADOW KORUNDU | Validation örneklemi yetersiz"

    # Aktif profil ciddi bozulduysa son stable profile'a kontrollü geri dönüş.
    current_active = profiles.get(regime, active)
    current_metrics = evaluate_profile(validation, current_active, min_samples=10, target="future_ret_3d")
    if not promoted and _rollback_needed(current_metrics) and regime in stable:
        profiles[regime] = stable[regime]
        meta["rollback_count"] = int(meta.get("rollback_count", 0)) + 1
        rollback = True
        decision_note = "ROLLBACK | Aktif profil güvenlik sınırını bozdu; stable profil geri alındı"

    meta["last_validation"] = validation_info
    meta["last_decision"] = {
        "regime": regime,
        "promoted": promoted,
        "rollback": rollback,
        "note": decision_note,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    state["status"] = (
        f"🧠 META ENGINE AUDIT | {regime} | "
        f"Active PF {active_metrics['profit_factor']:.2f} | "
        f"Shadow PF {candidate_metrics['profit_factor']:.2f} | {decision_note}"
    )
    return state, {
        "regime": regime,
        "status": "PROMOTED" if promoted else ("ROLLBACK" if rollback else "SHADOW"),
        "decision": decision_note,
        "active": active_metrics,
        "candidate": candidate_metrics,
        "profile": _profile_key(profiles.get(regime, candidate)),
    }


def run_evening_audit():
    print(f"[{datetime.now().strftime('%H:%M:%S')}] Adaptive BIST Meta-Engine kapanış denetimi başlatılıyor...")

    df_close = fetch_all_data()
    if df_close.empty:
        print("BIST kapanış verisi alınamadı.")
        return

    df_ledger = update_ledger_returns(df_close)
    completed = (
        df_ledger[df_ledger["is_completed"] == 1]
        if not df_ledger.empty and "is_completed" in df_ledger.columns
        else pd.DataFrame()
    )
    legacy_best = run_threshold_audit(completed) if len(completed) >= 15 else None

    state = load_ai_state()
    state, meta_report = run_adaptive_meta_audit(df_close, state)

    # Legacy threshold audit yalnızca raporlanır; artık Adaptive Meta profile'ı ezemez.
    state.setdefault("legacy_audit", {})
    if legacy_best is not None:
        state["legacy_audit"] = {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "min_score": legacy_best["min_score"],
            "n": legacy_best["n"],
            "win_rate": legacy_best["win_rate"],
            "profit_factor": legacy_best["profit_factor"],
            "avg_return": legacy_best["avg_return"],
        }

    save_ai_state(state)

    rep = "🧠 <b>ADAPTIVE BIST META-ENGINE / EVENING AUDIT</b>\n"
    rep += f"🗓 <i>{datetime.now().strftime('%Y-%m-%d %H:%M')} | Kapanış</i>\n"
    rep += "━━━━━━━━━━━━━━━━━━━━\n\n"
    rep += f"🌐 <b>Rejim:</b> {meta_report.get('regime', 'NORMAL')}\n"
    rep += f"🧪 <b>Karar:</b> {meta_report.get('status', 'WAIT')}\n"
    rep += f"• {meta_report.get('decision', meta_report.get('message', ''))}\n\n"

    active = meta_report.get("active", {})
    candidate = meta_report.get("candidate", {})
    if active:
        rep += (
            f"🟢 <b>ACTIVE:</b> N={active.get('n', 0)} | WR=%{active.get('win_rate', 0):.1f} | "
            f"PF={active.get('profit_factor', 0):.2f} | Ort={active.get('avg_return', 0):+.2f}\n"
        )
    if candidate:
        rep += (
            f"🟡 <b>SHADOW:</b> N={candidate.get('n', 0)} | WR=%{candidate.get('win_rate', 0):.1f} | "
            f"PF={candidate.get('profit_factor', 0):.2f} | Ort={candidate.get('avg_return', 0):+.2f}\n\n"
        )

    if legacy_best is not None:
        rep += (
            "📚 <b>Legacy threshold teşhisi:</b> "
            f"N={legacy_best['n']} | Eşik={legacy_best['min_score']:.1f} | "
            f"PF={legacy_best['profit_factor']:.2f} | WR=%{legacy_best['win_rate']:.1f}\n\n"
        )

    rep += "🛡️ <i>Aktif model yalnızca holdout doğrulaması ve güvenlik kuralları geçilirse değişir; aksi halde shadow olarak bekler.</i>"
    send_telegram_audit(rep)
    print("Adaptive BIST Meta-Engine kapanış denetimi tamamlandı.")


if __name__ == "__main__":
    run_evening_audit()
