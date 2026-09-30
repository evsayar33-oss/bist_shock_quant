from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO = Path.cwd() if (Path.cwd() / "main.py").exists() else HERE

src_guard = HERE / "autonomy_guard.py"
dst_guard = REPO / "autonomy_guard.py"
if src_guard.resolve() != dst_guard.resolve():
    if dst_guard.exists():
        backup_guard = dst_guard.with_suffix(dst_guard.suffix + ".pre_kurun_backup")
        if not backup_guard.exists():
            shutil.copy2(dst_guard, backup_guard)
    shutil.copy2(src_guard, dst_guard)


def patch(path: Path, old: str, new: str, label: str):
    if not path.exists():
        raise SystemExit(f"MISSING: {path}")
    text = path.read_text(encoding="utf-8")
    count = text.count(old)
    if count != 1:
        raise SystemExit(f"ANCHOR ERROR {label}: expected 1 match, found {count}")
    backup = path.with_suffix(path.suffix + ".pre_kurun_backup")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(text.replace(old, new), encoding="utf-8")


main = REPO / "main.py"
patch(
    main,
    "    log_shock_signals,\n",
    "    log_shock_signals,\n    load_signal_history,\n",
    "learner import",
)
patch(
    main,
    "    AI_STATE_FILE,\n)",
    "    AI_STATE_FILE,\n)\nfrom autonomy_guard import evaluate_autonomy_guard\n",
    "guard import",
)
patch(
    main,
    "    if df_scored.empty:\n        return\n\n    # Shadow profil",
    '''    if df_scored.empty:\n        return\n\n    # Regime Stress-Test / Drift / Safe-Mode. Underlying shock scores stay intact;\n    # the guard only raises the live entry threshold or blocks new entries.\n    guard_log = load_signal_history()\n    guard_result = evaluate_autonomy_guard(\n        state,\n        features=df_scored,\n        regime=market_snapshot,\n        regime_confidence=float(market_snapshot.get("confidence", 0.0)),\n        performance_returns=(guard_log["realized_3d"] if "realized_3d" in guard_log.columns else None),\n        data_quality_score=100.0,\n        row_count=len(df_current),\n        min_rows=100,\n        project="bist_shock",\n    )\n    if "effective_min_score" in df_scored.columns:\n        base_effective = pd.to_numeric(\n            df_scored["effective_min_score"], errors="coerce"\n        ).fillna(float(runtime_profile.get("min_score", 75.0)))\n    else:\n        base_effective = pd.Series(\n            float(runtime_profile.get("min_score", 75.0)), index=df_scored.index, dtype=float\n        )\n    df_scored["effective_min_score"] = base_effective + float(guard_result.get("signal_threshold_add", 0.0))\n    if guard_result.get("block_new_entries"):\n        df_scored["effective_min_score"] = 101.0\n\n    # Shadow profil''',
    "main guard",
)
patch(
    main,
    '    save_ai_state(state)\n\n    df_gecmis = gecmis_veriyi_yukle()\n',
    '    save_ai_state(state)\n\n    # Guard state must survive the same scan; it is stored inside the existing AI state file.\n    save_ai_state(state)\n\n    df_gecmis = gecmis_veriyi_yukle()\n',
    "persist guard state",
)

# Evening auditor: performance drift and recovery are evaluated at close too.
aud = REPO / "shock_auditor.py"
patch(
    aud,
    "from shock_learner import (\n",
    "from shock_learner import (\n",
    "auditor learner anchor",
)
patch(
    aud,
    "    save_ai_state,\n    split_training_validation,\n)",
    "    save_ai_state,\n    split_training_validation,\n    load_signal_history,\n)\nfrom autonomy_guard import evaluate_autonomy_guard",
    "auditor guard import",
)
patch(
    aud,
    "    state = load_ai_state()\n    state, meta_report = run_adaptive_meta_audit(df_close, state)\n",
    '''    state = load_ai_state()\n\n    # Close-of-day performance drift check; the morning scan already evaluates feature drift.\n    signal_hist = load_signal_history()\n    evaluate_autonomy_guard(\n        state,\n        features=None,\n        regime=snapshot,\n        regime_confidence=float(snapshot.get("confidence", 0.0)),\n        performance_returns=(signal_hist["realized_3d"] if "realized_3d" in signal_hist.columns else None),\n        data_quality_score=100.0,\n        row_count=len(df_close),\n        min_rows=100,\n        project="bist_shock",\n    )\n    state, meta_report = run_adaptive_meta_audit(df_close, state)\n''',
    "auditor guard call",
)
patch(
    aud,
    '    rep += f"🌐 <b>Rejim:</b> {meta_report.get(\'regime\', \'NORMAL\")}\\n"\n',
    '    rep += f"🌐 <b>Rejim:</b> {meta_report.get(\'regime\', \'NORMAL\")}\\n"\n    guard_view = state.get("autonomy_guard", {})\n    rep += f"🛡️ <b>Otonomi:</b> {guard_view.get(\'mode\', \'NORMAL\")} | x{guard_view.get(\'exposure_multiplier\', 1.0):.2f} | {guard_view.get(\'reason\', \'\')}\\n"\n',
    "auditor report",
)

for name in ("autonomy_guard.py", "main.py", "shock_auditor.py"):
    subprocess.check_call([sys.executable, "-m", "py_compile", str(REPO / name)])
print("OK: BIST Shock Kur-Unut guard installed. Backups: *.pre_kurun_backup")
