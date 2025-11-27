from __future__ import annotations
import math
from typing import Any, Dict, List, Optional

# --- tiny target resolver (previously targets.py) ---
_DEFAULT_TARGETS = {
    ("speech","clear"): (0.30, 0.60),
    ("quiet","calm"):  (0.25, 0.45),
    ("rehearsal","lively"): (0.60, 1.00),
    ("music","lively"): (0.60, 1.20),
}
def _resolve_target(use: str|None, goal: str|None):
    u = (use or "").strip().lower()
    g = (goal or "").strip().lower()
    if (u,g) in _DEFAULT_TARGETS:
        return _DEFAULT_TARGETS[(u,g)]
    if u in ("speech","talk","meeting"):
        return _DEFAULT_TARGETS[("speech","clear")]
    if g in ("quiet","calm","focused"):
        return _DEFAULT_TARGETS[("quiet","calm")]
    if u in ("rehearsal","practice") or g in ("lively","reverberant"):
        return _DEFAULT_TARGETS[("rehearsal","lively")]
    if u in ("music","listening"):
        return _DEFAULT_TARGETS[("music","lively")]
    return None

# --- helpers ---
def _sf(x: Any, dp:int=2) -> str:
    try:
        v = float(x)
        if not math.isfinite(v):
            return "—"
        return f"{v:.{dp}f}"
    except Exception:
        return "—"

def _spectral_label(low: float, high: float, thresh: float=0.15) -> str:
    try:
        if not (math.isfinite(low) and math.isfinite(high)):
            return "unknown"
        if low - high >= thresh:
            return "boomy"
        if high - low >= thresh:
            return "bright"
        return "balanced"
    except Exception:
        return "unknown"

def _qc_gate(r2: Optional[float], snr_db: Optional[float], capped: bool) -> str:
    try:
        r2_ok = (r2 is not None) and math.isfinite(r2) and (r2 >= 0.90)
        snr_ok = (snr_db is not None) and math.isfinite(snr_db) and (snr_db >= 6.0)
        if (not capped) and r2_ok and snr_ok:
            return "strong"
        return "weak"
    except Exception:
        return "weak"

# --- main: per-take direct feedback ---
def direct_feedback(f: Dict[str, Any], preds: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
    import math
    # --- pull values ---
    rt60_meas = f.get("rt60_s", float("nan"))
    edt_s     = f.get("edt_s", float("nan"))
    c50_db    = f.get("c50_db", float("nan"))
    c80_db    = f.get("c80_db", float("nan"))
    r2        = f.get("r2", float("nan"))
    snr_db    = f.get("snr_db", float("nan"))
    capped    = bool(f.get("capped", False))
    l90_db    = f.get("spl_l90_db", float("nan"))
    low_med   = f.get("rt60_low_med", float("nan"))
    mid_med   = f.get("rt60_mid_med", float("nan"))
    high_med  = f.get("rt60_high_med", float("nan"))

    rt60_pred = float(preds.get("rt60_pred", float("nan")) if preds else float("nan"))
    label     = str(preds.get("label","")).upper() if preds else "CHECK"

    # --- choose rt60 to present ---
    qc = _qc_gate(r2, snr_db, capped)
    if qc == "strong" and math.isfinite(rt60_meas):
        rt60_use, src = float(rt60_meas), "measured"
    else:
        if math.isfinite(rt60_pred) and math.isfinite(rt60_meas):
            rt60_use, src = float(rt60_pred), "fused"
        elif math.isfinite(rt60_pred):
            rt60_use, src = float(rt60_pred), "predicted"
        else:
            rt60_use, src = float(rt60_meas), "measured"

    # --- target band (optional) ---
    use  = (context.get("use","")  or "").strip()
    goal = (context.get("goal","") or "").strip()
    target = _resolve_target(use, goal)

    bullets: List[str] = []
    actions: List[str] = []

    # ------------------------------
    # 1) File confidence + capping (plain English; numbers in brackets)
    # ------------------------------
    if qc == "strong":
        text = f"High confidence reverberation estimate (r² = {_sf(r2,2)}, SNR = {_sf(snr_db,1)} dB)."
    else:
        text = f"Lower confidence reverberation estimate (r² = {_sf(r2,2)}, SNR = {_sf(snr_db,1)} dB)."

    if capped:
        # Clear English about what 'capped' means
        text += " Estimates exceed the recording length and were capped."
    bullets.append(text)

# ------------------------------
# 2) Decay + bands (decisive, parentheses, commas)
# ------------------------------
    spectral = _spectral_label(low_med, high_med)
    if math.isfinite(rt60_use):
        # (RT60 = x.xx s, EDT = y.yy s)
        vals = f"(RT60 = {_sf(rt60_use,2)} s" + (f", EDT = {_sf(edt_s,2)} s" if math.isfinite(edt_s) else "") + ")"

        # decay judgement
        if target:
            lo, hi = target
            if rt60_use > hi + 0.05:
                decay_judgement = "Decay time is high for this use"
            elif rt60_use < lo - 0.05:
                decay_judgement = "Decay time is short for this use"
            else:
                decay_judgement = "Decay time sits within the target range"
        else:
            decay_judgement = (
                "Decay time is high" if rt60_use >= 1.00 else
                ("Decay time is short" if rt60_use < 0.30 else "Decay time is moderate")
            )

        main = f"{decay_judgement} {vals}"

        # band flags: steadier thresholds
        flags = []
        def _is_high(v):
            if not math.isfinite(v): return False
            if target: return v > (target[1] + 0.30)
            return v > 1.80

        if _is_high(low_med):  flags.append(f"Low band is high (≈ {_sf(low_med,2)} s), indicating excess low-frequency energy")
        if _is_high(mid_med):  flags.append(f"Mid band is high (≈ {_sf(mid_med,2)} s)")
        if _is_high(high_med): flags.append(f"High band is high (≈ {_sf(high_med,2)} s)")

        if flags:
            band_txt = ", ".join(flags) + "."
        else:
            if spectral == "boomy" and math.isfinite(low_med) and math.isfinite(high_med) and abs(low_med - high_med) >= 0.20:
                band_txt = f"Energy leans to low frequencies (Low = {_sf(low_med,2)} s, High = {_sf(high_med,2)} s)."
            elif spectral == "bright" and math.isfinite(low_med) and math.isfinite(high_med) and abs(low_med - high_med) >= 0.20:
                band_txt = f"Energy leans to high frequencies (High = {_sf(high_med,2)} s, Low = {_sf(low_med,2)} s)."
            else:
                band_txt = "Spectral balance is even."

        bullets.append(f"{main}, {band_txt}")

    # ------------------------------
    # 3) Clarity + background (sentence first; values in parentheses with equals)
    # ------------------------------
    def _qual_c50(v):  # gentle adjectives
        return "high" if (math.isfinite(v) and v > 3.0) else ("moderate" if (math.isfinite(v) and v >= 0.0) else "low")
    def _qual_c80(v):
        return "high" if (math.isfinite(v) and v > 2.0) else ("moderate" if (math.isfinite(v) and v >= 0.0) else "low")
    def _qual_l90(v):
        if not math.isfinite(v): return ""
        if v >= 55: return "elevated"
        if v <= 40: return "low"
        return "moderate"

    parts = []
    if math.isfinite(c50_db) or math.isfinite(c80_db):
        c50_txt = f"C50 = {_sf(c50_db,2)} dB" if math.isfinite(c50_db) else "C50 = —"
        c80_txt = f"C80 = {_sf(c80_db,2)} dB" if math.isfinite(c80_db) else "C80 = —"
        parts.append(f"Speech clarity is {_qual_c50(c50_db)} and music clarity is {_qual_c80(c80_db)} ({c50_txt}, {c80_txt})")
    if math.isfinite(l90_db):
        parts.append(f"Background sound is {_qual_l90(l90_db)} (L90 = {_sf(l90_db,0)} dB)")

    if parts:
        # join with comma, end with period
        bullets.append(", ".join(parts) + ".")

    # # ------------------------------
    # # Actions
    # # ------------------------------
    # if math.isfinite(rt60_use) and target:
    #     lo, hi = target
    #     if rt60_use > hi + 0.05:
    #         actions.append("Add 2–4 m² of soft absorption to reduce decay.")
    #     elif rt60_use < lo - 0.05:
    #         actions.append("Use fewer soft absorbers or add diffusion for more presence.")
    # if qc == "weak" or label in ("CHECK","LOW"):
    #     actions.append("Retest with a longer, quieter take and steady phone placement")

    # # Append “see Info” note for CHECK/LOW on the last action
    # if label in ("CHECK","LOW") and actions:
    #     if not actions[-1].endswith(")."):
    #         actions[-1] += " (see Info for recording tips)."

    # Badge — disable to avoid any duplication in UI
    badge_reason = None

    return {
        "badge_reason": badge_reason,
        "bullets": bullets[:3],
        # "actions": actions[:2],
        "scores": {
            "rt60_s": rt60_use if math.isfinite(rt60_use) else None,
            "edt_s":  edt_s if math.isfinite(edt_s) else None,
            "c50_db": c50_db if math.isfinite(c50_db) else None,
            "c80_db": c80_db if math.isfinite(c80_db) else None,
            "l90_db": l90_db if math.isfinite(l90_db) else None,
            "spectral": spectral,
            "qc": {"r2": r2 if (r2 is not None and math.isfinite(r2)) else None,
                   "snr_db": snr_db if (snr_db is not None and math.isfinite(snr_db)) else None,
                   "capped": bool(capped)},
            "target": _resolve_target(context.get("use"), context.get("goal")) if target else None,
            "rt60_source": src,
        }
    }
