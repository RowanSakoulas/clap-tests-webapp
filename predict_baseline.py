#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import numpy as np, pandas as pd, joblib
from scipy.stats import linregress

# Repo-local imports
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import estimate_rt60 as rt
import estimate_spl as spl
import feedback as FB

# ---- feature lists (aligned to your training) ----
FEATURES_BASE = [
    "edt_s","c50_db","c80_db","r2","snr_db","spl_leq_db","spl_l90_db","capped"
]
BAND_FEATURES = [
    "rt60_band_mean","rt60_band_median","n_band_ok","n_band_check","n_band_low",
    "rt60_low_med","rt60_mid_med","rt60_high_med"
]

def _safe_float(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else float('nan')
    except:  # noqa: E722
        return float('nan')

def _center_freq(lo:float, hi:float)->float:
    try:
        return (float(lo)*float(hi))**0.5
    except:  # noqa: E722
        return 0.0

def _band_quality(method:str, r2:float, ok_thresh:float)->str:
    if method != "REG":
        return "CHECK"
    try:
        return "OK" if float(r2) >= ok_thresh else "CHECK"
    except:  # noqa: E722
        return "CHECK"

# ---- core: compute features for 1 file (no plotting args) ----
def compute_features_for_file(
    path: Path,
    save_plots: bool = False,
    graph_stem: str | None = None,
    preset: str = "",
    band_limit: bool = False,
) -> dict:
    """
    Run RT60 + SPL + band analysis for a single file.

    If save_plots=True, also write regression + spectrogram PNGs using the
    same logic as estimate_rt60.main(), and return their URLs in a "graphs"
    field.
    """
        # Configure RT60 engine for this call, if supported
    if hasattr(rt, "set_preset"):
        try:
            rt.set_preset(preset, bool(band_limit))
        except Exception:
            # Fall back to module defaults on any error
            pass

    # ---------- Full-band RT60 + ancillary ----------
    out = rt.estimate_rt60(path)
    if out is None:
        raise RuntimeError("RT60 analysis failed")

    rt60, r2, idxs, decay, sr, c50, c80, qflag, method, snr, edt_s, capped = out

    full: dict[str, object] = dict(
        rt60_s=_safe_float(rt60),
        r2=_safe_float(r2),
        edt_s=_safe_float(edt_s),
        c50_db=_safe_float(c50),
        c80_db=_safe_float(c80),
        snr_db=_safe_float(snr),
        capped=bool(capped),
    )

    # ---------- SPL summary (robust, smoothed) ----------
    try:
        y, sr_spl = spl.load_mono_f32(path)
        from scipy.ndimage import uniform_filter1d
        y = uniform_filter1d(y, size=int(0.01 * sr_spl) + 1)
        leq, lmax, l10, l90, _, _ = spl.summarize_spl(y, sr_spl)
        full.update(
            spl_leq_db=_safe_float(leq),
            spl_l90_db=_safe_float(l90),
        )
    except Exception:
        full.update(
            spl_leq_db=float("nan"),
            spl_l90_db=float("nan"),
        )

    # ---------- Band RTs (median + counts + low/mid/high) ----------
    y0, sr0 = rt.load_mono_f32(path)
    preset = rt.get_band_preset(int(sr0))
    if getattr(rt, "APPLY_BANDLIMIT", False):
        glo, ghi = getattr(rt, "BANDLIMIT_RANGE", (20.0, 20000.0))
        preset = [
            (max(lo, glo), min(hi, ghi), lab)
            for (lo, hi, lab) in preset
            if min(hi, ghi) > max(lo, glo)
        ]

    vals: list[float] = []
    lows: list[float] = []
    meds: list[float] = []
    highs: list[float] = []
    cnt_ok = cnt_check = cnt_low = 0
    band_rows: list[dict] = []

    for (lo, hi, lab) in preset:
        try:
            rtb, r2b, edt_b, c50_b, methb, capped_b = rt.estimate_rt60_for_band(
                y0, int(sr0), float(lo), float(hi)
            )
            cf = _center_freq(lo, hi)
            tag = rt.format_band_tag(lo, hi, lab)

            # Record row regardless, so UI shows all bands
            row = {
                "tag": tag,
                "lo_hz": float(lo),
                "hi_hz": float(hi),
                "rt60_s": _safe_float(rtb) if (rtb is not None and math.isfinite(rtb)) else None,
                "edt_s": _safe_float(edt_b),
                "c50_db": _safe_float(c50_b),
                "r2": _safe_float(r2b),
            }
            band_rows.append(row)

            # If RT60 is missing, treat as LOW and skip stats
            if rtb is None or (not math.isfinite(rtb)) or rtb <= 0:
                cnt_low += 1
                continue

            v = float(rtb)
            vals.append(v)
            if cf < 250:
                lows.append(v)
            elif cf <= 2000:
                meds.append(v)
            else:
                highs.append(v)

            q = _band_quality(methb, r2b, getattr(rt, "QUALITY_MIN_R2", 0.90))
            if q == "OK":
                cnt_ok += 1
            elif q == "CHECK":
                cnt_check += 1
            else:
                cnt_low += 1

        except Exception:
            cnt_low += 1
            band_rows.append({
                "tag": rt.format_band_tag(lo, hi, lab),
                "lo_hz": float(lo),
                "hi_hz": float(hi),
                "rt60_s": None,
                "edt_s": None,
                "c50_db": None,
                "r2": None,
            })

    full.update(
        rt60_band_mean=_safe_float(np.nanmean(vals) if len(vals) else float("nan")),
        rt60_band_median=_safe_float(np.nanmedian(vals) if len(vals) else float("nan")),
        n_band_ok=int(cnt_ok),
        n_band_check=int(cnt_check),
        n_band_low=int(cnt_low),
        rt60_low_med=_safe_float(np.nanmedian(lows) if len(lows) else float("nan")),
        rt60_mid_med=_safe_float(np.nanmedian(meds) if len(meds) else float("nan")),
        rt60_high_med=_safe_float(np.nanmedian(highs) if len(highs) else float("nan")),
    )

    full["rt60_bands"] = band_rows

    # ---------- Optional plots (regression + spectrogram) ----------
    graphs: dict[str, str] = {}
    if save_plots:
        try:
            i0, i1 = idxs
            stem = graph_stem or path.stem
            if i1 > i0:
                t = np.arange(len(decay)) / sr
                slope, inter, *_ = linregress(t[i0:i1], decay[i0:i1])
                rt.save_plot(stem, decay, sr, idxs, slope, inter)

            # Spectrogram using same band-limited + preprocessed signal
            y0s, sr0 = rt.load_mono_f32(path)
            ys = (
                rt.bandlimit(y0s, sr0, *rt.BANDLIMIT_RANGE)
                if getattr(rt, "APPLY_BANDLIMIT", False)
                else y0s
            )
            ys = rt.preprocess(
                ys,
                sr0,
                normalize=getattr(rt, "APPLY_NORMALIZE", True),
                trim_to_peak=True,
            )
            spec = rt.compute_spectrogram(ys, sr0)
            fmax_hint = (
                rt.BANDLIMIT_RANGE[1]
                if getattr(rt, "APPLY_BANDLIMIT", False)
                else None
            )
            rt.plot_spectrogram_png(stem, spec, fmax_hint)

            graphs = {
                "regression": f"/graphs/{stem}_rt60.png",
                "spectrogram": f"/graphs/{stem}_spec.png",
            }
        except Exception:
            graphs = {}

    full["graphs"] = graphs
    return full

def load_models(models_dir:Path):
    reg_p = models_dir/'regressor.joblib'
    cls_p = models_dir/'classifier.joblib'
    if not reg_p.exists() or not cls_p.exists():
        raise FileNotFoundError(f"Model files not found in {models_dir}")
    reg = joblib.load(reg_p)
    cls = joblib.load(cls_p)

    features = FEATURES_BASE + BAND_FEATURES
    met = models_dir/'metrics.csv'
    if met.exists():
        try:
            m = pd.read_csv(met).iloc[0]
            feats_txt = str(m.get('features_used',''))
            feats = [f for f in feats_txt.split('|') if f]
            if feats: features = feats
        except Exception:
            pass
    return reg, cls, features

def blend_rt60(meas_rt60:float, r2:float, snr_db:float, capped:bool, pred_rt60:float)->float:
    # Trust measurement when QC good
    if (not capped) and (r2 is not None and r2>=0.90) and (snr_db is not None and snr_db>=6.0):
        return float(meas_rt60)
    # Otherwise blend toward model prediction
    r2 = float(r2) if (r2 is not None and math.isfinite(r2)) else 0.0
    alpha_r2 = max(0.0, min(1.0, (0.90 - r2) / 0.20))
    alpha_cap = 0.6 if capped else 0.0
    alpha = max(alpha_r2, alpha_cap)
    return float((1.0 - alpha) * float(meas_rt60) + alpha * float(pred_rt60))

def tips_from_features(f:dict, rt60_pred:float, q_pred:str, q_probs:dict)->list[str]:
    tips = []
    rt60_meas = f.get('rt60_s', float('nan'))
    r2 = f.get('r2', float('nan'))
    snr = f.get('snr_db', float('nan'))
    capped = bool(f.get('capped', False))
    rt60_use = blend_rt60(rt60_meas, r2, snr, capped, rt60_pred)

    # Measurement quality
    if capped or (not math.isfinite(r2)) or (r2 < 0.90) or (q_pred == 'LOW'):
        tips.append('Estimate may be unreliable. Retest with a longer, quieter recording and stable phone placement.')
        if math.isfinite(rt60_meas) and abs(rt60_meas - rt60_pred) > 0.25:
            tips.append(f'Model suggests RT60 ≈ {rt60_pred:.2f} s; longer take will improve confidence.')

    # Spectral balance
    low  = f.get('rt60_low_med', float('nan'))
    high = f.get('rt60_high_med', float('nan'))
    if math.isfinite(rt60_use) and math.isfinite(low) and math.isfinite(high):
        lo_ok  = 0.3*rt60_use <= low  <= 3.0*rt60_use
        hi_ok  = 0.3*rt60_use <= high <= 3.0*rt60_use
        if lo_ok and hi_ok:
            d = low - high
            if d >= 0.15:
                tips.append(f'Low-frequency reverberation dominates (boomy): low≈{low:.2f}s vs high≈{high:.2f}s. Add bass traps or thicker absorbers.')
            elif d <= -0.15:
                tips.append(f'High-frequency reverberation dominates (bright): high≈{high:.2f}s vs low≈{low:.2f}s. Add curtains or soft panels.')

    # Global decay
    if math.isfinite(rt60_use) and rt60_use >= 1.2:
        tips.append('Highly reverberant; add absorption (panels, curtains, carpet).')
    elif math.isfinite(rt60_use) and rt60_use < 0.3:
        tips.append("Very dry; consider diffusion/reflective elements if you want more 'room feel'.")
    else:
        tips.append('Balanced decay for speech/music practice.')

    # Background noise
    l90 = f.get('spl_l90_db', float('nan'))
    if math.isfinite(l90) and l90 >= 50.0:
        tips.append('High background noise (L90); retest when quieter for cleaner estimates.')

    return tips[:3]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('audio')
    ap.add_argument('--models', default='models/phone')
    ap.add_argument('--json', action='store_true')
    ap.add_argument('--ignore-spl', action='store_true')
    ap.add_argument('--save-plots', action='store_true',
                    help='Ask estimate_rt60.py to save decay+spectrogram plots')
    ap.add_argument('--graphs-dir', default='graphs',
                    help='Where plots are written when --save-plots is used')
    ap.add_argument('--context-use', default='', help='Intended use (e.g., speech, music)')
    ap.add_argument('--context-goal', default='', help='Goal (e.g., clear, lively)')
    args = ap.parse_args()

    # Enable internal plotting in estimate_rt60.py
    if args.save_plots:
        if hasattr(rt, 'MAKE_PLOTS'):
            rt.MAKE_PLOTS = True
        Path(args.graphs_dir).mkdir(parents=True, exist_ok=True)
        if hasattr(rt, 'OUTPUT_DIR'):
            rt.OUTPUT_DIR = Path(args.graphs_dir)

    audio_path = Path(args.audio)
    models_dir = Path(args.models)

    # Fail-safe JSON emitter
    def emit_fail_json(err_msg):
        return {
            "file": str(audio_path),
            "error": f"rt60_analysis_failed: {err_msg}",
            "rt60_measured": None,
            "rt60_predicted": None,
            "rt60_fused": None,
            "quality_pred": "LOW",
            "quality_probs": {"LOW": 1.0, "CHECK": 0.0, "OK": 0.0},
            "qc": {"r2": None, "snr_db": None, "capped": True},
            "tips": ["Refer to app info page to improve recording"]
        }

    # Compute features (graceful failure)
    try:
        f = compute_features_for_file(
            audio_path,
            save_plots=args.save_plots,
            graph_stem=audio_path.stem,
        )
    except Exception as e:
        # Try to at least save a spectrogram, so you have a slide asset
        if args.save_plots:
            try:
                y0, sr0 = rt.load_mono_f32(audio_path)
                ys = rt.bandlimit(y0, sr0, *rt.BANDLIMIT_RANGE) if getattr(rt, 'APPLY_BANDLIMIT', False) else y0
                ys = rt.preprocess(ys, sr0, normalize=getattr(rt,'APPLY_NORMALIZE', True), trim_to_peak=True)
                spec = rt.compute_spectrogram(ys, sr0)
                fmax_hint = rt.BANDLIMIT_RANGE[1] if getattr(rt,'APPLY_BANDLIMIT', False) else None
                rt.plot_spectrogram_png(audio_path.stem, spec, fmax_hint)
            except Exception:
                pass
        print(json.dumps(emit_fail_json(str(e)), indent=2))
        return

    if args.ignore_spl:
        f['spl_leq_db'] = float('nan')
        f['spl_l90_db'] = float('nan')

    # Load models and predict
    reg, cls, features = load_models(models_dir)
    row = {k: f.get(k, float('nan')) for k in features}
    X = pd.DataFrame([row]).astype(float)
    rt60_pred = float(reg.predict(X)[0])

    try:
        proba = cls.predict_proba(X)[0]; classes = list(cls.classes_)
        q_pred = str(cls.predict(X)[0]); q_probs = {str(c): float(p) for c,p in zip(classes, proba)}
    except Exception:
        q_pred = str(cls.predict(X)[0]); q_probs = {}

    # Fused output for stability on weak QC
    rt60_meas = f.get('rt60_s', float('nan'))
    rt60_fused = blend_rt60(rt60_meas, f.get('r2', float('nan')), f.get('snr_db', float('nan')),
                            bool(f.get('capped', False)), rt60_pred)

    out = {
        "file": str(audio_path),
        "rt60_measured": None if not math.isfinite(rt60_meas) else float(rt60_meas),
        "rt60_predicted": rt60_pred,
        "rt60_fused": None if not math.isfinite(rt60_fused) else float(rt60_fused),
        "quality_pred": q_pred,
        "quality_probs": q_probs,
        "qc": {"r2": f.get('r2', None), "snr_db": f.get('snr_db', None), "capped": bool(f.get('capped', False))},
        "features_used": features,
        "rt60_low_med": f.get('rt60_low_med', None),
        "rt60_mid_med": f.get('rt60_mid_med', None),
        "rt60_high_med": f.get('rt60_high_med', None),
        "direct_feedback": FB.direct_feedback(f, {"rt60_pred": rt60_pred, "label": q_pred}, {"use": args.context_use, "goal": args.context_goal}),
            "tips": tips_from_features(f, rt60_pred, q_pred, q_probs),
    }

    if args.json:
        print(json.dumps(out, indent=2))
    else:
        print(f"\nFile: {audio_path}")
        print(f"Measured RT60: {out['rt60_measured'] if out['rt60_measured'] is not None else 'NA'} s")
        print(f"Predicted RT60: {rt60_pred:.3f} s  |  Fused: {out['rt60_fused'] if out['rt60_fused'] is not None else 'NA'} s")
        print(f"Quality: {q_pred}  (probs: {q_probs})")
        print("Tips:")
        for t in out["tips"]:
            print(f" - {t}")
        print("\n(JSON below)")
        print(json.dumps(out, indent=2))

if __name__ == "__main__":
    main()
