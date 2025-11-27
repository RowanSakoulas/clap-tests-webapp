# scripts/predict_from_dataset_row.py
from pathlib import Path
import argparse, json
import numpy as np
import pandas as pd
import joblib

def load_feature_list(models_dir: Path, fallback):
    m = models_dir / "metrics.csv"
    if m.exists():
        try:
            row = pd.read_csv(m).iloc[0]
            feats_txt = str(row.get("features_used", ""))
            feats = [f for f in feats_txt.split("|") if f]
            if feats:
                return feats
        except Exception:
            pass
    return fallback

FALLBACK_FEATURES = [
    "edt_s","c50_db","c80_db","r2","snr_db","spl_leq_db","spl_l90_db","capped",
    "rt60_band_mean","rt60_band_median","n_band_ok","n_band_check","n_band_low",
    "rt60_low_med","rt60_mid_med","rt60_high_med",
]

def add_band_aggregates(df: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    # Bring in rt60_bands.csv/parquet if present and build aggregates per file
    bcsv = data_dir / "rt60_bands.csv"
    bpar = data_dir / "rt60_bands.parquet"
    if not bcsv.exists() and not bpar.exists():
        # columns might already be present; just return
        return df
    bands = pd.read_csv(bcsv) if bcsv.exists() else pd.read_parquet(bpar)
    b = bands.copy()
    b["is_ok"]    = (b["quality"].str.upper()=="OK").astype(int, errors="ignore")
    b["is_check"] = (b["quality"].str.upper()=="CHECK").astype(int, errors="ignore")
    b["is_low"]   = (b["quality"].str.upper()=="LOW").astype(int, errors="ignore")
    b["rt60_valid"] = pd.to_numeric(b["rt60_s"], errors="coerce")
    ag = b.groupby("file").agg(
        rt60_band_mean=("rt60_valid","mean"),
        rt60_band_median=("rt60_valid","median"),
        n_band_ok=("is_ok","sum"),
        n_band_check=("is_check","sum"),
        n_band_low=("is_low","sum"),
        # optional low/mid/high if you trained with them:
        rt60_low_med=("rt60_valid", lambda s: np.nan),   # keep placeholders; your model may not require these
        rt60_mid_med=("rt60_valid", lambda s: np.nan),
        rt60_high_med=("rt60_valid", lambda s: np.nan),
    ).reset_index()
    return df.merge(ag, on="file", how="left")

def coerce_bool(x):
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if pd.isna(x): return False
    # strings like "True"/"False"/"1"/"0"
    s = str(x).strip().lower()
    if s in {"1","true","t","yes","y"}: return True
    return False

def blend_rt60(meas_rt60, r2, snr_db, capped, pred_rt60):
    try:
        if (not bool(capped)) and (float(r2) >= 0.90) and (float(snr_db) >= 6.0):
            return float(meas_rt60)
    except Exception:
        pass
    r2 = 0.0 if (r2 is None or not np.isfinite(r2)) else float(r2)
    alpha_r2  = max(0.0, min(1.0, (0.90 - r2) / 0.20))
    alpha_cap = 0.6 if bool(capped) else 0.0
    alpha = max(alpha_r2, alpha_cap)
    return (1.0 - alpha) * float(meas_rt60) + alpha * float(pred_rt60)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)          # e.g. data\phone_mic\dataset.csv
    ap.add_argument("--data-dir", required=False)        # e.g. data\phone_mic (to find rt60_bands.*)
    ap.add_argument("--models",  default="models/phone") # folder with regressor.joblib, classifier.joblib
    ap.add_argument("--file",    required=True)          # exact file name in the dataset
    ap.add_argument("--out",     required=True)          # path to write JSON
    args = ap.parse_args()

    dataset_path = Path(args.dataset)
    data_dir = Path(args.data_dir) if args.data_dir else dataset_path.parent
    models_dir = Path(args.models)

    df = pd.read_csv(dataset_path)
    # merge band aggregates if needed
    need_cols = {"rt60_band_mean","rt60_band_median","n_band_ok","n_band_check","n_band_low"}
    if not need_cols.issubset(df.columns):
        df = add_band_aggregates(df, data_dir)

    # locate row
    row = df.loc[df["file"] == args.file]
    if row.empty:
        raise SystemExit(f"File not found in dataset: {args.file}")
    r = row.iloc[0].to_dict()

    # features expected by the trained model
    features = load_feature_list(models_dir, FALLBACK_FEATURES)

    # ensure all expected columns exist; if missing, create with dataset medians or zeros
    for f in features:
        if f not in df.columns:
            if f in {"n_band_ok","n_band_check","n_band_low"}:
                df[f] = 0
            else:
                df[f] = np.nan

    # build X with medians for NaNs
    med = {f: float(np.nanmedian(df[f].astype(float))) for f in features}
    X = pd.DataFrame([{k: r.get(k, np.nan) for k in features}]).astype(float).fillna(med)

    # load models
    reg = joblib.load(models_dir / "regressor.joblib")
    cls = joblib.load(models_dir / "classifier.joblib")

    rt60_meas = float(r["rt60_s"]) if pd.notna(r["rt60_s"]) else None
    rt60_pred = float(reg.predict(X)[0])
    q_pred    = str(cls.predict(X)[0])
    try:
        proba = cls.predict_proba(X)[0]
        q_probs = {str(c): float(p) for c,p in zip(cls.classes_, proba)}
    except Exception:
        q_probs = {}

    r2      = r.get("r2", None)
    snr_db  = r.get("snr_db", None)
    capped  = coerce_bool(r.get("capped", False))
    rt60_fused = None if rt60_meas is None else blend_rt60(rt60_meas, r2, snr_db, capped, rt60_pred)

    out = {
        "file": args.file,
        "rt60_measured": rt60_meas,
        "rt60_predicted": rt60_pred,
        "rt60_fused": rt60_fused,
        "quality_pred": q_pred,
        "quality_probs": q_probs,
        "qc": {"r2": r2, "snr_db": snr_db, "capped": capped},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
