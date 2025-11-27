import argparse, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import glob

def find_dataset(data_dir: Path):
    # 1) Prefer explicit files
    for fname in ["dataset.csv", "dataset.parquet"]:
        p = data_dir / fname
        if p.exists():
            return p
    # 2) Recursively look under ./data/**/dataset.csv|parquet and pick newest
    cands = []
    for pat in ["**/dataset.csv", "**/dataset.parquet"]:
        cands += list((data_dir).rglob(pat))
    if not cands:
        raise FileNotFoundError("Could not find dataset.csv/parquet under {}".format(data_dir))
    return max(cands, key=lambda p: p.stat().st_mtime)

def read_table(p: Path) -> pd.DataFrame:
    if p.suffix.lower()==".parquet":
        return pd.read_parquet(p)
    return pd.read_csv(p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data", help="Folder that contains dataset.csv (or subfolders).")
    ap.add_argument("--models", default="models/phone", help="Folder with regressor.joblib")
    ap.add_argument("--out", default="slides/hard_subset_stability.png")
    ap.add_argument("--table-out", default="slides/hard_subset_snapshot.csv")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    models_dir = Path(args.models)
    slides_dir = Path(args.out).parent
    slides_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = find_dataset(data_dir)
    print(f"[i] Using dataset: {dataset_path}")
    df = read_table(dataset_path)

    # optional bands file (for extra features) if present next to dataset
    bands_path_csv = dataset_path.parent / "rt60_bands.csv"
    bands_path_parq = dataset_path.parent / "rt60_bands.parquet"
    if bands_path_csv.exists():
        bands = pd.read_csv(bands_path_csv)
    elif bands_path_parq.exists():
        bands = pd.read_parquet(bands_path_parq)
    else:
        bands = None

    if bands is not None and "file" in df.columns and "file" in bands.columns:
        # Construct simple aggregates used in training
        b = bands.copy()
        b["is_ok"] = (b["quality"].str.upper()=="OK").astype(int, errors="ignore")
        b["is_check"] = (b["quality"].str.upper()=="CHECK").astype(int, errors="ignore")
        b["is_low"] = (b["quality"].str.upper()=="LOW").astype(int, errors="ignore")
        b["rt60_valid"] = pd.to_numeric(b["rt60_s"], errors="coerce")
        ag = b.groupby("file").agg(
            rt60_band_mean=("rt60_valid","mean"),
            rt60_band_median=("rt60_valid","median"),
            n_band_ok=("is_ok","sum"),
            n_band_check=("is_check","sum"),
            n_band_low=("is_low","sum"),
        ).reset_index()
        df = df.merge(ag, on="file", how="left")

    # Features (only those that exist)
    candidate_features = [
        "edt_s","c50_db","c80_db","r2","snr_db",
        "spl_leq_db","spl_l90_db","capped",
        "rt60_band_mean","rt60_band_median","n_band_ok","n_band_check","n_band_low",
        "rt60_low_med","rt60_mid_med","rt60_high_med",
    ]
    features = [f for f in candidate_features if f in df.columns]

    # Hard subset: noisy / weak fit / capped
    df["capped_bool"] = df.get("capped").astype(bool, errors="ignore") if "capped" in df.columns else False
    mask = pd.Series(False, index=df.index)
    if "r2" in df.columns:
        mask |= df["r2"] < 0.95
    if "snr_db" in df.columns:
        mask |= df["snr_db"] < 10
    mask |= df["capped_bool"]
    hard = df[mask].copy()
    hard = hard.replace([np.inf,-np.inf], np.nan).dropna(subset=["rt60_s"])
    if hard.empty:
        raise RuntimeError("Hard subset is empty; relax thresholds or check dataset.")

    # Model
    reg = joblib.load(models_dir / "regressor.joblib")

    X = hard[features].astype(float).fillna(hard[features].median(numeric_only=True))
    p_ml = reg.predict(X)

    def fuse_rt60(rt_meas, r2, snr, capped, rt_ml):
        try:
            if (not capped) and (float(r2)>=0.90) and (float(snr)>=6.0):
                return float(rt_meas)
        except Exception:
            pass
        r2 = float(r2) if pd.notna(r2) else 0.0
        alpha_r2 = max(0.0, min(1.0, (0.90 - r2) / 0.20))
        alpha_cap = 0.6 if bool(capped) else 0.0
        alpha = max(alpha_r2, alpha_cap)
        return float((1.0 - alpha)*float(rt_meas) + alpha*float(rt_ml))

    hard["rt60_fused"] = [
        fuse_rt60(m, r2, s, c, ml)
        for m,r2,s,c,ml in zip(hard["rt60_s"], hard.get("r2",0), hard.get("snr_db",0), hard["capped_bool"], p_ml)
    ]

    # Median stability vs measurement (no external ground truth)
    delta = np.abs(hard["rt60_fused"] - hard["rt60_s"])
    med_delta = float(np.median(delta))

    import matplotlib.pyplot as plt
    plt.figure(figsize=(5.8,4))
    bars = plt.bar(["Rules-only", "Rules+ML (fused)"], [0.0, med_delta])
    plt.title(f"Hard clips (n={len(hard)}): stability vs measured RT60")
    plt.ylabel("Median |fused − measured| (s)")
    plt.ylim(0, max(0.4, med_delta*1.5))
    for b,v in zip(bars, [0.0, med_delta]):
        plt.text(b.get_x()+b.get_width()/2, v+0.01, f"{v:.2f}s", ha="center", va="bottom")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.out, dpi=160); plt.close()

    # small snapshot table for the slide notes
    keep_cols = [c for c in ["file","r2","snr_db","capped","rt60_s","rt60_fused"] if c in hard.columns]
    hard[keep_cols].head(12).to_csv(args.table_out, index=False)
    print(f"[i] Saved {args.out} and {args.table_out}")

if __name__ == "__main__":
    main()
