import pandas as pd, matplotlib.pyplot as plt
from pathlib import Path

p = Path("metrics_history.csv")
df = pd.read_csv(p)

# Robust parsing + sane bounds
if "stamp" in df.columns:
    df["ts"] = pd.to_datetime(df["stamp"], errors="coerce")
else:
    df["ts"] = range(len(df))

df["mae_rt60_s"] = pd.to_numeric(df.get("mae_rt60_s"), errors="coerce").clip(lower=0)
df["f1_macro_quality"] = pd.to_numeric(df.get("f1_macro_quality"), errors="coerce").clip(0,1)
df["tag"] = df.get("tag", "")

# Short label for x-axis
def short_label(row):
    if pd.notna(row["ts"]):
        return row["ts"].strftime("%d %b %H:%M")  # e.g., 16 Oct 09:56
    return str(row.name)

df["label"] = df.apply(short_label, axis=1)

# ---- Quality Macro-F1
plt.figure(figsize=(7.5,4))
plt.plot(df["label"], df["f1_macro_quality"], marker="o")
plt.title("Quality Macro-F1 over runs")
plt.xlabel("Run time")
plt.ylabel("Macro-F1 (0–1)")
plt.ylim(0.6, 1.02)          # keep focus where action is
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
Path("slides").mkdir(exist_ok=True)
plt.savefig("slides/quality_f1_history_pretty.png", dpi=160); plt.close()

# ---- RT60 MAE
plt.figure(figsize=(7.5,4))
plt.plot(df["label"], df["mae_rt60_s"], marker="o")
plt.title("RT60 MAE over runs")
plt.xlabel("Run time")
plt.ylabel("MAE (seconds)")
plt.ylim(0, max(0.5, df["mae_rt60_s"].max()*1.1))
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30, ha="right")
plt.tight_layout()
plt.savefig("slides/rt60_mae_history_pretty.png", dpi=160); plt.close()

print("Saved slides/quality_f1_history_pretty.png and slides/rt60_mae_history_pretty.png")
