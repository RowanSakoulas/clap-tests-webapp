from __future__ import annotations
import shutil, subprocess, tempfile
from pathlib import Path
import numpy as np
import soundfile as sf
from scipy.ndimage import uniform_filter1d
from scipy.signal import lfilter


# ================ Config ================
INPUT_DIR      = Path("sounds")
TARGET_FILE    = None                   # Path("sounds/pinknoisetest.mp3") to force a single file
WEIGHTING      = "A"                    # "A" or "Z"
CAL_DB_OFFSET  = 70.0 - 49.5            # add your calibration offset here (dB)
WINDOW_SEC     = 1.0                    # RMS window
HOP_SEC        = 0.5                    # RMS hop
PRINT_COMPACT  = False                  # True -> print only Leq
MAKE_PLOTS     = True                   # True -> save graphs/<stem>_spl.png
CSV_DELIM      = ", "                   # match RT file spacing
SUPPORTED_SF   = {".wav",".flac",".ogg",".aiff",".aif",".aifc"}
P0             = 20e-6                  # 20 µPa


# ================ Audio I/O ================
def _ffmpeg_to_wav(in_path: Path, sr: int|None=None, mono: bool=True) -> Path:
    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg not found on PATH.")
    tmp = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    cmd = ["ffmpeg","-y","-i",str(in_path)]
    if mono: cmd += ["-ac","1"]
    if sr:   cmd += ["-ar",str(sr)]
    cmd += [str(tmp)]
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    return tmp

def load_mono_f32(path: Path, target_sr: int|None=None) -> tuple[np.ndarray,int]:
    ext, tmp = path.suffix.lower(), None
    try:
        if ext in SUPPORTED_SF:
            y, sr = sf.read(str(path), always_2d=False)
        else:
            tmp = _ffmpeg_to_wav(path, sr=target_sr, mono=True)
            y, sr = sf.read(str(tmp), always_2d=False)
        if isinstance(y, np.ndarray) and y.ndim == 2:
            y = y.mean(axis=1)
        return y.astype(np.float32, copy=False), int(sr)
    finally:
        if tmp and tmp.exists():
            try: tmp.unlink()
            except OSError: pass


# ================ Weighting ================
def apply_weighting(y: np.ndarray, sr: int, mode: str="A") -> np.ndarray:
    if mode.upper() != "A":
        return y
    # Try Acoustics' A-weighting; fall back to passthrough if unavailable
    try:
        from acoustics.signal import A_weighting
        b, a = A_weighting(sr)
        return lfilter(b, a, y).astype(np.float32, copy=False)
    except Exception:
        return y


# ================ SPL Core ================
def frame_rms(y: np.ndarray, sr: int, win_s: float=1.0, hop_s: float=0.5) -> tuple[np.ndarray,np.ndarray]:
    n = max(1, int(round(win_s * sr)))
    h = max(1, int(round(hop_s * sr)))
    if len(y) < n:
        return np.array([]), np.array([])
    rms = []
    for i in range(0, len(y) - n + 1, h):
        w = y[i:i+n]
        rms.append(float(np.sqrt(np.mean(w*w)) + 1e-20))
    t = (np.arange(len(rms)) * h + (n/2)) / sr
    return np.array(t), np.array(rms)

def db_from_rms(rms: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(rms / P0 + 1e-20) + CAL_DB_OFFSET

def summarize_spl(y: np.ndarray, sr: int):
    yw = apply_weighting(y, sr, WEIGHTING)
    # Leq over full clip
    leq_db = 10.0 * np.log10(np.mean(yw*yw) / (P0*P0) + 1e-20) + CAL_DB_OFFSET
    # Rolling RMS trace
    t, rms = frame_rms(yw, sr, WINDOW_SEC, HOP_SEC)
    if rms.size == 0:
        return leq_db, np.nan, np.nan, np.nan, t, np.array([])
    db_trace = db_from_rms(rms)
    lmax = float(np.max(db_trace))
    l10  = float(np.percentile(db_trace, 90))
    l90  = float(np.percentile(db_trace, 10))
    return leq_db, lmax, l10, l90, t, db_trace


# ================ Plotting ================
def save_plot(stem: str, t: np.ndarray, db_trace: np.ndarray):
    if not MAKE_PLOTS or db_trace.size == 0:
        return
    import matplotlib.pyplot as plt
    out_dir = Path("graphs"); out_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9,3.5), dpi=150)
    plt.plot(t, db_trace)
    plt.xlabel("Time (s)"); plt.ylabel("SPL (dB)")
    plt.title("A-weighted time trace")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / f"{stem}_spl.png")
    plt.close()


# ================ Run ================
def main():
    files = [TARGET_FILE] if TARGET_FILE else [
        p for p in sorted(Path(INPUT_DIR).iterdir())
        if p.is_file() and p.suffix.lower() in {".wav",".flac",".aiff",".aif",".ogg",".m4a",".mp3"}
    ]
    if not files:
        print("No audio files found."); return

    header = ["file","Leq_dB"] if PRINT_COMPACT else ["file","Leq_dB","Lmax_dB","L10_dB","L90_dB"]
    print(CSV_DELIM.join(header))

    for p in files:
        try:
            y, sr = load_mono_f32(Path(p))
            # tiny de-click / smooth (≈10 ms) before metering
            y = uniform_filter1d(y, size=int(0.01*sr)+1)
            leq, lmax, l10, l90, t, db_trace = summarize_spl(y, sr)
            if PRINT_COMPACT:
                row = [p.name, f"{leq:.1f}"]
            else:
                row = [p.name, f"{leq:.1f}", f"{lmax:.1f}", f"{l10:.1f}", f"{l90:.1f}"]
            print(CSV_DELIM.join(row))
            save_plot(Path(p).stem, t, db_trace)
        except Exception as e:
            print(f"{p.name}{CSV_DELIM}error: {e}")

if __name__ == "__main__":
    main()
