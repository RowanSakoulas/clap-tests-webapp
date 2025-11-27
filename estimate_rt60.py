from __future__ import annotations
import shutil, subprocess, tempfile
from pathlib import Path
from typing import Tuple
import numpy as np, soundfile as sf
from scipy.stats import linregress
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, stft


# ================ Config ================
INPUT_DIR  = Path("sounds")
OUTPUT_DIR = Path("graphs")

USER_LABEL = "ML 2"          # optional user label
BAND_PRESET = "Phone-safe"   # "Musician" | "Reference Octaves" | "Phone-safe"

MAKE_PLOTS  = True                  # save decay-fit plot + spectrogram
PRINT_EDT   = True                  # include EDT (0..-10 dB) on main row
DO_BAND_RTS = True                  # also print per-band RTs (rows)

APPLY_GATED_TRIM = True             # onset trim via PRE_PEAK_MS before peak
APPLY_BANDLIMIT  = False            # full-band analysis unless True
APPLY_NORMALIZE  = True             # RMS normalize before trimming
PRE_PEAK_MS      = 5.0              # audio kept before impulse (ms)

QUALITY_MIN_R2     = 0.95           # OK if r² >= this AND SNR >= threshold; else CHECK/LOW
QUALITY_MIN_SNR_DB = 6.0            # SNR gate for OK status

SPEC_NFFT, SPEC_HOP, SPEC_DB_FLOOR, SAVE_SPEC_NPZ = 2048, 512, -80.0, True

SUPPORTED_SF = {".wav",".flac",".ogg",".aiff",".aif",".aifc"} 

BAND_PRESET = "Phone-safe"   # "Musician" | "Reference Octaves" | "Phone-safe" | "speech"

# Resolve preset name and mode label
def _title_preset(n: str) -> str:
    n = (n or "").strip().lower()
    if n in ("musician", "music", "broadband"):
        return "Musician"
    if n in ("reference octaves", "reference", "octaves", "iso", "speech"):
        # Treat "speech" as the ISO octave layout
        return "Reference Octaves"
    if n in ("phone-safe", "phone safe", "phone"):
        return "Phone-safe"
    return "Musician"

# Runtime-configurable globals
PRESET_NAME = _title_preset(BAND_PRESET)
BANDLIMIT_RANGE = (200.0, 4000.0)
MODE_LABEL = ""
# keep existing APPLY_BANDLIMIT definition above; we will mutate it here

def set_preset(preset: str | None = None, band_limit: bool = False) -> None:
    """
    Update preset + band-limit for one analysis run.

    preset: "phone-safe", "speech", "musician", ""/None.
    band_limit: if True and preset is not Phone-safe, focus analysis on 200-4000 Hz.
    """
    global PRESET_NAME, BANDLIMIT_RANGE, APPLY_BANDLIMIT, MODE_LABEL, BAND_PRESET

    name = _title_preset(preset or BAND_PRESET)
    PRESET_NAME = name
    BAND_PRESET = name

    if name == "Phone-safe":
        # Phone-safe always uses 125–8000 Hz window
        APPLY_BANDLIMIT = True
        BANDLIMIT_RANGE = (125.0, 8000.0)
    else:
        if band_limit:
            # Speech-style focused band (200–4000 Hz)
            APPLY_BANDLIMIT = True
            BANDLIMIT_RANGE = (200.0, 4000.0)
        else:
            # Full band
            APPLY_BANDLIMIT = False
            BANDLIMIT_RANGE = (20.0, 20000.0)

    MODE_LABEL = (USER_LABEL.strip() + " · " + PRESET_NAME) if USER_LABEL.strip() else PRESET_NAME

# Initialise defaults for CLI usage
set_preset(BAND_PRESET, False)

# ================ Audio I/O ================
def _ffmpeg_to_wav(in_path: Path, sr: int|None=None, mono: bool=True) -> Path:
    if shutil.which("ffmpeg") is None: raise RuntimeError("ffmpeg not found on PATH.")
    tmp = Path(tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name)
    cmd = ["ffmpeg","-y","-i",str(in_path)] + (["-ac","1"] if mono else []) + (["-ar",str(sr)] if sr else []) + [str(tmp)]
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
        if isinstance(y, np.ndarray) and y.ndim == 2: y = y.mean(axis=1)
        return y.astype(np.float32, copy=False), int(sr)
    finally:
        if tmp and tmp.exists():
            try: tmp.unlink()
            except OSError: pass


# ================ Preprocessing ================
def preprocess(y: np.ndarray, sr: int, normalize: bool=APPLY_NORMALIZE, trim_to_peak: bool=True) -> np.ndarray:
    if normalize:
        rms = float(np.sqrt(np.mean(y*y)));  y = y/(rms+1e-12) if rms>0 else y
    if trim_to_peak:
        pk = int(np.argmax(np.abs(y))); pre = max(0, pk - int(round(PRE_PEAK_MS*1e-3*sr)))
        y = y[pre:]
    return y

def bandlimit(y: np.ndarray, sr: int, lo: float, hi: float) -> np.ndarray:
    b,a = butter(2, [lo/(sr*0.5), hi/(sr*0.5)], btype="band")
    return filtfilt(b,a,y).astype(np.float32, copy=False)

def impulse_noise_stats(y: np.ndarray, sr: int, env_win_ms: float=3, pre_win_ms: float=800) -> float:
    win=max(64,int(env_win_ms/1000*sr)); env=np.sqrt(uniform_filter1d(y*y,size=win))
    pk=int(np.argmax(env)); peak=float(env[pk])+1e-12; pre=env[max(0,pk-int(pre_win_ms/1000*sr)):pk]
    noise=float(np.median(pre))+1e-12 if pre.size else 1e-12
    return 20.0*np.log10(peak/noise)


# ================ Decay & RT Estimation ================
def schroeder_db_noisecomp(y: np.ndarray, sr: int, tail_sec: float=0.5) -> np.ndarray:
    y2 = (y.astype(np.float64)**2)
    tail = y2[-max(128,int(tail_sec*sr)):] if len(y2) else y2
    N = float(np.median(tail)) if tail.size else 0.0
    e = y2 - N; e[e<0]=0
    sch = np.cumsum(e[::-1])[::-1]; sch /= (sch.max()+1e-20)
    return 10.0*np.log10(np.maximum(sch,1e-20))

def _usable_end_idx_from_noise(sch_db: np.ndarray, margin_db: float=3.0) -> int:
    n = len(sch_db);  tail = sch_db[-max(50,n//10):] if n>=50 else sch_db
    floor_db = float(np.median(tail)) if tail.size else -120.0
    idx = np.where(sch_db > floor_db + margin_db)[0]
    return int(idx[-1]) if idx.size else max(0,n-1)

def fit_decay_auto(sch_db: np.ndarray, sr: int, min_samples: int=12):
    end_cap = _usable_end_idx_from_noise(sch_db, 3.0)
    t = np.arange(len(sch_db))/sr
    best=None
    for (d0,d1,prio) in [(-5,-35,0),(-5,-25,1),(0,-10,2)]:
        lo,hi = np.where(sch_db<=d0)[0], np.where(sch_db<=d1)[0]
        if lo.size==0 or hi.size==0: continue
        i0,i1 = int(lo[0]), min(int(hi[0]), end_cap)
        if i1<=i0 or (i1-i0)<min_samples: continue
        x,y = t[i0:i1], sch_db[i0:i1]
        slope,inter,r,_,_ = linregress(x,y)
        if not np.isfinite(slope) or slope>=0: continue
        r2, span = float(r*r), float(sch_db[i0]-sch_db[i1])
        cand=(prio,-r2,-span,(slope,inter,r2,i0,i1))
        if best is None or cand<best: best=cand
    return best[-1] if best else None

def rt60_from_crossing(sch_db: np.ndarray, sr: int, drop_db: float=60.0) -> float|None:
    target = -abs(drop_db)
    if sch_db.min()>target: return None
    return int(np.argmax(sch_db<=target))/sr

def rt60_from_fit_window(sch_db: np.ndarray, sr: int, d0: float, d1: float, min_samples: int=12):
    lo,hi = np.where(sch_db<=d0)[0], np.where(sch_db<=d1)[0]
    if lo.size==0 or hi.size==0: return None,None
    i0,i1 = int(lo[0]), int(hi[0])
    if i1<=i0 or (i1-i0)<min_samples: return None,None
    t = np.arange(len(sch_db))/sr; x,y=t[i0:i1], sch_db[i0:i1]
    slope,inter,r,_,_ = linregress(x,y)
    if not np.isfinite(slope) or slope>=0: return None,None
    return (-60.0/slope), float(r*r)

def compute_edt_seconds(decay_db: np.ndarray, sr: int) -> float|None:
    edt_t60,_ = rt60_from_fit_window(decay_db, sr, 0.0,-10.0);  return edt_t60

def clarity_cxx(y: np.ndarray, sr: int, split_ms: int=50, pre_ms: float=PRE_PEAK_MS) -> float:
    off = int(round(pre_ms*1e-3*sr)); n=max(1,int(round(split_ms*1e-3*sr))); y2=(y.astype(np.float32)**2)
    if off>=len(y2): return float("nan")
    early=float(np.sum(y2[off:off+n]))+1e-20; late=float(np.sum(y2[off+n:]))+1e-20
    return 10.0*np.log10(early/late)


# ================ Per-Band RTs (presets) ================
def get_band_preset(sr: int) -> list[tuple[float,float,str|None]]:
    nyq=0.95*0.5*sr
    def clamp(lo,hi):
        lo2,hi2=max(1.0,float(lo)),min(float(hi),nyq)
        return (lo2,hi2) if hi2>lo2 else None
    if PRESET_NAME=="Reference Octaves":
        k=2**0.5; centers=[31.5,63,125,250,500,1000,2000,4000,8000,16000]
        out=[];  [out.append((*b,None)) for fc in centers if (b:=clamp(fc/k,fc*k))]
        return out
    if PRESET_NAME=="Phone-safe":
        out=[];  [out.append((*b,None)) for b in [clamp(125,250),clamp(250,500),clamp(500,2000),clamp(2000,4000),clamp(4000,8000)] if b]
        return out
    bands=[(20,250,"Sub/Bass"),(250,500,"Low-Mid"),(500,2000,"Mid"),(2000,4000,"High-Mid"),(4000,6000,"Presence"),(6000,20000,"Air")]
    out=[];  [out.append((*b,lab)) for (lo,hi,lab) in bands if (b:=clamp(lo,hi))]
    return out

def _hr(lo: float, hi: float) -> str:
    s=lambda x: (f"{int(round(x))//1000}k" if x>=1000 else f"{int(round(x))}")
    return f"{s(lo)}-{s(hi)}"

def format_band_tag(lo: float, hi: float, lab: str|None) -> str:
    rng=_hr(lo,hi)
    return f"{lab} ({rng})" if (PRESET_NAME=="Musician" and lab) else rng

def estimate_rt60_for_band(y0: np.ndarray, sr: int, lo_hz: float, hi_hz: float):
    nyq = 0.95 * 0.5 * sr
    lo, hi = max(1.0, float(lo_hz)), min(float(hi_hz), nyq)
    if hi <= lo:
        return None, None, float("nan"), float("nan"), "NA", False

    # Band-limit + preprocess like the full-band path
    yb = bandlimit(y0, sr, lo, hi)
    yb = preprocess(yb, sr, normalize=APPLY_NORMALIZE, trim_to_peak=True)
    dur = len(yb) / sr
    if dur < max(256 / sr, 0.1):
        return None, None, float("nan"), float("nan"), "NA", False

    # Per-band clarity + decay curve
    c50_b = clarity_cxx(yb, sr, 50)
    decay = schroeder_db_noisecomp(yb, sr, 0.5)

    rt_b, r2_b, meth = None, np.nan, "NA"
    fit = fit_decay_auto(decay, sr)
    if fit is not None:
        slope, _, r2_b, i0, i1 = fit
        if np.isfinite(slope) and slope < 0:
            rt_b, meth = -60.0 / slope, "REG"

    if rt_b is None:
        rt = rt60_from_crossing(decay, sr, 60.0)
        if rt is not None:
            rt_b, meth = rt, "CROSS"

    if rt_b is None:
        edt_t60, r2e = rt60_from_fit_window(decay, sr, 0.0, -10.0)
        if edt_t60 is not None:
            rt_b, r2_b, meth = 6.0 * edt_t60, r2e, "EDT×6"
        else:
            t20, r2t = rt60_from_fit_window(decay, sr, -5.0, -25.0)
            if t20 is not None:
                rt_b, r2_b, meth = 3.0 * t20, r2t, "T20×3"

    if (rt_b is None) or (not np.isfinite(rt_b)):
        return None, None, float("nan"), float("nan"), "NA", False

    capped = (rt_b >= 0.95 * dur - 1e-6)
    rt_b = min(rt_b, 0.95 * dur)

    # EDT (s) in the same sense as the full-band estimate
    edt_b = compute_edt_seconds(decay, sr) if PRINT_EDT else float("nan")

    return rt_b, r2_b, edt_b, c50_b, meth, capped

# ================ Full-band RT60 Pipeline ================
def estimate_rt60(file_path: Path):
    y0,sr = load_mono_f32(Path(file_path))
    snr_db = impulse_noise_stats(y0,sr,pre_win_ms=800)

    y = bandlimit(y0,sr,*BANDLIMIT_RANGE) if APPLY_BANDLIMIT else y0
    y = preprocess(y,sr,normalize=APPLY_NORMALIZE,trim_to_peak=True)

    dur=len(y)/sr
    if dur<=0: return None

    c50,c80 = clarity_cxx(y,sr,50), clarity_cxx(y,sr,80)
    decay = schroeder_db_noisecomp(y,sr,0.5)

    r2,method,rt60,i0,i1 = np.nan,"REG",None,0,0
    fit=fit_decay_auto(decay,sr)
    if fit is not None:
        slope,inter,r2,i0,i1=fit
        if np.isfinite(slope) and slope<0: rt60=-60.0/slope

    if rt60 is None or not np.isfinite(rt60):
        rt=rt60_from_crossing(decay,sr,60.0)
        if rt is not None: rt60,method=rt,"CROSS"

    if rt60 is None:
        edt,r2e=rt60_from_fit_window(decay,sr,0.0,-10.0)
        if edt is not None: rt60,r2,method=6.0*edt,r2e,"EDT×6"
        else:
            t20,r2t=rt60_from_fit_window(decay,sr,-5.0,-25.0)
            if t20 is not None: rt60,r2,method=3.0*t20,r2t,"T20×3"

    if rt60 is None or not np.isfinite(rt60): return None
    rt60=min(rt60,0.95*dur);  capped=(rt60>=0.95*dur-1e-6)
    if rt60<0.1 or rt60>10.0: return None

    edt_s = compute_edt_seconds(decay,sr) if PRINT_EDT else None
    low_r2 = (np.isfinite(r2) and r2<QUALITY_MIN_R2); low_snr=(snr_db<QUALITY_MIN_SNR_DB); used_fb=(method!="REG")
    quality = (
        "LOW" if capped or (low_r2 and low_snr) or (used_fb and low_r2)
        else ("CHECK" if (low_r2 or low_snr or used_fb) else "OK")
    )

    return (rt60, r2, (i0,i1), decay, sr, c50, c80, quality, method, snr_db, edt_s, capped)


# ================ Spectrogram & plots ================
def compute_spectrogram(y: np.ndarray, sr: int):
    if y.size==0:
        return {"S_db":np.zeros((1,1),np.float32),"freqs":np.array([0.0],np.float32),"times":np.array([0.0],np.float32),"sr":int(sr)}
    nper,nover=int(SPEC_NFFT), int(SPEC_NFFT)-int(SPEC_HOP)
    f,t,Z = stft(y.astype(np.float32,copy=False), fs=sr, nperseg=nper, noverlap=nover, window="hann", padded=False, boundary=None)
    P=(np.abs(Z)**2).astype(np.float32); S_db=10.0*np.log10(np.maximum(P,1e-20)); S_db=np.clip(S_db,SPEC_DB_FLOOR,None).astype(np.float32)
    return {"S_db":S_db,"freqs":f.astype(np.float32),"times":t.astype(np.float32),"sr":int(sr)}

def plot_spectrogram_png(stem: str, spec: dict, fmax_hint: float|None):
    if not MAKE_PLOTS: return
    import matplotlib.pyplot as plt
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    S_db,f,t,sr = spec["S_db"], spec["freqs"], spec["times"], spec["sr"]
    nyq=0.5*sr*0.98; fmax=min(nyq,fmax_hint) if fmax_hint is not None else nyq; mask=f<=fmax; Sd,ff=S_db[mask,:],f[mask]
    plt.figure(figsize=(9,3.2), dpi=150)
    extent=[float(t[0] if t.size else 0.0), float(t[-1] if t.size else 0.0), float(ff[0] if ff.size else 0.0), float(ff[-1] if ff.size else fmax)]
    plt.imshow(Sd, origin="lower", aspect="auto", extent=extent, cmap="magma", vmin=SPEC_DB_FLOOR, vmax=Sd.max() if Sd.size else 0)
    plt.colorbar(label="Power (dB)"); plt.xlabel("Time (s)"); plt.ylabel("Frequency (Hz)")
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/f"{stem}_spec.png"); plt.close()

def save_plot(stem: str, decay: np.ndarray, sr: int, idxs: tuple[int,int], slope: float, intercept: float):
    if not MAKE_PLOTS: return
    import matplotlib.pyplot as plt
    t=np.arange(len(decay))/sr; i0,i1=idxs
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9,3.5))
    plt.plot(t,decay,label="Decay (dB)")
    if i1>i0:
        plt.plot(t[i0:i1], slope*t[i0:i1]+intercept, "r--", label="Fit")
    plt.xlabel("Time (s)"); plt.ylabel("dB"); plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR/f"{stem}_rt60.png", dpi=150); plt.close()


# ================ Main ================
def main():
    files=[p for p in sorted(Path(INPUT_DIR).iterdir()) if p.is_file() and p.suffix.lower() in {".wav",".flac",".aiff",".aif",".ogg",".m4a",".mp3"}]
    if not files: print("No audio files found in 'sounds/'."); return
    cols=["row_type","file","rt60_s"];  cols+=(["edt_s"] if PRINT_EDT else []);  cols+=["r2","c50_db","c80_db","mode","quality","method","snr_db","capped"];  cols+=(["band_hz"] if DO_BAND_RTS else [])
    print(", ".join(cols))
    for p in files:
        try:
            out=estimate_rt60(p)
            if out is None:
                row=["full",p.name,""];  row+=([""] if PRINT_EDT else []);  row+=["","","",MODE_LABEL,"LOW","NA","",""];  row+=([""] if DO_BAND_RTS else [])
                print(", ".join(map(str,row)));  continue
            rt60,r2,idxs,decay,sr,c50,c80,qflag,method,snr,edt_s,capped=out
            row=["full",p.name,f"{rt60:.3f}"];  row+=( [f"{(edt_s or np.nan):.3f}"] if PRINT_EDT else [] )
            row+=[f"{r2:.3f}",f"{c50:.2f}",f"{c80:.2f}",MODE_LABEL,qflag,method,f"{snr:.1f}",str(bool(capped))]
            print(", ".join(map(str,row)))

            if DO_BAND_RTS:
                bands=get_band_preset(sr)
                if APPLY_BANDLIMIT:
                    glo,ghi=BANDLIMIT_RANGE
                    bands=[(max(lo,glo),min(hi,ghi),lab) for (lo,hi,lab) in bands if hi>glo and lo<ghi]
                y0,_=load_mono_f32(p)
                for (lo,hi,lab) in bands:
                    rtb,r2b,methb,capped_b=estimate_rt60_for_band(y0,sr,lo,hi)
                    tag=format_band_tag(lo,hi,lab)
                    if (rtb is None) or (rtb<=0):
                        band_row=["band",p.name,""]; band_row+=([""] if PRINT_EDT else []); band_row+=["","","",MODE_LABEL,"LOW","NA",f"{snr:.1f}","",tag]
                    else:
                        qtag="CHECK" if (methb!="REG" or (np.isfinite(r2b) and r2b<QUALITY_MIN_R2)) else "OK"
                        band_row=["band",p.name,f"{rtb:.3f}"]; band_row+=([""] if PRINT_EDT else []); band_row+=[f"{r2b:.3f}" if np.isfinite(r2b) else "","","",MODE_LABEL,qtag,methb,f"{snr:.1f}",str(bool(capped_b)),tag]
                    print(", ".join(band_row))

            i0,i1=idxs
            if MAKE_PLOTS and i1>i0:
                t=np.arange(len(decay))/sr; slope,inter,_,_,_=linregress(t[i0:i1],decay[i0:i1]); save_plot(p.stem,decay,sr,idxs,slope,inter)
                y0s,_=load_mono_f32(p); ys=bandlimit(y0s,sr,*BANDLIMIT_RANGE) if APPLY_BANDLIMIT else y0s; ys=preprocess(ys,sr,normalize=APPLY_NORMALIZE,trim_to_peak=True)
                spec=compute_spectrogram(ys,sr); plot_spectrogram_png(p.stem, spec, fmax_hint=BANDLIMIT_RANGE[1] if APPLY_BANDLIMIT else None)

        except Exception as e:
            print(f"{p.name}, error: {e}, {MODE_LABEL}")

if __name__ == "__main__":
    main()
