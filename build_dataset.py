#!/usr/bin/env python3
# python build_dataset.py --sounds sounds --out data
from __future__ import annotations
import argparse, sys, math
from pathlib import Path
from typing import List, Optional
import numpy as np, pandas as pd
THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))
import estimate_rt60 as rt
import estimate_spl as spl
AUDIO_EXTS={".wav",".flac",".aiff",".aif",".ogg",".m4a",".mp3",".aifc"}
def list_audio_files(sounds:Path)->List[Path]:
    return [p for p in sorted(Path(sounds).iterdir()) if p.is_file() and p.suffix.lower() in AUDIO_EXTS]
def safe_float(x):
    try:
        v=float(x)
        if not math.isfinite(v): return np.nan
        return v
    except: return np.nan
def quality_from_band(meth:str, r2:Optional[float])->str:
    if meth!='REG': return 'CHECK'
    try: return 'OK' if float(r2)>=rt.QUALITY_MIN_R2 else 'CHECK'
    except: return 'CHECK'
def process_file(path:Path, include_bands:bool, spl_cal_delta:float):
    try: out=rt.estimate_rt60(path)
    except Exception as e: return {'error':f'rt60 failed: {e}'}, [], None
    if out is None:
        full=dict(file=path.name, rt60_s=np.nan, edt_s=np.nan, r2=np.nan, c50_db=np.nan, c80_db=np.nan, snr_db=np.nan, quality='LOW', method='NA', mode=rt.MODE_LABEL, capped=False)
        sr_hint=None; bands=[]
    else:
        rt60,r2,idxs,decay,sr,c50,c80,q,method,snr,edt_s,capped=out
        full=dict(file=path.name, rt60_s=safe_float(rt60), edt_s=safe_float(edt_s), r2=safe_float(r2), c50_db=safe_float(c50), c80_db=safe_float(c80), snr_db=safe_float(snr), quality=str(q), method=str(method), mode=str(rt.MODE_LABEL), capped=bool(capped))
        sr_hint=int(sr)
    try:
        y,sr_spl=spl.load_mono_f32(path)
        from scipy.ndimage import uniform_filter1d
        y=uniform_filter1d(y, size=int(0.01*sr_spl)+1)
        leq,lmax,l10,l90,_,_=spl.summarize_spl(y,sr_spl)
        full.update(spl_leq_db=safe_float(leq+spl_cal_delta), spl_lmax_db=safe_float((lmax if math.isfinite(lmax) else np.nan)+spl_cal_delta), spl_l10_db=safe_float((l10 if math.isfinite(l10) else np.nan)+spl_cal_delta), spl_l90_db=safe_float((l90 if math.isfinite(l90) else np.nan)+spl_cal_delta))
    except Exception:
        full.update(spl_leq_db=np.nan,spl_lmax_db=np.nan,spl_l10_db=np.nan,spl_l90_db=np.nan)
    bands_out=[]
    if include_bands:
        try:
            if sr_hint is None: y0,sr_hint=rt.load_mono_f32(path)
            else: y0,_=rt.load_mono_f32(path)
            preset=rt.get_band_preset(int(sr_hint))
            if getattr(rt,'APPLY_BANDLIMIT',False):
                glo,ghi=getattr(rt,'BANDLIMIT_RANGE',(20.0,20000.0)); clipped=[]
                for (lo,hi,lab) in preset:
                    lo2,hi2=max(lo,glo),min(hi,ghi)
                    if hi2>lo2: clipped.append((lo2,hi2,lab))
                preset=clipped
            for (lo,hi,lab) in preset:
                try:
                    rtb,r2b,methb,capped_b=rt.estimate_rt60_for_band(y0,int(sr_hint),float(lo),float(hi))
                    tag=rt.format_band_tag(float(lo),float(hi),lab)
                    if (rtb is None) or (not math.isfinite(rtb)) or (rtb<=0):
                        bands_out.append(dict(file=path.name, band_lo_hz=float(lo), band_hi_hz=float(hi), band_label=(lab or ''), band_tag=str(tag), rt60_s=np.nan, r2=np.nan, method='NA', quality='LOW', capped=False, mode=str(rt.MODE_LABEL)))
                    else:
                        qtag=quality_from_band(methb,r2b)
                        bands_out.append(dict(file=path.name, band_lo_hz=float(lo), band_hi_hz=float(hi), band_label=(lab or ''), band_tag=str(tag), rt60_s=safe_float(rtb), r2=safe_float(r2b), method=str(methb), quality=qtag, capped=bool(capped_b), mode=str(rt.MODE_LABEL)))
                except Exception:
                    bands_out.append(dict(file=path.name, band_lo_hz=float(lo), band_hi_hz=float(hi), band_label=(lab or ''), band_tag=str(rt.format_band_tag(lo,hi,lab)), rt60_s=np.nan, r2=np.nan, method='NA', quality='LOW', capped=False, mode=str(rt.MODE_LABEL)))
        except Exception:
            pass
    return full, bands_out, sr_hint
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--sounds',type=str,default='sounds'); ap.add_argument('--out',type=str,default='data'); ap.add_argument('--no-bands',action='store_true'); ap.add_argument('--spl-cal-delta',type=float,default=0.0); args=ap.parse_args()
    sounds=Path(args.sounds); out=Path(args.out); out.mkdir(parents=True,exist_ok=True)
    files=list_audio_files(sounds)
    if not files:
        print('No audio files found in', sounds)
        return
    rows_full=[]; rows_band=[]; errors=0
    for p in files:
        full,bands,_=process_file(p, include_bands=(not args.no_bands), spl_cal_delta=float(args.spl_cal_delta))
        if isinstance(full,dict) and 'error' in full: errors+=1
        else: rows_full.append(full); rows_band.extend(bands)
    import pandas as pd
    df_full=pd.DataFrame(rows_full, columns=['file','rt60_s','edt_s','r2','c50_db','c80_db','snr_db','quality','method','mode','capped','spl_leq_db','spl_lmax_db','spl_l10_db','spl_l90_db']); df_full.to_csv(out/'dataset.csv',index=False)
    try: df_full.to_parquet(out/'dataset.parquet',index=False)
    except: pass
    if (not args.no_bands) and rows_band:
        df_band=pd.DataFrame(rows_band, columns=['file','band_lo_hz','band_hi_hz','band_label','band_tag','rt60_s','r2','method','quality','capped','mode']); df_band.to_csv(out/'rt60_bands.csv',index=False)
        try: df_band.to_parquet(out/'rt60_bands.parquet',index=False)
        except: pass
    print(f'Processed {len(files)} files; wrote {len(rows_full)} full rows'+(f', {len(rows_band)} band rows' if not args.no_bands else '')+(f'; {errors} errors' if errors else ''))
    print('Outputs in:', out)
if __name__=='__main__': main()
