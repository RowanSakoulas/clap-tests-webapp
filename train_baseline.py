#!/usr/bin/env python3
# python train_baseline.py --data data --out models
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt, joblib
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import mean_absolute_error, f1_score
FEATURES_BASE=['edt_s','c50_db','c80_db','r2','snr_db','spl_leq_db','spl_l90_db','capped']
BAND_FEATURES=['rt60_band_mean','rt60_band_median','n_band_ok','n_band_check','n_band_low']
def safe_read_df(csv,parq):
    p=Path(parq)
    if p.exists(): return pd.read_parquet(p)
    return pd.read_csv(csv)
def compute_band_aggregates(df):
    if df is None or df.empty: return pd.DataFrame(columns=['file']+BAND_FEATURES)
    dfv=df.copy()
    dfv['is_ok']=(dfv['quality'].astype(str).str.upper()=='OK').astype(int)
    dfv['is_check']=(dfv['quality'].astype(str).str.upper()=='CHECK').astype(int)
    dfv['is_low']=(dfv['quality'].astype(str).str.upper()=='LOW').astype(int)
    dfv['rt60_valid']=pd.to_numeric(dfv['rt60_s'],errors='coerce')
    dfv.loc[~np.isfinite(dfv['rt60_valid']),'rt60_valid']=np.nan
    agg=dfv.groupby('file').agg(rt60_band_mean=('rt60_valid','mean'),
                                rt60_band_median=('rt60_valid','median'),
                                n_band_ok=('is_ok','sum'),
                                n_band_check=('is_check','sum'),
                                n_band_low=('is_low','sum')).reset_index()
    return agg
def filter_training_rows(df):
    df2=df.copy(); 
    if not np.issubdtype(df2['capped'].dtype, np.number): df2['capped']=df2['capped'].astype(int, errors='ignore')
    y=pd.to_numeric(df2['rt60_s'],errors='coerce')
    mask=(y>=0.1)&(y<=10.0)
    return df2.loc[mask].reset_index(drop=True)
def train_reg(df,features,out_dir):
    X=df[features].astype(float).fillna(df[features].median(numeric_only=True)); y=df['rt60_s'].astype(float)
    model=RandomForestRegressor(n_estimators=300,random_state=42)
    n=len(df); n_splits=min(5,n) if n>=3 else 2; cv=KFold(n_splits=n_splits,shuffle=True,random_state=42)
    maes=[]
    for tr,te in cv.split(X):
        model.fit(X.iloc[tr],y.iloc[tr]); p=model.predict(X.iloc[te]); maes.append(mean_absolute_error(y.iloc[te],p))
    mae=float(np.mean(maes)); model.fit(X,y); Path(out_dir).mkdir(parents=True,exist_ok=True); joblib.dump(model, Path(out_dir)/'regressor.joblib')
    try:
        imp=model.feature_importances_; order=np.argsort(imp)[::-1]; plt.figure(figsize=(6,4)); plt.bar(range(len(features)),np.array(imp)[order])
        plt.xticks(range(len(features)),np.array(features)[order],rotation=45,ha='right'); plt.title('Regression feature importance'); plt.tight_layout()
        plt.savefig(Path(out_dir)/'feature_importance_reg.png',dpi=160); plt.close()
    except: pass
    return mae, model
def train_cls(df,features,out_dir):
    X=df[features].astype(float).fillna(df[features].median(numeric_only=True)); y=df['quality'].astype(str)
    model=RandomForestClassifier(n_estimators=400,class_weight='balanced',random_state=42)
    n=len(df); class_min=y.value_counts().min() if len(y)>0 else 0
    n_splits=int(max(2, min(5, n, class_min)))
    if n_splits<2:
        model.fit(X,y); Path(out_dir).mkdir(parents=True,exist_ok=True); joblib.dump(model, Path(out_dir)/'classifier.joblib'); return float('nan'), model
    cv=StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=42)
    f1s=[]
    for tr,te in cv.split(X,y):
        model.fit(X.iloc[tr],y.iloc[tr]); p=model.predict(X.iloc[te]); f1s.append(f1_score(y.iloc[te],p,average='macro'))
    f1=float(np.mean(f1s)); model.fit(X,y); Path(out_dir).mkdir(parents=True,exist_ok=True); joblib.dump(model, Path(out_dir)/'classifier.joblib')
    try:
        imp=model.feature_importances_; order=np.argsort(imp)[::-1]; plt.figure(figsize=(6,4)); plt.bar(range(len(features)),np.array(imp)[order])
        plt.xticks(range(len(features)),np.array(features)[order],rotation=45,ha='right'); plt.title('Classification feature importance'); plt.tight_layout()
        plt.savefig(Path(out_dir)/'feature_importance_cls.png',dpi=160); plt.close()
    except: pass
    return f1, model
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',type=str,default='data'); ap.add_argument('--out',type=str,default='models'); args=ap.parse_args()
    data_dir=Path(args.data); out_dir=Path(args.out); out_dir.mkdir(parents=True,exist_ok=True)
    df_full=safe_read_df(data_dir/'dataset.csv', data_dir/'dataset.parquet')
    df_band=None
    if (data_dir/'rt60_bands.csv').exists() or (data_dir/'rt60_bands.parquet').exists():
        df_band=safe_read_df(data_dir/'rt60_bands.csv', data_dir/'rt60_bands.parquet')
    df_agg=compute_band_aggregates(df_band) if df_band is not None else pd.DataFrame(columns=['file']+BAND_FEATURES)
    df=df_full.merge(df_agg,on='file',how='left'); df_train=filter_training_rows(df)
    features=FEATURES_BASE+[c for c in BAND_FEATURES if c in df_train.columns]
    mae,_=train_reg(df_train,features,out_dir); f1,_=train_cls(df_train,features,out_dir)
    pd.DataFrame([{'n_rows_full':len(df_full),'n_rows_train':len(df_train),'mae_rt60_s':mae,'f1_macro_quality':f1,'features_used':'|'.join(features)}]).to_csv(out_dir/'metrics.csv',index=False)
    (out_dir/'report.md').write_text("\n".join([
        '# Baseline ML Report','',f'- Rows (full): **{len(df_full)}**; rows (trainable): **{len(df_train)}**',
        f'- Regression (rt60_s) MAE (CV): **{mae:.3f} s**',f'- Classification (quality) macro-F1 (CV): **{(f1 if f1==f1 else float("nan")):.3f}**','',
        '## Features','`'+', '.join(features)+'`','','## Files','- regressor.joblib / classifier.joblib','- metrics.csv','- feature_importance_reg.png / feature_importance_cls.png']))
    print('Saved models and report to', out_dir)
if __name__=='__main__': main()
