# Baseline ML Report

- Rows (full): **136**; rows (trainable): **136**
- Regression (rt60_s) MAE (CV): **0.406 s**
- Classification (quality) macro-F1 (CV): **0.978**

## Features
`edt_s, c50_db, c80_db, r2, snr_db, spl_leq_db, spl_l90_db, capped, rt60_band_mean, rt60_band_median, n_band_ok, n_band_check, n_band_low`

## Files
- regressor.joblib / classifier.joblib
- metrics.csv
- feature_importance_reg.png / feature_importance_cls.png