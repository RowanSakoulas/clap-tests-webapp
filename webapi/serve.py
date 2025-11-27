#!/usr/bin/env python3
from __future__ import annotations
import math, tempfile
import pandas as pd
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import warnings
warnings.filterwarnings("ignore", category=SyntaxWarning)
from datetime import datetime, timezone
from contextlib import asynccontextmanager
import os

# Resolve paths (serve.py is inside /webapi)
THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[1]
WEBAPP_DIR = PROJECT_ROOT / "webapp"
WEBAPI_DIR = Path(__file__).parent.resolve()
INDEX_HTML = WEBAPP_DIR / "index.html"
MODEL_ROOT = PROJECT_ROOT / "models"
GRAPHS_DIR = PROJECT_ROOT / "graphs"  # serve saved plots

# Make root importable so we can import predict_baseline.py at project root
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

ALLOWED_MODELS = {
    "phone_mic": "phone_mic",      # default
    "external_mic": "external_mic"
}

CURRENT_MODEL_KEY = "phone_mic"
REG = CLS = None
FEATURES: list[str] = []
READY_MSG = "API starting..."

def load_models_for(model_key: str) -> None:
    global REG, CLS, FEATURES, READY_MSG, CURRENT_MODEL_KEY

    if model_key not in ALLOWED_MODELS:
        raise RuntimeError(f"Unsupported model key: {model_key}")

    folder_name = ALLOWED_MODELS[model_key]
    model_dir = MODEL_ROOT / folder_name
    if not model_dir.exists():
        raise RuntimeError(f"Model directory not found: {model_dir}")

    REG_, CLS_, FEATURES_ = PB.load_models(model_dir)
    REG, CLS, FEATURES = REG_, CLS_, FEATURES_
    CURRENT_MODEL_KEY = model_key
    READY_MSG = f"API ready (models\\{folder_name})"

def _strip_bad_numbers(obj):
    """Recursively replace NaN / inf floats with None so JSON is happy."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: _strip_bad_numbers(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_strip_bad_numbers(v) for v in obj]
    return obj

def _sanitize_json(payload: dict) -> dict:
    return _strip_bad_numbers(payload)

import predict_baseline as PB
import feedback as FB
from webapi import ai_llm as AILLM

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Runs once at startup
    try:
        print("[API] started:", STARTED_AT)
        print("[API] routes:", sorted({r.path for r in app.routes}))
    except Exception as e:
        print("[API] startup info error:", repr(e))
    yield

app = FastAPI(title="Clap Tests API", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],)
STARTED_AT = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

# Mount static frontend and graphs folder
app.mount("/webapp", StaticFiles(directory=str(WEBAPP_DIR), html=False), name="webapp")
if GRAPHS_DIR.exists():
    app.mount("/graphs", StaticFiles(directory=str(GRAPHS_DIR), html=False), name="graphs")

# Load models once
try:
    load_models_for("phone_mic")
except Exception as e:
    REG = CLS = None
    FEATURES = []
    READY_MSG = f"Model load failed: {e}"

@app.get("/")
def root():
    return FileResponse(str(INDEX_HTML))

@app.get("/ping")
def ping():
    return JSONResponse({"status": "ok" if REG and CLS else "error", "message": READY_MSG})

@app.get("/health")
def health():
    ok = bool(REG and CLS)
    return JSONResponse({
        "status": "ok" if ok else "error",
        "message": READY_MSG,
        "models_loaded": ok,
        "feature_count": len(FEATURES or []),
        "current_model": CURRENT_MODEL_KEY,
        "started_at": STARTED_AT,
        "routes": sorted({r.path for r in app.routes}),
    })

@app.post("/model/{key}")
def set_model(key: str):
    """
    Switch to a different model profile (e.g. phone_mic / external_mic).
    """
    try:
        load_models_for(key)
        ok = bool(REG and CLS)
        return {
            "status": "ok" if ok else "error",
            "message": READY_MSG,
            "model": CURRENT_MODEL_KEY,
            "feature_count": len(FEATURES or []),
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model '{key}': {e}")

@app.get("/analyze")
def analyze_get():
    return JSONResponse({"detail": "Use POST /analyze"})

@app.post("/analyze")
async def analyze(
    files: List[UploadFile] = File(...),
    context_use: str = Form(""),
    context_goal: str = Form(""),
    context_label: str = Form(""),
    preset: str = Form(""),
    per_band: str = Form("0"),
    band_limit: str = Form("0"),
    save_plots: str = Form("0"),
):
    if REG is None or CLS is None:
        return JSONResponse({"results": [], "error": "models not loaded"}, status_code=500)

    results = []

    for uf in files:
        suffix = Path(uf.filename).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            raw = await uf.read()
            tmp.write(raw)
            tmp_path = Path(tmp.name)

        try:
            # --- features & model prediction ---
            f = PB.compute_features_for_file(
                tmp_path,
                save_plots=bool(int(save_plots or "0")),
                graph_stem=Path(uf.filename).stem,
                preset=preset,
                band_limit=bool(int(band_limit or "0")),
            )
            row = {k: f.get(k, float("nan")) for k in FEATURES}
            X = pd.DataFrame([row], columns=FEATURES).astype(float)
            if not FEATURES:
                return JSONResponse({"results": [], "error": "model features not available"}, status_code=500)

            rt60_pred = float(REG.predict(X)[0])
            try:
                q_pred = str(CLS.predict(X)[0])
                q_probs = {}
                if hasattr(CLS, "predict_proba"):
                    p = CLS.predict_proba(X)[0]
                    if hasattr(CLS, "classes_"):
                        q_probs = {str(c): float(pp) for c, pp in zip(CLS.classes_, p)}
            except Exception:
                q_pred, q_probs = "CHECK", {}

            rt60_meas = f.get("rt60_s", float("nan"))
            r2 = f.get("r2", float("nan"))
            snr = f.get("snr_db", float("nan"))
            capped = bool(f.get("capped", False))
            rt60_fused = PB.blend_rt60(rt60_meas, r2, snr, capped, rt60_pred)

            # --- structured, per-take feedback (deterministic) ---
            direct_fb = FB.direct_feedback(
                f,
                {"rt60_pred": float(rt60_pred), "label": q_pred},
                {"use": context_use, "goal": context_goal, "label": context_label},
            )

            # --- result payload for this file ---
            results.append({
                "name": uf.filename,
                "display": uf.filename,
                "rt60_measured": None if not math.isfinite(rt60_meas) else float(rt60_meas),
                "rt60_predicted": float(rt60_pred),
                "rt60_fused": None if not math.isfinite(rt60_fused) else float(rt60_fused),
                "edt_s": f.get("edt_s"),
                "c50_db": f.get("c50_db"),
                "c80_db": f.get("c80_db"),
                "qc": {"r2": f.get("r2"), "snr_db": f.get("snr_db"), "capped": bool(f.get("capped", False))},
                "spl_l90_db": f.get("spl_l90_db"),
                "quality_pred": q_pred,
                "quality_probs": q_probs,
                "rt60_low_med": f.get("rt60_low_med"),
                "rt60_mid_med": f.get("rt60_mid_med"),
                "rt60_high_med": f.get("rt60_high_med"),
                "rt60_bands": f.get("rt60_bands", []),
                "direct_feedback": direct_fb,
                "context": {"use": context_use, "goal": context_goal, "label": context_label},
                "preset": preset,
                "per_band": bool(int(per_band or "0")),
                "band_limit": bool(int(band_limit or "0")),
                "save_plots": bool(int(save_plots or "0")),
                "graphs": f.get("graphs", {}),
            })

        except Exception as e:
            results.append({"name": uf.filename, "error": str(e)})

        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    display_name = uf.filename
    graph_stem = Path(display_name).stem

    counts = {"OK": 0, "CHECK": 0, "LOW": 0}
    for r in results:
        q = str(r.get("quality_pred", "")).upper()
        if q in counts:
            counts[q] += 1

    ml_info = {}
    # Average class probabilities across takes (if available)
    if any("quality_probs" in r for r in results):
        prob_acc = {}
        prob_n = 0
        for r in results:
            probs = r.get("quality_probs") or {}
            if probs:
                for k,v in probs.items():
                    prob_acc[k] = prob_acc.get(k, 0.0) + float(v)
                prob_n += 1
        if prob_n:
            ml_info["avg_quality_probs"] = {k: v/prob_n for k, v in prob_acc.items()}

    print(f"[/analyze] files={len(files)} ctx_use='{context_use}' ctx_goal='{context_goal}' ctx_label='{context_label}' "
          f"preset='{preset}' per_band='{per_band}' band_limit='{band_limit}' save_plots='{save_plots}'")

    # Top feature importances from classifier, if present
    try:
        import numpy as np
        if hasattr(CLS, "feature_importances_") and FEATURES:
            imps = list(map(float, getattr(CLS, "feature_importances_")))
            pairs = sorted(zip(FEATURES, imps), key=lambda x: x[1], reverse=True)[:5]
            ml_info["top_features"] = [{"name": n, "weight": w} for n,w in pairs if w > 0]
    except Exception:
        pass

    # --- AI room summary ---
    ctx = {"use": context_use, "goal": context_goal, "session": context_label}

    try:
        ai_room = AILLM.generate_summary(results, ctx, ml_info)
        # keep only JSON-safe fields
        keep = {}
        for k in ("bullets", "actions", "error", "error_msg"):
            v = ai_room.get(k)
            if k in ("bullets", "actions"):
                keep[k] = v if isinstance(v, list) else []
            elif v is not None:
                keep[k] = str(v)
        ai_room = keep
        if ai_room.get("error"):
            print("[AI]", ai_room["error"], "-", ai_room.get("error_msg",""))
    except Exception as e:
        print("[AI] call failed in /analyze:", repr(e))
        ai_room = {"bullets": [], "actions": [], "error": "server_exception",
                "error_msg": "AI summary unavailable: server exception"}

    legacy_ai = ""
    try:
        if ai_room.get("bullets"):
            legacy_ai = " ".join(str(b) for b in ai_room["bullets"] if b)
    except Exception:
        legacy_ai = ""

    payload = {
        "results": results,
        "summary": counts,
        "room_summary": ai_room,
        "ai": legacy_ai,
        "ai_summary": legacy_ai,
    }
    return _sanitize_json(payload)

