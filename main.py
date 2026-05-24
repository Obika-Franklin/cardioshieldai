"""
CardioShield AI — Python ML Inference Service
Serves real Random Forest + SMOTE and VGG16 ECG predictions.
Runs on port 8001 (internal, called by the Node.js API server).
"""

import os
import sys
import logging
import base64
import io
from pathlib import Path
from contextlib import asynccontextmanager

import numpy as np
import joblib
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cardioshield-ml")

# ── Paths ────────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
# Instead of searching deep in the monorepo root, create a local models directory
MODELS_DIR = SCRIPT_DIR / "models"

RF_MODEL_PATH    = MODELS_DIR / "rf_model.pkl"
PREPROCESSOR_PATH= MODELS_DIR / "preprocessor.pkl"
VGG16_PATH       = MODELS_DIR / "vgg16_ecg_model.keras"
VGG16_URL        = "https://github.com/Obika-Franklin/cardioshield-ai/releases/download/v1.0.0/vgg16_ecg_model.keras"
RF_MODEL_URL     = "https://github.com/Obika-Franklin/cardioshield-ai/releases/download/rf_model/rf_model.pkl"
PREPROCESSOR_URL = "https://github.com/Obika-Franklin/cardioshield-ai/releases/download/preprocessor/preprocessor.pkl"

# ── Lazy-loaded models ────────────────────────────────────────────────────────
_rf_model    = None
_preprocessor = None
_vgg16_model = None

# ── ECG class mapping ─────────────────────────────────────────────────────────
# Alphabetical order from folder names used during ImageDataGenerator training
# (confirmed from notebook classification report output):
#   0: "ECG Images of Myocardial Infarction Patients ..."  → Myocardial Infarction
#   1: "ECG Images of Patient that have History of MI ..."  → History of MI
#   2: "ECG Images of Patient that have abnormal heartbeat ..." → Abnormal Heartbeat
#   3: "Normal Person ECG Images ..."                        → Normal ECG
ECG_CLASS_MAP = {
    0: {"key": "myocardial_infarction", "label": "Myocardial Infarction",             "riskLevel": "high"},
    1: {"key": "history_mi",            "label": "History of Myocardial Infarction",   "riskLevel": "high"},
    2: {"key": "abnormal_heartbeat",    "label": "Abnormal Heartbeat",                "riskLevel": "moderate"},
    3: {"key": "normal",                "label": "Normal ECG",                        "riskLevel": "low"},
}

ECG_FINDINGS = {
    "normal":               "Regular P waves, normal PR interval (160–200ms), QRS within normal limits (<120ms). No ischemic changes. ST segments isoelectric. No arrhythmia detected.",
    "myocardial_infarction":"Pathological Q waves, ST elevation, and T-wave inversion consistent with myocardial infarction. Urgent cardiology review required. Do not delay reperfusion evaluation.",
    "history_mi":           "Residual Q waves and chronic ST-T changes consistent with prior myocardial infarction. Secondary prevention and cardiologist follow-up strongly recommended.",
    "abnormal_heartbeat":   "Abnormal rhythm morphology detected. Features inconsistent with normal sinus rhythm — may indicate arrhythmia, conduction abnormality, or ectopic beats. Clinical correlation required.",
}

# ── Model loading ─────────────────────────────────────────────────────────────

def download_model(url: str, dest: Path, label: str) -> bool:
    """Download a model file from a URL if not already present."""
    if dest.exists():
        logger.info(f"{label} found at {dest}")
        return True
    logger.info(f"Downloading {label} from {url} ...")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    try:
        resp = requests.get(url, stream=True, timeout=300)
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        downloaded = 0
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total and total > 1_000_000:
                        pct = downloaded / total * 100
                        logger.info(f"  {label}: {pct:.1f}% ({downloaded // 1_000_000}MB / {total // 1_000_000}MB)")
        logger.info(f"{label} download complete ({downloaded // 1024}KB).")
        return True
    except Exception as e:
        logger.error(f"Failed to download {label}: {e}")
        if dest.exists():
            dest.unlink()
        return False

def get_rf_models():
    global _rf_model, _preprocessor
    if _rf_model is None:
        if not RF_MODEL_PATH.exists():
            raise FileNotFoundError(f"RF model not found at {RF_MODEL_PATH}.")
        if not PREPROCESSOR_PATH.exists():
            raise FileNotFoundError(f"Preprocessor not found at {PREPROCESSOR_PATH}.")
        logger.info("Loading RF model and preprocessor via joblib...")
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            _rf_model    = joblib.load(RF_MODEL_PATH)
            _preprocessor = joblib.load(PREPROCESSOR_PATH)
        logger.info(f"RF model loaded: {type(_rf_model).__name__}, "
                    f"n_estimators={_rf_model.n_estimators}, "
                    f"n_features_in={_rf_model.n_features_in_}")
    return _rf_model, _preprocessor

def get_vgg16():
    global _vgg16_model
    if _vgg16_model is None:
        if not VGG16_PATH.exists():
            raise FileNotFoundError(f"VGG16 model not found at {VGG16_PATH}.")
        logger.info("Loading VGG16 model (this may take a moment)...")
        import tensorflow as tf
        _vgg16_model = tf.keras.models.load_model(str(VGG16_PATH))
        logger.info(f"VGG16 loaded. Input shape: {_vgg16_model.input_shape}")
    return _vgg16_model

# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("=== CardioShield ML Service starting ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download RF model and preprocessor if missing
    download_model(RF_MODEL_URL,     RF_MODEL_PATH,     "RF model (rf_model.pkl)")
    download_model(PREPROCESSOR_URL, PREPROCESSOR_PATH, "Preprocessor (preprocessor.pkl)")

    # Pre-load RF models
    if RF_MODEL_PATH.exists() and PREPROCESSOR_PATH.exists():
        try:
            get_rf_models()
        except Exception as e:
            logger.warning(f"Could not pre-load RF models: {e}")
    else:
        logger.warning(f"RF models still missing in {MODELS_DIR}.")

    # Download VGG16 if needed, then pre-load
    vgg16_ok = download_model(VGG16_URL, VGG16_PATH, "VGG16 model")
    if vgg16_ok:
        try:
            get_vgg16()
        except Exception as e:
            logger.warning(f"Could not pre-load VGG16: {e}")

    logger.info("=== CardioShield ML Service ready ===")
    yield
    logger.info("ML Service shutting down.")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="CardioShield ML Service", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Schemas ───────────────────────────────────────────────────────────────────

class PatientFeatures(BaseModel):
    age: float
    sex: float
    chestPainType: float
    restingBpS: float
    cholesterol: float
    fastingBloodSugar: float
    restingEcg: float
    maxHeartRate: float
    exerciseAngina: float
    oldpeak: float
    stSlope: float

class EcgImageRequest(BaseModel):
    imageData: str  # base64 encoded (may include data URI prefix)

# ── Helpers ───────────────────────────────────────────────────────────────────

def build_feature_dataframe(data: PatientFeatures):
    """Build a pandas DataFrame with the exact column names the preprocessor was fitted on."""
    import pandas as pd
    # Column names must match preprocessor.feature_names_in_ exactly:
    # ['age', 'sex', 'chest pain type', 'resting bp s', 'cholesterol',
    #  'fasting blood sugar', 'resting ecg', 'max heart rate',
    #  'exercise angina', 'oldpeak', 'ST slope']
    return pd.DataFrame([{
        "age":                data.age,
        "sex":                data.sex,
        "chest pain type":    data.chestPainType,
        "resting bp s":       data.restingBpS,
        "cholesterol":        data.cholesterol,
        "fasting blood sugar":data.fastingBloodSugar,
        "resting ecg":        data.restingEcg,
        "max heart rate":     data.maxHeartRate,
        "exercise angina":    data.exerciseAngina,
        "oldpeak":            data.oldpeak,
        "ST slope":           data.stSlope,
    }])


def validate_ecg_image(img: Image.Image) -> dict:
    """
    Structural pre-screening to detect likely non-ECG images before running
    the CNN. Uses three lightweight signal checks:

    1. Grayscale dominance — ECG strips are near-monochrome (mostly black/white
       with a coloured grid at most). If the image has high colour saturation it
       is almost certainly not an ECG scan.
    2. High-frequency horizontal content — ECG waveforms create strong
       horizontal frequency components. Natural photos have more isotropic
       frequency content.
    3. Dark-pixel density — ECG traces typically have a high proportion of very
       dark pixels (the trace on a light background).

    Returns a dict with keys:
        is_likely_ecg  : bool
        reason         : str
        confidence     : "HIGH" | "MEDIUM" | "LOW"
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)

    # ── Check 1: Colour saturation ────────────────────────────────────────────
    # Convert to HSV-like saturation. Real ECG scans are mostly grey-scale.
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    cmax = np.maximum.reduce([r, g, b])
    cmin = np.minimum.reduce([r, g, b])
    chroma = cmax - cmin  # 0 = grey, 255 = full colour
    mean_chroma = float(np.mean(chroma))
    high_saturation = mean_chroma > 55  # typical ECG paper: ~10–30

    # ── Check 2: Horizontal frequency dominance ───────────────────────────────
    grey = np.array(img.convert("L"), dtype=np.float32)
    # Row-wise variance (captures horizontal striping from waveforms)
    row_var = float(np.mean(np.var(grey, axis=1)))
    # Col-wise variance (captures vertical striping, less relevant for ECG)
    col_var = float(np.mean(np.var(grey, axis=0)))
    # ECGs: row_var >> col_var (waveforms run horizontally)
    freq_ratio = row_var / (col_var + 1e-6)
    poor_freq = freq_ratio < 0.5  # natural images tend to be more isotropic

    # ── Check 3: Dark-pixel density ───────────────────────────────────────────
    dark_ratio = float(np.mean(grey < 80))  # fraction of clearly dark pixels
    no_trace = dark_ratio < 0.02  # essentially no dark content = no ECG trace

    # ── Combine signals ───────────────────────────────────────────────────────
    fail_count = sum([high_saturation, poor_freq, no_trace])

    if fail_count == 0:
        return {"is_likely_ecg": True,  "reason": "Image passes ECG structural checks", "confidence": "HIGH"}
    elif fail_count == 1:
        reasons = []
        if high_saturation: reasons.append(f"high colour saturation ({mean_chroma:.0f})")
        if poor_freq:        reasons.append("low horizontal frequency ratio")
        if no_trace:         reasons.append("insufficient dark pixel content")
        return {"is_likely_ecg": True,  "reason": f"Marginal ECG likelihood — {reasons[0]}", "confidence": "MEDIUM"}
    else:
        reasons = []
        if high_saturation: reasons.append(f"high colour saturation ({mean_chroma:.0f})")
        if poor_freq:        reasons.append("low horizontal frequency content")
        if no_trace:         reasons.append("no ECG trace detected")
        return {"is_likely_ecg": False, "reason": f"Not a valid ECG image: {', '.join(reasons)}", "confidence": "HIGH"}


def decode_and_preprocess_image(image_data: str, target_size: tuple) -> tuple[np.ndarray, Image.Image]:
    """Decode base64 image, apply ECG-specific preprocessing, and resize to target_size.

    ECG-Specific preprocessing pipeline:
      1. Convert to greyscale — ECG strips are monochrome; this removes
         irrelevant colour information that would confuse the model.
      2. CLAHE contrast enhancement — sharpens faint waveform details
         (works on the greyscale version).
      3. Threshold binarisation — suppresses grid-line noise and isolates
         the ECG trace against a clean white background.
      4. Convert back to RGB — the VGG16 expects 3 channels.

    NOTE: Training used ImageDataGenerator() with NO rescale parameter, so the
    model was trained on raw pixel values in [0, 255]. We match that here —
    no division by 255, no VGG16 preprocess_input.
    """
    # Strip data URI prefix if present (e.g. "data:image/png;base64,...")
    if "," in image_data:
        image_data = image_data.split(",", 1)[1]
    img_bytes = base64.b64decode(image_data)
    original_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")

    # Step 1: Greyscale
    grey = np.array(original_img.convert("L"), dtype=np.float32)

    # Step 2: CLAHE-style contrast enhancement via histogram normalisation
    # (avoids the OpenCV dependency while achieving a similar result)
    p2, p98 = np.percentile(grey, 2), np.percentile(grey, 98)
    if p98 > p2:
        grey = np.clip((grey - p2) / (p98 - p2) * 255.0, 0, 255)

    # Step 3: Adaptive-style binarisation — pixels below the local mean
    # become dark (the waveform), everything else becomes white (background).
    # This removes grid noise while preserving the ECG trace.
    local_mean = float(np.mean(grey))
    # Soft threshold: push background toward white, trace toward black
    alpha = 1.8  # contrast stretch factor
    grey = np.clip(alpha * (grey - local_mean) + local_mean, 0, 255)

    # Step 4: Back to 3-channel RGB (replicate greyscale into R, G, B)
    grey_uint8 = grey.astype(np.uint8)
    processed_img = Image.fromarray(grey_uint8, mode="L").convert("RGB")

    # Resize to model input size
    processed_img = processed_img.resize((target_size[1], target_size[0]))  # PIL uses (W, H)
    arr = np.array(processed_img, dtype=np.float32)  # raw [0, 255] — matches training
    return np.expand_dims(arr, axis=0), original_img  # (1, H, W, 3), original PIL image


def check_prediction_confidence(preds: np.ndarray) -> dict:
    """
    Entropy-based out-of-distribution detection.

    A model trained only on ECG images will still produce a softmax output for
    any input — but for non-ECG images the distribution tends to be more
    uniform (high entropy) because no class is a good fit.

    Real ECG images typically yield a clear dominant class (low entropy).

    Returns:
        is_confident : bool   — True if distribution is peaked (ECG-like)
        entropy      : float  — Shannon entropy of the output distribution
        reason       : str
    """
    probs = preds.astype(np.float64)
    probs = np.clip(probs, 1e-9, 1.0)
    probs = probs / probs.sum()
    entropy = float(-np.sum(probs * np.log(probs)))
    max_prob = float(np.max(preds))

    # Thresholds calibrated against a 4-class uniform distribution (max entropy ≈ 1.386)
    # ECG images typically: entropy < 0.8, max_prob > 0.55
    if entropy > 1.1 or max_prob < 0.40:
        return {
            "is_confident": False,
            "entropy": entropy,
            "max_prob": max_prob,
            "reason": f"Model confidence too low for reliable ECG classification (entropy={entropy:.2f}, max_prob={max_prob:.0%}). Image may not be a valid ECG.",
        }
    elif entropy > 0.8 or max_prob < 0.55:
        return {
            "is_confident": True,
            "entropy": entropy,
            "max_prob": max_prob,
            "reason": f"Moderate CNN confidence (entropy={entropy:.2f}, max_prob={max_prob:.0%}). Treat result with caution.",
        }
    else:
        return {
            "is_confident": True,
            "entropy": entropy,
            "max_prob": max_prob,
            "reason": f"Good CNN confidence (entropy={entropy:.2f}, max_prob={max_prob:.0%}).",
        }


def risk_from_probability(prob: float) -> str:
    if prob < 0.35:
        return "low"
    elif prob < 0.65:
        return "moderate"
    else:
        return "high"

# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "rf_loaded": _rf_model is not None,
        "vgg16_loaded": _vgg16_model is not None,
        "rf_model_exists": RF_MODEL_PATH.exists(),
        "preprocessor_exists": PREPROCESSOR_PATH.exists(),
        "vgg16_exists": VGG16_PATH.exists(),
    }


@app.post("/predict/rf")
def predict_rf(data: PatientFeatures):
    try:
        rf_model, preprocessor = get_rf_models()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"RF model load error: {e}")
        raise HTTPException(status_code=500, detail=f"RF model load error: {e}")

    try:
        features_df = build_feature_dataframe(data)
        features_transformed = preprocessor.transform(features_df)

        proba = rf_model.predict_proba(features_transformed)[0]
        # Class 1 = heart disease present
        risk_prob = float(proba[1]) if len(proba) > 1 else float(proba[0])
        risk_score = round(risk_prob * 100, 2)
        risk_level = risk_from_probability(risk_prob)

        # Aggregate 22 transformed feature importances back to 11 original features.
        # Mapping: prefix of get_feature_names_out() → original name
        raw_imp = rf_model.feature_importances_
        try:
            tf_names = preprocessor.get_feature_names_out()
            # Group by original feature (strip pipeline prefix and OHE suffix)
            grouped: dict[str, float] = {}
            prefix_map = {
                "num_pipeline__age": "Age",
                "num_pipeline__resting bp s": "Resting BP",
                "num_pipeline__cholesterol": "Cholesterol",
                "num_pipeline__max heart rate": "Max Heart Rate",
                "num_pipeline__oldpeak": "Oldpeak (ST Depression)",
                "cat_pipeline__sex": "Sex",
                "cat_pipeline__chest pain type": "Chest Pain Type",
                "cat_pipeline__fasting blood sugar": "Fasting Blood Sugar",
                "cat_pipeline__resting ecg": "Resting ECG",
                "cat_pipeline__exercise angina": "Exercise Angina",
                "cat_pipeline__ST slope": "ST Slope",
            }
            for i, tf_name in enumerate(tf_names):
                matched = None
                for prefix, label in prefix_map.items():
                    if tf_name == prefix or tf_name.startswith(prefix + "_"):
                        matched = label
                        break
                if matched:
                    grouped[matched] = grouped.get(matched, 0.0) + float(raw_imp[i])
            features_out = sorted(
                [{"name": k, "importance": round(v, 4)} for k, v in grouped.items()],
                key=lambda x: x["importance"], reverse=True
            )
        except Exception:
            # Fallback: use raw importances with generic names
            features_out = sorted(
                [{"name": f"Feature {i}", "importance": round(float(v), 4)} for i, v in enumerate(raw_imp)],
                key=lambda x: x["importance"], reverse=True
            )

        if risk_level == "low":
            recommendation = "Low cardiovascular risk. Maintain healthy lifestyle with regular exercise and balanced diet. Routine annual check-ups recommended."
        elif risk_level == "moderate":
            recommendation = "Moderate cardiovascular risk. Follow-up with primary care physician within 4–6 weeks. Consider lifestyle modifications and monitoring of blood pressure and cholesterol."
        else:
            recommendation = "High cardiovascular risk detected. Urgent clinical review recommended within 1–2 weeks. Referral to cardiologist advised. Immediate assessment of ST changes, blood pressure, and lipid panel required."

        return {
            "riskScore": risk_score,
            "riskLevel": risk_level,
            "rfProbability": risk_prob,
            "recommendation": recommendation,
            "features": features_out,
            "modelAccuracy": 0.9202,
            "modelPrecision": 0.921,
            "modelRecall": 0.919,
            "modelF1": 0.920,
        }
    except Exception as e:
        logger.error(f"RF prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"RF prediction error: {e}")


@app.post("/predict/ecg")
def predict_ecg(data: EcgImageRequest):
    try:
        model = get_vgg16()
    except FileNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except Exception as e:
        logger.error(f"VGG16 load error: {e}")
        raise HTTPException(status_code=500, detail=f"VGG16 load error: {e}")

    try:
        # Determine input size from model
        input_shape = model.input_shape  # (None, H, W, C)
        h, w = input_shape[1], input_shape[2]
        if h is None or w is None:
            h, w = 224, 224  # Default VGG16 input

        # ── Layer 1: Structural ECG validation ────────────────────────────────
        # Strip data URI prefix to get the raw base64 for PIL
        raw_b64 = data.imageData
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",", 1)[1]
        import io as _io
        original_img = Image.open(_io.BytesIO(base64.b64decode(raw_b64))).convert("RGB")
        structural_check = validate_ecg_image(original_img)
        logger.info(f"ECG structural check: {structural_check}")

        if not structural_check["is_likely_ecg"]:
            # Reject outright — don't waste CNN inference on an obviously non-ECG image
            return {
                "classification": "Rejected — Not an ECG Image",
                "confidence": 0.0,
                "findings": f"Image rejected before CNN classification. {structural_check['reason']}. Please upload a clear ECG strip or 12-lead ECG image.",
                "riskLevel": "low",
                "probabilities": {},
                "isDemo": False,
                "modelAccuracy": 0.7483,
                "modelPrecision": 0.778,
                "modelRecall": 0.7483,
                "modelF1": 0.751,
                "modelName": "VGG16 ECG Classifier",
                "validationWarning": structural_check["reason"],
                "rejected": True,
            }

        # ── Layer 2: ECG-specific preprocessing then CNN inference ────────────
        img_arr, _ = decode_and_preprocess_image(data.imageData, (h, w))
        preds = model.predict(img_arr, verbose=0)[0]  # shape: (num_classes,)

        # ── Layer 3: Entropy-based confidence check ───────────────────────────
        confidence_check = check_prediction_confidence(preds)
        logger.info(f"ECG confidence check: {confidence_check}")

        predicted_idx = int(np.argmax(preds))
        confidence = float(preds[predicted_idx])

        # Map index → class info
        class_info = ECG_CLASS_MAP.get(predicted_idx, {
            "key": "unknown",
            "label": "Unknown",
            "riskLevel": "moderate",
        })
        class_key = class_info["key"]

        all_probs = {
            ECG_CLASS_MAP[i]["key"]: float(preds[i])
            for i in range(len(preds)) if i in ECG_CLASS_MAP
        }

        # Build findings: use standard findings, but append any confidence warnings
        findings = ECG_FINDINGS.get(class_key, "ECG waveform analyzed. Clinical correlation recommended.")
        validation_warning = None

        if not confidence_check["is_confident"]:
            # Model is uncertain — flag the result prominently
            class_info = {
                "key": class_key,
                "label": f"{class_info['label']} (Low Confidence — verify manually)",
                "riskLevel": "moderate",
            }
            findings = f"LOW CONFIDENCE RESULT — {confidence_check['reason']} {findings}"
            validation_warning = confidence_check["reason"]
        elif structural_check["confidence"] == "MEDIUM":
            validation_warning = structural_check["reason"]

        return {
            "classification": class_info["label"],
            "confidence": confidence,
            "findings": findings,
            "riskLevel": class_info["riskLevel"],
            "probabilities": all_probs,
            "isDemo": False,
            "modelAccuracy": 0.7483,
            "modelPrecision": 0.778,
            "modelRecall": 0.7483,
            "modelF1": 0.751,
            "modelName": "VGG16 ECG Classifier",
            **({"validationWarning": validation_warning} if validation_warning else {}),
            "rejected": False,
        }
    except Exception as e:
        logger.error(f"ECG prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"ECG prediction error: {e}")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8008)) 
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
