from dotenv import load_dotenv
load_dotenv()

import os
import io
import joblib
import numpy as np
import pandas as pd
import time
from functools import lru_cache

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Any, Optional

from lime.lime_tabular import LimeTabularExplainer
import google.generativeai as genai


# ===================== PATHS ===================== #

MODEL_PATH = "network_attack_model.joblib"
SCALER_PATH = "network_attack_scaler.joblib"
PCA_PATH = "network_attack_pca.joblib"
ENCODERS_PATH = "network_attack_label_encoders.joblib"
XTRAIN_PATH = "X_train.npy"


# ===================== FASTAPI APP ===================== #

app = FastAPI(title="Network Attack Detection API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ===================== GLOBALS ===================== #

clf = None
scaler = None
pca = None
label_encoders = None
X_train = None
explainer: Optional[LimeTabularExplainer] = None

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Caching for mitigation requests to avoid quota issues
mitigation_cache = {}
last_api_call = 0
MIN_REQUEST_INTERVAL = 5  # Minimum 5 seconds between API calls


# ===================== Pydantic MODELS ===================== #

class ManualInputRequest(BaseModel):
    values: str  # comma-separated string from frontend


class MitigationRequest(BaseModel):
    attack_type: str
    top_features: List[Dict[str, Any]]  # expects list of {feature, contribution}


class PredictionResponse(BaseModel):
    predicted_attack: str
    confidence: float
    class_probabilities: Dict[str, float]
    lime_features: Optional[List[Dict[str, Any]]] = None
    top_original_features: Optional[List[Dict[str, Any]]] = None


class BatchPredictionItem(BaseModel):
    row: int
    predicted_attack: str
    confidence: float


class BatchPredictionResponse(BaseModel):
    predictions: List[BatchPredictionItem]


class MitigationStep(BaseModel):
    feature: str
    recommended_change: str


class MitigationResponse(BaseModel):
    attack_type: str
    mitigations: Optional[List[MitigationStep]] = None
    raw_text: Optional[str] = None


# ===================== STARTUP: LOAD MODELS + LIME ===================== #

@app.on_event("startup")
def load_models():
    global clf, scaler, pca, label_encoders, X_train, explainer

    # ----- Model -----
    try:
        if os.path.exists(MODEL_PATH):
            print(f"[INFO] Loading model from {MODEL_PATH}")
            clf_loaded = joblib.load(MODEL_PATH)
            # ensure string classes
            if hasattr(clf_loaded, "classes_"):
                clf_loaded.classes_ = np.array([str(c) for c in clf_loaded.classes_])
            clf_local = clf_loaded
        else:
            print(f"[WARNING] {MODEL_PATH} not found.")
            clf_local = None
    except Exception as e:
        print(f"[WARNING] Error loading model: {e}")
        clf_local = None
    finally:
        globals()["clf"] = clf_local

    # ----- Scaler -----
    try:
        if os.path.exists(SCALER_PATH):
            print(f"[INFO] Loading scaler from {SCALER_PATH}")
            scaler_local = joblib.load(SCALER_PATH)
        else:
            print(f"[WARNING] {SCALER_PATH} not found.")
            scaler_local = None
    except Exception as e:
        print(f"[WARNING] Error loading scaler: {e}")
        scaler_local = None
    finally:
        globals()["scaler"] = scaler_local

    # ----- PCA -----
    try:
        if os.path.exists(PCA_PATH):
            print(f"[INFO] Loading PCA from {PCA_PATH}")
            pca_local = joblib.load(PCA_PATH)
        else:
            print(f"[WARNING] {PCA_PATH} not found.")
            pca_local = None
    except Exception as e:
        print(f"[WARNING] Error loading PCA: {e}")
        pca_local = None
    finally:
        globals()["pca"] = pca_local

    # ----- Label encoders (optional) -----
    try:
        if os.path.exists(ENCODERS_PATH):
            print(f"[INFO] Loading label encoders from {ENCODERS_PATH}")
            enc_local = joblib.load(ENCODERS_PATH)
        else:
            print(f"[WARNING] {ENCODERS_PATH} not found.")
            enc_local = None
    except Exception as e:
        print(f"[WARNING] Error loading encoders: {e}")
        enc_local = None
    finally:
        globals()["label_encoders"] = enc_local

    # ----- X_train for LIME -----
    try:
        if os.path.exists(XTRAIN_PATH):
            print(f"[INFO] Loading X_train from {XTRAIN_PATH}")
            X_local = np.load(XTRAIN_PATH)
        else:
            print(f"[WARNING] {XTRAIN_PATH} not found. Using random fallback.")
            n_components = pca_local.n_components_ if pca_local is not None else 10
            X_local = np.random.randn(200, n_components)
    except Exception as e:
        print(f"[WARNING] Could not load X_train.npy: {e}")
        n_components = pca_local.n_components_ if pca_local is not None else 10
        X_local = np.random.randn(200, n_components)
    finally:
        globals()["X_train"] = X_local

    # ----- LIME explainer -----
    expl = None
    if X_local is not None and pca_local is not None and clf_local is not None:
        try:
            expl = LimeTabularExplainer(
                training_data=X_local,
                feature_names=[f"PC{i+1}" for i in range(X_local.shape[1])],
                class_names=list(clf_local.classes_),
                mode="classification",
            )
            print("[INFO] LIME explainer initialized.")
        except Exception as e:
            print(f"[WARNING] Could not initialize LIME explainer: {e}")
            expl = None
    globals()["explainer"] = expl

    print("Startup complete. Models loaded status:", {
        "clf": clf is not None,
        "scaler": scaler is not None,
        "pca": pca is not None,
        "x_train": X_train is not None,
        "explainer": explainer is not None,
    })


# ===================== HELPER CONSTANTS ===================== #

MISSING_TOKENS = {"-", "na", "n/a", "null", "none", ""}

PROTO_MAP = {
    "tcp": 6.0,
    "udp": 17.0,
    "icmp": 1.0,
    # add more if you want
    # "mqtt": 1883.0,
}


# ===================== HELPER FUNCTIONS ===================== #

def parse_input_values(values_str: str) -> np.ndarray:
    """
    Parse comma-separated string into numpy array.

    Handles:
      - numeric values (including '6.00E06')
      - protocol names: tcp/udp/icmp -> numeric codes
      - missing markers: '-', 'na', 'null' -> 0.0
      - unknown strings -> 0.0
    """
    tokens = values_str.replace("\r", " ").replace("\n", " ").split(",")
    floats: List[float] = []
    unknown_tokens: List[str] = []

    for raw in tokens:
        t = raw.strip()
        if not t:
            continue

        lower = t.lower()

        # missing markers
        if lower in MISSING_TOKENS:
            floats.append(0.0)
            continue

        # known protocols
        if lower in PROTO_MAP:
            floats.append(PROTO_MAP[lower])
            continue

        # numeric
        try:
            v = float(t)
            floats.append(v)
            continue
        except Exception:
            # unknown string -> 0.0
            floats.append(0.0)
            unknown_tokens.append(t)

    if not floats:
        raise HTTPException(
            status_code=400,
            detail="No valid numeric values were found in 'values'.",
        )

    if unknown_tokens:
        print(f"[WARN] Unknown tokens mapped to 0.0: {unknown_tokens}")

    return np.array(floats).reshape(1, -1)


def value_to_float(v) -> float:
    """Convert individual cell to float using same rules as parse_input_values."""
    if pd.isna(v):
        return 0.0

    s = str(v).strip()
    if s == "":
        return 0.0

    lower = s.lower()
    if lower in MISSING_TOKENS:
        return 0.0
    if lower in PROTO_MAP:
        return PROTO_MAP[lower]

    try:
        return float(s)
    except Exception:
        return 0.0


def dataframe_to_numeric(df: pd.DataFrame) -> np.ndarray:
    """
    Convert full dataframe to numeric matrix:
      - use label_encoders when available (same as training)
      - fallback to value_to_float() for other string columns
    """
    df = df.copy()

    # 1) apply saved label_encoders if present
    if isinstance(label_encoders, dict):
        for col, le in label_encoders.items():
            if col in df.columns:
                try:
                    classes = [str(c) for c in getattr(le, "classes_", [])]
                    mapping = {c: i for i, c in enumerate(classes)}
                    df[col] = df[col].astype(str).map(lambda v: mapping.get(str(v), -1))
                except Exception as e:
                    print(f"[WARN] Could not apply label_encoder for column '{col}': {e}")

    # 2) any remaining non-numeric columns -> value_to_float
    for col in df.columns:
        if not np.issubdtype(df[col].dtype, np.number):
            df[col] = df[col].map(value_to_float)

    # 3) fill NaNs and cast
    df = df.fillna(0.0)
    try:
        X = df.values.astype(float)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Could not convert dataframe to numeric matrix: {e}",
        )

    return X


def get_lime_explanation(instance_pca: np.ndarray, predicted_class: str) -> List[Dict[str, Any]]:
    """Generate LIME explanation (on PCA features) for a single instance."""
    if explainer is None or clf is None:
        return []

    try:
        x = instance_pca.flatten()

        def predict_proba_wrapper(X):
            return clf.predict_proba(X)

        exp = explainer.explain_instance(
            x,
            predict_proba_wrapper,
            num_features=10,
            top_labels=1,
        )

        class_idx = list(clf.classes_).index(predicted_class)
        lime_list = exp.as_list(label=class_idx)

        feat_list: List[Dict[str, Any]] = []
        for feature, contribution in lime_list:
            feat_list.append({
                "feature": feature,
                "contribution": float(contribution),
            })

        feat_list.sort(key=lambda f: abs(f["contribution"]), reverse=True)
        return feat_list

    except Exception as e:
        print(f"[WARNING] LIME explanation error: {e}")
        return []


def map_pca_to_original_features(
    lime_features: List[Dict[str, Any]],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    """
    Map PCA components from LIME explanation to original features.
    Original feature names are unknown, so we use Feature_1, Feature_2, ...
    """
    if pca is None or not lime_features:
        return []

    try:
        components = pca.components_  # [n_components, n_original_features]
        original_importance: Dict[str, float] = {}

        for lf in lime_features[:top_n]:
            feat_str = lf["feature"]
            contrib = abs(lf["contribution"])

            first_token = feat_str.split()[0]  # 'PC1', 'PC2', etc.
            if not first_token.lower().startswith("pc"):
                continue

            try:
                pc_idx = int(first_token[2:]) - 1  # 'PC1' -> 0
            except Exception:
                continue

            if pc_idx < 0 or pc_idx >= components.shape[0]:
                continue

            weights = np.abs(components[pc_idx])
            for i, w in enumerate(weights):
                feat_name = f"Feature_{i+1}"
                original_importance[feat_name] = original_importance.get(feat_name, 0.0) + w * contrib

        top_feats = [
            {"feature": name, "importance": float(imp)}
            for name, imp in original_importance.items()
        ]
        top_feats.sort(key=lambda x: x["importance"], reverse=True)
        return top_feats[:top_n]

    except Exception as e:
        print(f"[WARNING] Feature mapping error: {e}")
        return []


def generate_mitigation_gemini(attack_type: str, top_features: List[Dict[str, Any]]) -> str:
    """Call Gemini to get mitigation instructions with caching and rate limiting."""
    global last_api_call
    
    if not GEMINI_API_KEY:
        return "Gemini API key not configured. Set GEMINI_API_KEY environment variable."

    # Create cache key from attack type and top features
    cache_key = f"{attack_type}_{hash(tuple((f.get('feature'), round(f.get('contribution', 0), 2)) for f in top_features[:5]))}"
    
    # Check cache first
    if cache_key in mitigation_cache:
        return mitigation_cache[cache_key]
    
    # Rate limiting - wait if necessary
    current_time = time.time()
    time_since_last_call = current_time - last_api_call
    if time_since_last_call < MIN_REQUEST_INTERVAL:
        wait_time = MIN_REQUEST_INTERVAL - time_since_last_call
        time.sleep(wait_time)
    
    try:
        model = genai.GenerativeModel("gemini-2.0-flash")

        feat_lines = []
        for f in top_features[:5]:
            name = f.get("feature", f.get("name", "Unknown"))
            contrib = f.get("contribution", f.get("importance", 0.0))
            feat_lines.append(f"- {name}: {contrib:.3f}")
        features_text = "\n".join(feat_lines)

        prompt = (
            f"A machine learning model for network attack detection predicted '{attack_type}'.\n"
            f"Top influential features from LIME are:\n{features_text}\n\n"
            f"For each feature, recommend a specific, actionable change or control "
            f"(e.g., set threshold, increase/decrease value/rate, block, rate-limit, filter, etc.) "
            f"to mitigate this attack.\n\n"
            f"Format your response exactly as:\n"
            f"Feature: [feature_name]\n"
            f"Recommended change: [specific action]\n\n"
            f"Provide 3–5 concise recommendations."
        )

        last_api_call = time.time()
        response = model.generate_content(prompt)
        result = response.text
        
        # Cache the result for 1 hour
        mitigation_cache[cache_key] = result
        
        return result

    except Exception as e:
        error_msg = f"Error generating mitigation: {e}"
        return error_msg


def parse_gemini_response(text: str) -> List[Dict[str, str]]:
    """Parse Gemini text into structured list of feature → recommended_change."""
    if not text:
        return []

    lines = text.strip().split("\n")
    mitigations: List[Dict[str, str]] = []
    current_feature = None
    current_change = None

    for line in lines:
        line = line.strip()
        if line.lower().startswith("feature:"):
            current_feature = line.split(":", 1)[1].strip()
        elif line.lower().startswith("recommended change:"):
            current_change = line.split(":", 1)[1].strip()
            if current_feature and current_change:
                mitigations.append({
                    "feature": current_feature,
                    "recommended_change": current_change,
                })
                current_feature = None
                current_change = None

    return mitigations


# ===================== ROUTES ===================== #

@app.get("/")
def root():
    return {
        "message": "Network Attack Detection API",
        "status": "running",
        "models_loaded": clf is not None,
        "scaler_loaded": scaler is not None,
        "pca_loaded": pca is not None,
        "explainer_loaded": explainer is not None,
        "gemini_configured": GEMINI_API_KEY is not None,
    }


@app.get("/ping")
def ping():
    return {"ping": "pong"}


@app.post("/predict/manual", response_model=PredictionResponse)
async def predict_manual(request: Request, values: Optional[str] = None):
    """
    Accepts:
      - Query param:   /predict/manual?values=1,2,3,...
      - OR JSON body:  {"values": "1,2,3,..."}
    """
    if clf is None or scaler is None or pca is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    # 1) Try query param first
    values_str = values

    # 2) If missing, try JSON body
    if not values_str:
        try:
            body = await request.json()
            values_str = body.get("values")
        except Exception:
            values_str = None

    if not values_str:
        raise HTTPException(status_code=400, detail="Missing 'values' in query or JSON body")

    # 3) Parse CSV -> numpy
    X_raw = parse_input_values(str(values_str))

    # 4) Scale + PCA
    try:
        X_scaled = scaler.transform(X_raw)
        X_pca = pca.transform(X_scaled)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing failed: {e}")

    # 5) Predict
    try:
        pred = clf.predict(X_pca)[0]
        probs = clf.predict_proba(X_pca)[0]
        confidence = float(np.max(probs))
        class_probs = {str(c): float(p) for c, p in zip(clf.classes_, probs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")

    # 6) LIME + original feature mapping
    lime_feats = get_lime_explanation(X_pca, str(pred))
    top_original = map_pca_to_original_features(lime_feats, top_n=5)

    return PredictionResponse(
        predicted_attack=str(pred),
        confidence=confidence,
        class_probabilities=class_probs,
        lime_features=lime_feats,
        top_original_features=top_original,
    )


@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(file: UploadFile = File(...)):
    """Batch prediction from uploaded CSV/Excel (handles categorical + numeric)."""
    if clf is None or scaler is None or pca is None:
        raise HTTPException(status_code=503, detail="Model artifacts not loaded")

    try:
        contents = await file.read()

        # Load into DataFrame
        if file.filename.endswith(".csv"):
            df = pd.read_csv(io.StringIO(contents.decode("utf-8")))
        elif file.filename.endswith((".xlsx", ".xls")):
            df = pd.read_excel(io.BytesIO(contents))
        else:
            raise HTTPException(
                status_code=400,
                detail="Unsupported file format, use CSV or Excel",
            )

        if df.shape[1] == 0:
            raise HTTPException(status_code=400, detail="Uploaded file has no columns")

        # Convert to numeric matrix (handles categoricals)
        X = dataframe_to_numeric(df)

        # Optional feature-count check
        expected = getattr(scaler, "n_features_in_", None)
        if expected is not None and X.shape[1] != expected:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Feature count mismatch: model expects {expected} features, "
                    f"but file has {X.shape[1]}. Make sure you upload the same "
                    f"columns used during training, in the same order."
                ),
            )

        # Scale + PCA
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)

        # Predict
        preds = clf.predict(X_pca)
        probs = clf.predict_proba(X_pca)

        results: List[BatchPredictionItem] = []
        for i, (pred, prob_vec) in enumerate(zip(preds, probs)):
            conf = float(np.max(prob_vec))
            results.append(
                BatchPredictionItem(
                    row=i,
                    predicted_attack=str(pred),
                    confidence=conf,
                )
            )

        return BatchPredictionResponse(predictions=results)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {e}")


@app.post("/mitigation", response_model=MitigationResponse)
async def get_mitigation(request_data: MitigationRequest):
    """Generate mitigation recommendations for an attack using Gemini."""
    try:
        text = generate_mitigation_gemini(request_data.attack_type, request_data.top_features)
        mitigations_raw = parse_gemini_response(text)

        return MitigationResponse(
            attack_type=request_data.attack_type,
            mitigations=[MitigationStep(**m) for m in mitigations_raw] if mitigations_raw else None,
            raw_text=None if mitigations_raw else text,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Mitigation generation failed: {e}")


# ===================== MAIN (for direct run) ===================== #

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
