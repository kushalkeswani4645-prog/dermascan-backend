"""
DermaScan — Real Prediction API
=====================================================================
Wraps your friend's trained model in a proper web API that
dermascan.html can call directly, fixing three issues found in the
original files:

  1. app.py was a Streamlit app (its own full website) — this is a
     Flask API instead, which is what a separate HTML frontend needs
     to call over the network.
  2. class_indices.json (22 classes, includes "Unknown_Normal") didn't
     match the model with verified performance data (21 classes,
     "skin_disease_mobilenetv2_finetuned.keras", 36.5% test accuracy).
     This uses the correct 21-class mapping instead, taken directly
     from that model's own training notebook output.
  3. predict.py preprocessed images with MobileNetV2's preprocess_input
     (scales to -1..1), but the verified model was actually TRAINED
     with simple rescale=1./255 (scales to 0..1). This uses the
     matching 0..1 preprocessing so predictions aren't degraded by a
     silent mismatch between training and inference.

SETUP:
  1. Put these 3 files in the same folder as your trained model:
       - this file (real_predict_api.py)
       - class_indices_verified.json  (already correct, from Claude)
       - disease_data_21.json          (already correct, from Claude)
  2. Also copy skin_disease_mobilenetv2_finetuned.keras into that
     same folder (or update MODEL_PATH below to point to it).
  3. pip install flask flask-cors tensorflow pillow numpy
  4. python real_predict_api.py
     -> runs at http://localhost:5000

Then update dermascan.html's analyze button to call
http://localhost:5000/api/predict (see the note at the bottom of
this file for the exact one-line change).
=====================================================================
"""

import os
import json
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image

APP_DIR = os.path.dirname(os.path.abspath(__file__))

# ---- Adjust this if your model file has a different name/location ----
MODEL_PATH = os.path.join(APP_DIR, "skin_disease_mobilenetv2_finetuned.keras")
CLASS_INDICES_PATH = os.path.join(APP_DIR, "class_indices_verified.json")
DISEASE_DATA_PATH = os.path.join(APP_DIR, "disease_data_21.json")
IMG_SIZE = (224, 224)

app = Flask(__name__)
CORS(app)

with open(DISEASE_DATA_PATH) as f:
    DISEASE_DATA = json.load(f)

with open(CLASS_INDICES_PATH) as f:
    class_indices = json.load(f)  # {"Acne": 0, "Actinic_Keratosis": 1, ...}
idx_to_class = {v: k for k, v in class_indices.items()}
CLASS_NAMES = [idx_to_class[i] for i in range(len(idx_to_class))]

MODEL = None
MODEL_LOAD_ERROR = None

try:
    if os.path.exists(MODEL_PATH):
        from tensorflow.keras.models import load_model
        MODEL = load_model(MODEL_PATH)
        print(f"Loaded model from {MODEL_PATH}")
        print(f"Classes ({len(CLASS_NAMES)}):", CLASS_NAMES)
    else:
        MODEL_LOAD_ERROR = f"Model file not found at {MODEL_PATH}"
        print(f"WARNING: {MODEL_LOAD_ERROR}")
        print("Server will run in DEMO MODE (mock predictions) until this is fixed.")
except Exception as e:
    MODEL_LOAD_ERROR = str(e)
    print(f"WARNING: Could not load model ({e})")
    print("Server will run in DEMO MODE (mock predictions) until this is fixed.")


def preprocess_image(pil_image):
    """
    IMPORTANT: this uses simple 0..1 rescaling to match how the
    verified model was actually trained (ImageDataGenerator(rescale=1./255)
    in the training notebook) — NOT MobileNetV2's preprocess_input,
    which scales differently and would mismatch this specific model.
    """
    img = pil_image.convert("RGB").resize(IMG_SIZE)
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0)


def run_demo_prediction():
    weights = np.random.dirichlet(np.ones(len(CLASS_NAMES)) * 0.6)
    return {name: float(w) for name, w in zip(CLASS_NAMES, weights)}


def run_real_prediction(pil_image):
    batch = preprocess_image(pil_image)
    preds = MODEL.predict(batch, verbose=0)[0]
    return {name: float(p) for name, p in zip(CLASS_NAMES, preds)}


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "mode": "demo" if MODEL is None else "trained-model",
        "model_load_error": MODEL_LOAD_ERROR,
        "classes": CLASS_NAMES,
        "num_classes": len(CLASS_NAMES),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded. Send as form field 'image'."}), 400

    file = request.files["image"]
    try:
        pil_image = Image.open(file.stream)
    except Exception:
        return jsonify({"error": "Could not read uploaded file as an image."}), 400

    probs = run_demo_prediction() if MODEL is None else run_real_prediction(pil_image)

    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)
    top_class, top_conf = ranked[0]
    info = DISEASE_DATA.get(top_class, {})

    response = {
        "mode": "demo" if MODEL is None else "trained-model",
        "prediction": {
            "code": top_class,
            "confidence": round(top_conf * 100, 2),
            **info,
        },
        "all_probabilities": [
            {"code": c, "name": DISEASE_DATA.get(c, {}).get("full_name", c),
             "confidence": round(p * 100, 2)}
            for c, p in ranked
        ],
        "disclaimer": (
            "This tool provides an informational, AI-generated estimate only. "
            "It is not a medical diagnosis. Please consult a licensed "
            "dermatologist for any skin concern."
        ),
    }
    return jsonify(response)


if __name__ == "__main__":
    if MODEL is None:
        print("=" * 70)
        print("Running in DEMO MODE — real model not loaded.")
        print("Check MODEL_PATH at the top of this file points to your")
        print("actual .keras file, and that it's in the same folder.")
        print("=" * 70)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)


# =====================================================================
# CONNECTING THIS TO dermascan.html
# =====================================================================
# In dermascan.html, find the analyzeBtn click handler (search for
# "BACKEND HOOK" near the mock prediction). Replace the mock result
# block with:
#
#   const formData = new FormData();
#   formData.append('image', selectedFile);
#   const res = await fetch('http://localhost:5000/api/predict', {
#     method: 'POST', body: formData
#   });
#   const data = await res.json();
#   sessionStorage.setItem('dermascan_last_result', JSON.stringify({
#     conditionName: data.prediction.full_name,
#     category: data.prediction.category,
#     severity: data.prediction.severity,
#     confidence: data.prediction.confidence,
#     description: data.prediction.description,
#     symptoms: data.prediction.common_symptoms,
#     treatment: data.prediction.general_treatment_info,
#     recommendedAction: data.prediction.recommended_action,
#     allProbabilities: data.all_probabilities,
#   }));
#   saveScanToHistory(...);
#   goTo('result');
#
# Send this file back to Claude and it'll make this exact edit for you
# directly in dermascan.html, and test it end-to-end.
# =====================================================================
