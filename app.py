"""
Scholarship Eligibility Prediction - Flask REST API
=====================================================
Endpoints:
    POST /predict          - Predict eligibility + SHAP explanations
    GET  /model/info       - Model metadata & global feature importances
    POST /batch_predict    - Predict for multiple students (CSV upload)
    GET  /health           - Health check

Academic Score Indicator:
    Replaces the old CGPA field. Clients must send either:
      - 'academic_score': float (pre-computed weighted score, 0–100)
    OR the five individual class percentages with optional custom weightages:
      - 'class_8_pct', 'class_9_pct', 'class_10_pct', 'class_11_pct', 'class_12_pct'
      - 'weights': { "class_8": 0.05, "class_9": 0.10, "class_10": 0.15,
                     "class_11": 0.20, "class_12": 0.50 }  (optional, must sum to 1)

Run:
    python app.py
    # or with gunicorn:
    gunicorn -w 4 -b 0.0.0.0:5000 app:app

Requirements:
    pip install flask flask-cors scikit-learn joblib numpy pandas
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import json
import os
import io
import csv
from functools import lru_cache
from typing import Dict, List, Any

app = Flask(__name__)
CORS(app)

MODEL_DIR = "model_artifacts"

# Default per-class weightages (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "class_8":  0.05,
    "class_9":  0.10,
    "class_10": 0.15,
    "class_11": 0.20,
    "class_12": 0.50,
}


# ─── Model loading ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_model():
    path = os.path.join(MODEL_DIR, "random_forest.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Model not found at {path}. Run train_model.py first."
        )
    return joblib.load(path)


@lru_cache(maxsize=1)
def get_meta() -> Dict:
    path = os.path.join(MODEL_DIR, "model_meta.json")
    with open(path) as f:
        return json.load(f)


# ─── Academic Score Indicator ───────────────────────────────────────────────────

def validate_weights(weights: Dict) -> (bool, str):
    """Validate that custom weightages are provided and sum to 1.0."""
    required = ["class_8", "class_9", "class_10", "class_11", "class_12"]
    for key in required:
        if key not in weights:
            return False, f"Missing weight key: {key}"
        if not (0.0 <= float(weights[key]) <= 1.0):
            return False, f"Weight for {key} must be between 0 and 1"
    total = sum(float(weights[k]) for k in required)
    if abs(total - 1.0) > 0.01:
        return False, f"Weights must sum to 1.0, got {total:.4f}"
    return True, ""


def compute_academic_score(data: Dict) -> float:
    """
    Compute the Academic Score Indicator from class percentages and weightages.
    If 'academic_score' is directly provided, use it as-is.
    Otherwise compute from individual class percentages.
    """
    # Direct score provided (e.g., from batch CSV)
    if "academic_score" in data:
        return float(data["academic_score"])

    # Resolve weights
    weights = data.get("weights", DEFAULT_WEIGHTS)
    if isinstance(weights, str):
        try:
            weights = json.loads(weights)
        except Exception:
            weights = DEFAULT_WEIGHTS

    w8  = float(weights.get("class_8",  DEFAULT_WEIGHTS["class_8"]))
    w9  = float(weights.get("class_9",  DEFAULT_WEIGHTS["class_9"]))
    w10 = float(weights.get("class_10", DEFAULT_WEIGHTS["class_10"]))
    w11 = float(weights.get("class_11", DEFAULT_WEIGHTS["class_11"]))
    w12 = float(weights.get("class_12", DEFAULT_WEIGHTS["class_12"]))

    p8  = float(data.get("class_8_pct",  0))
    p9  = float(data.get("class_9_pct",  0))
    p10 = float(data.get("class_10_pct", 0))
    p11 = float(data.get("class_11_pct", 0))
    p12 = float(data.get("class_12_pct", 0))

    score = p8 * w8 + p9 * w9 + p10 * w10 + p11 * w11 + p12 * w12
    return round(score, 4)


# ─── SHAP-style explanation ────────────────────────────────────────────────────

def compute_local_shap(model, X_sample: np.ndarray, n_features: int) -> np.ndarray:
    """Decision-path contribution analysis across all trees in the forest."""
    contributions = np.zeros(n_features)

    for tree in model.estimators_:
        tree_ = tree.tree_
        node_indicator = tree.decision_path(X_sample.reshape(1, -1))
        node_ids = node_indicator.indices

        for i in range(len(node_ids) - 1):
            node_id = node_ids[i]
            feature = tree_.feature[node_id]
            if feature < 0:
                continue

            def class_prob(nid):
                vals = tree_.value[nid][0]
                return vals[1] / vals.sum()

            parent_prob = class_prob(node_id)
            left_child  = tree_.children_left[node_id]
            right_child = tree_.children_right[node_id]

            if X_sample[feature] <= tree_.threshold[node_id]:
                child_prob = class_prob(left_child)
            else:
                child_prob = class_prob(right_child)

            contributions[feature] += child_prob - parent_prob

    return contributions / len(model.estimators_)


def build_shap_explanation(contributions: np.ndarray, feature_names: List[str]) -> Dict:
    """Structure SHAP contributions into a ranked explanation dict."""
    shap_items = [
        {
            "feature": feature_names[i],
            "contribution": round(float(contributions[i]), 4),
            "direction": "positive" if contributions[i] >= 0 else "negative",
        }
        for i in range(len(feature_names))
    ]
    shap_items.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    top_positive = [x for x in shap_items if x["direction"] == "positive"][:3]
    top_negative = [x for x in shap_items if x["direction"] == "negative"][:3]

    return {
        "all_features": shap_items,
        "top_positive_factors": top_positive,
        "top_negative_factors": top_negative,
        "dominant_feature": shap_items[0]["feature"] if shap_items else None,
    }


def encode_input(data: Dict, meta: Dict) -> np.ndarray:
    """Encode a raw input dict into a feature vector the model can consume."""
    enc = meta["encoders"]

    gender_val    = enc["gender"].get(data.get("gender", "male"), 1)
    location_val  = enc["location"].get(data.get("location", "urban"), 1)
    category_val  = enc["category"].get(data.get("category", "General"), 0)
    academic_score = compute_academic_score(data)
    income        = float(data.get("parents_income", 150000))
    age           = float(data.get("age", 22))

    return np.array([gender_val, location_val, category_val,
                     academic_score, income, age])


def generate_narrative(eligible: bool, shap_explanation: Dict, confidence: float) -> str:
    """Generate a human-readable explanation string from SHAP results."""
    top_pos = shap_explanation["top_positive_factors"]
    top_neg = shap_explanation["top_negative_factors"]

    if eligible:
        pos_str = ", ".join(x["feature"] for x in top_pos[:2]) if top_pos else "multiple factors"
        narrative = (
            f"This student is predicted ELIGIBLE with {confidence:.0f}% confidence. "
            f"Key positive factors: {pos_str}. "
        )
        if top_neg:
            neg_str = ", ".join(x["feature"] for x in top_neg[:1])
            narrative += f"Minor limiting factor: {neg_str}."
    else:
        neg_str = ", ".join(x["feature"] for x in top_neg[:2]) if top_neg else "insufficient scores"
        narrative = (
            f"This student is predicted NOT ELIGIBLE with {confidence:.0f}% confidence. "
            f"Main limiting factors: {neg_str}. "
        )
        if top_pos:
            pos_str = ", ".join(x["feature"] for x in top_pos[:1])
            narrative += f"Positive factor: {pos_str}. Improving Academic Score or income documentation may help."

    return narrative


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/model/info", methods=["GET"])
def model_info():
    meta = get_meta()
    return jsonify({
        "accuracy":           meta["accuracy"],
        "roc_auc":            meta.get("roc_auc"),
        "training_samples":   meta["training_samples"],
        "test_samples":       meta["test_samples"],
        "eligibility_rate":   meta["eligibility_rate"],
        "feature_names":      meta["feature_names"],
        "global_importances": dict(zip(meta["feature_names"], meta["feature_importances"])),
        "valid_categories":   list(meta["encoders"]["category"].keys()),
        "valid_genders":      list(meta["encoders"]["gender"].keys()),
        "valid_locations":    list(meta["encoders"]["location"].keys()),
        "default_weights":    meta.get("default_weights", DEFAULT_WEIGHTS),
        "data_stats":         meta["stats"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict scholarship eligibility for a single student.

    Request body (JSON) — option A (pre-computed score):
        {
          "gender":         "female" | "male" | "other",
          "location":       "rural" | "urban",
          "category":       "SC" | "ST" | "OBC" | "General" | "NRI",
          "academic_score": 0–100,
          "parents_income": 20000–300000,
          "age":            18–30
        }

    Request body (JSON) — option B (raw percentages):
        {
          "gender":         "female",
          "location":       "rural",
          "category":       "SC",
          "class_8_pct":    85,
          "class_9_pct":    88,
          "class_10_pct":   90,
          "class_11_pct":   87,
          "class_12_pct":   92,
          "weights": {
            "class_8": 0.05, "class_9": 0.10, "class_10": 0.15,
            "class_11": 0.20, "class_12": 0.50
          },
          "parents_income": 100000,
          "age":            20
        }

    Response:
        {
          "eligible":           true | false,
          "confidence":         0–100 (float),
          "probability":        { "eligible": float, "not_eligible": float },
          "academic_score":     float,
          "shap":               { ... },
          "narrative":          string,
          "input_received":     { ... }
        }
    """
    body = request.get_json(force=True)
    if not body:
        return jsonify({"error": "JSON body required"}), 400

    # Accept either pre-computed academic_score OR individual class percentages
    has_score = "academic_score" in body
    has_classes = all(
        k in body for k in
        ["class_8_pct", "class_9_pct", "class_10_pct", "class_11_pct", "class_12_pct"]
    )
    if not has_score and not has_classes:
        return jsonify({
            "error": (
                "Provide either 'academic_score' OR all five class percentages "
                "('class_8_pct', 'class_9_pct', 'class_10_pct', 'class_11_pct', 'class_12_pct')"
            )
        }), 400

    required_others = ["gender", "location", "category", "parents_income", "age"]
    missing = [k for k in required_others if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    # Validate custom weights if provided
    if "weights" in body:
        valid, msg = validate_weights(body["weights"])
        if not valid:
            return jsonify({"error": f"Invalid weights: {msg}"}), 400

    try:
        model          = get_model()
        meta           = get_meta()
        X              = encode_input(body, meta)
        academic_score = compute_academic_score(body)

        prob       = model.predict_proba(X.reshape(1, -1))[0]
        eligible   = bool(prob[1] >= 0.5)
        confidence = float(prob[1] * 100) if eligible else float(prob[0] * 100)

        contribs   = compute_local_shap(model, X, len(meta["feature_names"]))
        shap_exp   = build_shap_explanation(contribs, meta["feature_names"])
        narrative  = generate_narrative(eligible, shap_exp, confidence)

        return jsonify({
            "eligible":       eligible,
            "confidence":     round(confidence, 2),
            "probability":    {
                "eligible":     round(float(prob[1]), 4),
                "not_eligible": round(float(prob[0]), 4),
            },
            "academic_score": round(academic_score, 2),
            "shap":           shap_exp,
            "narrative":      narrative,
            "input_received": body,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Batch prediction from a CSV file upload.

    CSV columns (required):
        gender, location, category, parents_income, age

    Academic score columns — use ONE of:
      Option A:  academic_score  (pre-computed weighted score 0–100)
      Option B:  class_8_pct, class_9_pct, class_10_pct, class_11_pct, class_12_pct
                 (default weights applied: 5%, 10%, 15%, 20%, 50%)

    Returns a list of predictions (one per row).
    """
    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV file with key 'file'"}), 400

    file    = request.files["file"]
    content = file.read().decode("utf-8")
    reader  = csv.DictReader(io.StringIO(content))

    model   = get_model()
    meta    = get_meta()
    results: List[Dict[str, Any]] = []

    for idx, row in enumerate(reader):
        try:
            row_lower = {k.lower().strip(): v.strip() for k, v in row.items()}

            # Build body dict — support both academic_score and class columns
            body: Dict[str, Any] = {
                "gender":         row_lower.get("gender", "male"),
                "location":       row_lower.get("location", "urban"),
                "category":       row_lower.get("category", "General"),
                "parents_income": float(row_lower.get("parents_income", 150000)),
                "age":            float(row_lower.get("age", 22)),
            }

            if "academic_score" in row_lower and row_lower["academic_score"]:
                body["academic_score"] = float(row_lower["academic_score"])
            elif all(k in row_lower for k in
                     ["class_8_pct", "class_9_pct", "class_10_pct",
                      "class_11_pct", "class_12_pct"]):
                body["class_8_pct"]  = float(row_lower["class_8_pct"])
                body["class_9_pct"]  = float(row_lower["class_9_pct"])
                body["class_10_pct"] = float(row_lower["class_10_pct"])
                body["class_11_pct"] = float(row_lower["class_11_pct"])
                body["class_12_pct"] = float(row_lower["class_12_pct"])
            elif "cgpa" in row_lower and row_lower["cgpa"]:
                # Backward-compat: convert CGPA → academic_score
                body["academic_score"] = float(row_lower["cgpa"]) * 10.0
            else:
                raise ValueError(
                    "Row must contain 'academic_score', five class_X_pct columns, or 'cgpa'"
                )

            X              = encode_input(body, meta)
            academic_score = compute_academic_score(body)
            prob           = model.predict_proba(X.reshape(1, -1))[0]

            results.append({
                "row":           idx + 1,
                "eligible":      bool(prob[1] >= 0.5),
                "confidence":    round(float(max(prob)) * 100, 2),
                "academic_score": round(academic_score, 2),
                "input":         body,
            })
        except Exception as e:
            results.append({"row": idx + 1, "error": str(e)})

    eligible_count = sum(1 for r in results if r.get("eligible"))
    return jsonify({
        "total":            len(results),
        "eligible_count":   eligible_count,
        "ineligible_count": len(results) - eligible_count,
        "predictions":      results,
    })


if __name__ == "__main__":
    import os
    print("Starting Scholarship Eligibility API …")
    print("Endpoints:")
    print("  GET  /health")
    print("  GET  /model/info")
    print("  POST /predict         (JSON: class percentages + weights, or academic_score)")
    print("  POST /batch_predict   (CSV: class_X_pct columns, academic_score, or cgpa)")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
