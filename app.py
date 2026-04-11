"""
Scholarship Eligibility Prediction — Flask REST API (v2)
=========================================================
Breaking changes vs v1
-----------------------
• CGPA field removed from /predict and /batch_predict
• New grade percentage fields: class_8_pct, class_9_pct, class_10_pct,
  class_11_pct, class_12_pct  (each 0–100)
• Optional weights object in /predict and /batch_predict
  { "class_8": 0.10, "class_9": 0.10, "class_10": 0.25, "class_11": 0.25, "class_12": 0.30 }
  Weights are normalised automatically if they don't sum to 1.

Endpoints
---------
    POST /predict           – single student
    POST /batch_predict     – CSV file upload
    GET  /model/info        – model metadata & global feature importances
    GET  /health            – health check

Run
---
    python app.py
    gunicorn -w 4 -b 0.0.0.0:5000 app:app
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
from typing import Dict, List, Any, Optional

app = Flask(__name__)
CORS(app)

MODEL_DIR = "model_artifacts"

# ── Default weights ────────────────────────────────────────────────────────────
DEFAULT_WEIGHTS = {
    "class_8":  0.10,
    "class_9":  0.10,
    "class_10": 0.25,
    "class_11": 0.25,
    "class_12": 0.30,
}

# Default category bonus points (mirrors train_model.py)
DEFAULT_CATEGORY_BONUSES = {
    "SC":      20,
    "ST":      20,
    "OBC":     12,
    "General":  0,
    "NRI":    -10,
}


# ── Model loading ──────────────────────────────────────────────────────────────

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


# ── Weight helpers ─────────────────────────────────────────────────────────────

def normalise_weights(weights: Dict[str, float]) -> Dict[str, float]:
    """Ensure weights sum to exactly 1.0."""
    total = sum(weights.values())
    if total <= 0:
        raise ValueError("Weights must be positive and non-zero.")
    return {k: v / total for k, v in weights.items()}


def parse_weights(raw: Optional[Dict]) -> Dict[str, float]:
    """Parse and normalise user-supplied weights, falling back to defaults."""
    if not raw:
        return dict(DEFAULT_WEIGHTS)

    required_keys = {"class_8", "class_9", "class_10", "class_11", "class_12"}
    missing = required_keys - set(raw.keys())
    if missing:
        raise ValueError(f"Weight keys missing: {sorted(missing)}")

    weights = {k: float(raw[k]) for k in required_keys}
    if any(v < 0 for v in weights.values()):
        raise ValueError("Weights cannot be negative.")

    return normalise_weights(weights)


def parse_category_bonuses(raw: Optional[Dict]) -> Dict[str, float]:
    """
    Parse user-supplied category bonus points, falling back to defaults.
    Values are plain numbers (can be negative); no normalisation applied.
    """
    if not raw:
        return dict(DEFAULT_CATEGORY_BONUSES)

    valid_keys = set(DEFAULT_CATEGORY_BONUSES.keys())
    unknown = set(raw.keys()) - valid_keys
    if unknown:
        raise ValueError(f"Unknown category keys: {sorted(unknown)}. Valid: {sorted(valid_keys)}")

    # Merge with defaults so omitted categories still have a value
    bonuses = dict(DEFAULT_CATEGORY_BONUSES)
    for k, v in raw.items():
        bonuses[k] = float(v)
    return bonuses


# ── Academic Score Indicator ───────────────────────────────────────────────────

def compute_academic_score(pcts: Dict[str, float], weights: Dict[str, float]) -> float:
    """
    Weighted average of five grade percentages.
    pcts keys: class_8_pct, class_9_pct, class_10_pct, class_11_pct, class_12_pct
    weights keys: class_8, class_9, class_10, class_11, class_12
    Returns a value in [0, 100].
    """
    score = (
        pcts["class_8_pct"]  * weights["class_8"]
        + pcts["class_9_pct"]  * weights["class_9"]
        + pcts["class_10_pct"] * weights["class_10"]
        + pcts["class_11_pct"] * weights["class_11"]
        + pcts["class_12_pct"] * weights["class_12"]
    )
    return float(np.clip(score, 0, 100))


def validate_percentages(data: Dict) -> Dict[str, float]:
    """Extract and validate the five grade percentage fields."""
    pct_keys = ["class_8_pct", "class_9_pct", "class_10_pct", "class_11_pct", "class_12_pct"]
    pcts = {}
    errors = []
    for key in pct_keys:
        try:
            val = float(data[key])
        except (KeyError, TypeError, ValueError):
            errors.append(f"'{key}' must be a number")
            continue
        if not (0 <= val <= 100):
            errors.append(f"'{key}' must be between 0 and 100 (got {val})")
        pcts[key] = val

    if errors:
        raise ValueError("; ".join(errors))
    return pcts


# ── SHAP-style explanation ─────────────────────────────────────────────────────

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
    shap_items = [
        {
            "feature":      feature_names[i],
            "contribution": round(float(contributions[i]), 4),
            "direction":    "positive" if contributions[i] >= 0 else "negative",
        }
        for i in range(len(feature_names))
    ]
    shap_items.sort(key=lambda x: abs(x["contribution"]), reverse=True)

    top_positive = [x for x in shap_items if x["direction"] == "positive"][:3]
    top_negative = [x for x in shap_items if x["direction"] == "negative"][:3]

    return {
        "all_features":         shap_items,
        "top_positive_factors": top_positive,
        "top_negative_factors": top_negative,
        "dominant_feature":     shap_items[0]["feature"] if shap_items else None,
    }


def encode_input(data: Dict, pcts: Dict[str, float], academic_score: float, meta: Dict) -> np.ndarray:
    """Encode a raw input dict + computed academic score into a feature vector."""
    enc = meta["encoders"]

    gender_val   = enc["gender"].get(data.get("gender", "male"), 1)
    location_val = enc["location"].get(data.get("location", "urban"), 1)
    category_val = enc["category"].get(data.get("category", "General"), 0)
    income       = float(data.get("parents_income", 150000))
    age          = float(data.get("age", 22))

    return np.array([
        gender_val, location_val, category_val,
        pcts["class_8_pct"], pcts["class_9_pct"], pcts["class_10_pct"],
        pcts["class_11_pct"], pcts["class_12_pct"],
        academic_score,
        income, age,
    ])


def generate_narrative(eligible: bool, shap_explanation: Dict, confidence: float,
                        academic_score: float) -> str:
    top_pos = shap_explanation["top_positive_factors"]
    top_neg = shap_explanation["top_negative_factors"]

    if eligible:
        pos_str = ", ".join(x["feature"] for x in top_pos[:2]) if top_pos else "multiple factors"
        narrative = (
            f"This student is predicted ELIGIBLE with {confidence:.0f}% confidence. "
            f"Academic Score Indicator: {academic_score:.1f}/100. "
            f"Key positive factors: {pos_str}. "
        )
        if top_neg:
            neg_str = ", ".join(x["feature"] for x in top_neg[:1])
            narrative += f"Minor limiting factor: {neg_str}."
    else:
        neg_str = ", ".join(x["feature"] for x in top_neg[:2]) if top_neg else "low academic scores"
        narrative = (
            f"This student is predicted NOT ELIGIBLE with {confidence:.0f}% confidence. "
            f"Academic Score Indicator: {academic_score:.1f}/100. "
            f"Main limiting factors: {neg_str}. "
        )
        if top_pos:
            pos_str = ", ".join(x["feature"] for x in top_pos[:1])
            narrative += (
                f"Positive factor: {pos_str}. "
                f"Improving class percentages or documenting financial need may help."
            )

    return narrative


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


@app.route("/model/info", methods=["GET"])
def model_info():
    meta = get_meta()
    return jsonify({
        "accuracy":                meta["accuracy"],
        "roc_auc":                 meta.get("roc_auc"),
        "training_samples":        meta["training_samples"],
        "test_samples":            meta["test_samples"],
        "eligibility_rate":        meta["eligibility_rate"],
        "feature_names":           meta["feature_names"],
        "global_importances":      dict(zip(meta["feature_names"], meta["feature_importances"])),
        "valid_categories":        list(meta["encoders"]["category"].keys()),
        "valid_genders":           list(meta["encoders"]["gender"].keys()),
        "valid_locations":         list(meta["encoders"]["location"].keys()),
        "default_weights":         meta.get("default_weights", DEFAULT_WEIGHTS),
        "default_category_bonuses": meta.get("default_category_bonuses", DEFAULT_CATEGORY_BONUSES),
        "data_stats":              meta["stats"],
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict scholarship eligibility for a single student.

    Request body (JSON):
    {
      "gender":         "female" | "male" | "other",
      "location":       "rural" | "urban",
      "category":       "SC" | "ST" | "OBC" | "General" | "NRI",
      "class_8_pct":    0–100,
      "class_9_pct":    0–100,
      "class_10_pct":   0–100,
      "class_11_pct":   0–100,
      "class_12_pct":   0–100,
      "parents_income": 20000–300000,
      "age":            18–30,
      "weights": {                     ← optional grade weights (normalised if ≠ 1)
        "class_8": 0.10, "class_9": 0.10, "class_10": 0.25,
        "class_11": 0.25, "class_12": 0.30
      },
      "category_bonuses": {            ← optional; any subset of categories
        "SC": 20, "ST": 20, "OBC": 12, "General": 0, "NRI": -10
      }
    }

    Response:
    {
      "eligible":              true | false,
      "confidence":            0–100 (float),
      "probability":           { "eligible": float, "not_eligible": float },
      "academic_score":        float (0–100),
      "eligibility_score":     float  ← rule-based score used for the label
      "weights_used":          { ... },
      "category_bonuses_used": { ... },
      "shap":                  { ... },
      "narrative":             string,
      "input_received":        { ... }
    }
    """
    body = request.get_json(force=True)
    if not body:
        return jsonify({"error": "JSON body required"}), 400

    required = [
        "gender", "location", "category",
        "class_8_pct", "class_9_pct", "class_10_pct", "class_11_pct", "class_12_pct",
        "parents_income", "age",
    ]
    missing = [k for k in required if k not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        # Validate percentages
        pcts = validate_percentages(body)

        # Resolve grade weights
        weights = parse_weights(body.get("weights"))

        # Resolve category bonuses
        category_bonuses = parse_category_bonuses(body.get("category_bonuses"))

        model = get_model()
        meta  = get_meta()

        # Compute derived academic score
        academic_score = compute_academic_score(pcts, weights)

        # Compute rule-based eligibility score (for transparency)
        cat    = body.get("category", "General")
        income = float(body.get("parents_income", 150000))
        age    = float(body.get("age", 22))
        loc    = body.get("location", "urban")

        elig_score = (
            (academic_score / 100) * 40
            + max(0, (300000 - income) / 300000) * 30
            + category_bonuses.get(cat, 0)
            + (8 if loc == "rural" else 0)
            + max(0, (30 - age) / 12) * 5
        )

        # Build feature vector
        X = encode_input(body, pcts, academic_score, meta)

        prob       = model.predict_proba(X.reshape(1, -1))[0]
        eligible   = bool(prob[1] >= 0.5)
        confidence = float(prob[1] * 100) if eligible else float(prob[0] * 100)

        contribs  = compute_local_shap(model, X, len(meta["feature_names"]))
        shap_exp  = build_shap_explanation(contribs, meta["feature_names"])
        narrative = generate_narrative(eligible, shap_exp, confidence, academic_score)

        return jsonify({
            "eligible":              eligible,
            "confidence":            round(confidence, 2),
            "probability": {
                "eligible":     round(float(prob[1]), 4),
                "not_eligible": round(float(prob[0]), 4),
            },
            "academic_score":        round(academic_score, 2),
            "eligibility_score":     round(float(elig_score), 2),
            "weights_used":          weights,
            "category_bonuses_used": category_bonuses,
            "shap":                  shap_exp,
            "narrative":             narrative,
            "input_received":        body,
        })

    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """
    Batch prediction from a CSV file upload.

    CSV must have columns:
        gender, location, category,
        class_8_pct, class_9_pct, class_10_pct, class_11_pct, class_12_pct,
        parents_income, age

    Optional form fields (JSON strings):
      weights          – grade weights  e.g. {"class_8":0.1,...}
      category_bonuses – bonus points   e.g. {"SC":20,"ST":20,"OBC":12,"General":0,"NRI":-10}

    Returns a list of predictions (one per CSV row).
    """
    if "file" not in request.files:
        return jsonify({"error": "Upload a CSV file with key 'file'"}), 400

    # Parse optional weights
    raw_weights = request.form.get("weights") or request.args.get("weights")
    try:
        weights_dict = json.loads(raw_weights) if raw_weights else None
        weights = parse_weights(weights_dict)
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({"error": f"Invalid weights: {e}"}), 400

    # Parse optional category bonuses
    raw_bonuses = request.form.get("category_bonuses") or request.args.get("category_bonuses")
    try:
        bonuses_dict = json.loads(raw_bonuses) if raw_bonuses else None
        category_bonuses = parse_category_bonuses(bonuses_dict)
    except (json.JSONDecodeError, ValueError) as e:
        return jsonify({"error": f"Invalid category_bonuses: {e}"}), 400

    file    = request.files["file"]
    content = file.read().decode("utf-8")
    reader  = csv.DictReader(io.StringIO(content))

    model   = get_model()
    meta    = get_meta()
    results: List[Dict[str, Any]] = []

    for idx, row in enumerate(reader):
        try:
            row_lower = {k.lower().strip(): v for k, v in row.items()}

            body = {
                "gender":         row_lower.get("gender", "male"),
                "location":       row_lower.get("location", "urban"),
                "category":       row_lower.get("category", "General"),
                "class_8_pct":    float(row_lower.get("class_8_pct", 60)),
                "class_9_pct":    float(row_lower.get("class_9_pct", 60)),
                "class_10_pct":   float(row_lower.get("class_10_pct", 60)),
                "class_11_pct":   float(row_lower.get("class_11_pct", 60)),
                "class_12_pct":   float(row_lower.get("class_12_pct", 60)),
                "parents_income": float(row_lower.get("parents_income", 150000)),
                "age":            float(row_lower.get("age", 22)),
            }

            pcts           = validate_percentages(body)
            academic_score = compute_academic_score(pcts, weights)
            X              = encode_input(body, pcts, academic_score, meta)
            prob           = model.predict_proba(X.reshape(1, -1))[0]

            # Rule-based score for transparency
            cat    = body["category"]
            income = body["parents_income"]
            age    = body["age"]
            loc    = body["location"]
            elig_score = (
                (academic_score / 100) * 40
                + max(0, (300000 - income) / 300000) * 30
                + category_bonuses.get(cat, 0)
                + (8 if loc == "rural" else 0)
                + max(0, (30 - age) / 12) * 5
            )

            results.append({
                "row":              idx + 1,
                "eligible":         bool(prob[1] >= 0.5),
                "confidence":       round(float(max(prob)) * 100, 2),
                "academic_score":   round(academic_score, 2),
                "eligibility_score": round(float(elig_score), 2),
                "input":            body,
            })
        except Exception as e:
            results.append({"row": idx + 1, "error": str(e)})

    eligible_count = sum(1 for r in results if r.get("eligible"))
    return jsonify({
        "total":                 len(results),
        "eligible_count":        eligible_count,
        "ineligible_count":      len(results) - eligible_count,
        "weights_used":          weights,
        "category_bonuses_used": category_bonuses,
        "predictions":           results,
    })


if __name__ == "__main__":
    import os
    print("Starting Scholarship Eligibility API (v2) …")
    print("Endpoints:")
    print("  GET  /health")
    print("  GET  /model/info")
    print("  POST /predict")
    print("  POST /batch_predict  (multipart/form-data, field='file')")
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=False, host="0.0.0.0", port=5000)
