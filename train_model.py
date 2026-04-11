import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import json
import os

DATA_PATH = "Scholarshipelegibility.xlsx"
MODEL_DIR = "model_artifacts"
os.makedirs(MODEL_DIR, exist_ok=True)

# Default weights for each grade's percentage contribution to Academic Score Indicator
DEFAULT_WEIGHTS = {
    "class_8":  0.10,   # 8th grade percentage
    "class_9":  0.10,   # 9th grade percentage
    "class_10": 0.25,   # 10th grade percentage (board exam — higher weight)
    "class_11": 0.25,   # 11th grade percentage
    "class_12": 0.30,   # 12th grade percentage (most recent — highest weight)
}

# Default bonus points added to the eligibility score for each reservation category.
# Range is intentionally open-ended: positive = advantage, negative = disadvantage.
# The overall eligibility threshold is 45 points (out of ~103 max).
DEFAULT_CATEGORY_BONUSES = {
    "SC":      20,   # Scheduled Caste
    "ST":      20,   # Scheduled Tribe
    "OBC":     12,   # Other Backward Class
    "General":  0,   # No bonus
    "NRI":    -10,   # Non Resident Indian (slight penalty)
}


def compute_academic_score(row, weights=None):
    """
    Compute the weighted Academic Score Indicator from five grade percentages.
    Weights must sum to 1.0 (they are normalised automatically if they don't).
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        # Normalise
        weights = {k: v / total_weight for k, v in weights.items()}

    score = (
        row.get("class_8_pct",  row.get("CGPA", 70) * 10) * weights["class_8"]
        + row.get("class_9_pct",  row.get("CGPA", 70) * 10) * weights["class_9"]
        + row.get("class_10_pct", row.get("CGPA", 70) * 10) * weights["class_10"]
        + row.get("class_11_pct", row.get("CGPA", 70) * 10) * weights["class_11"]
        + row.get("class_12_pct", row.get("CGPA", 70) * 10) * weights["class_12"]
    )
    # Clamp to [0, 100]
    return float(np.clip(score, 0, 100))


def determine_eligibility(row, weights=None, category_bonuses=None):
    """
    Rule-based eligibility scoring system (generates ground-truth labels).
    Academic merit now uses the Academic Score Indicator (0-100 scale)
    instead of CGPA (0-10).

    category_bonuses: dict mapping category name → integer bonus points.
                      Defaults to DEFAULT_CATEGORY_BONUSES if not supplied.
    """
    if category_bonuses is None:
        category_bonuses = DEFAULT_CATEGORY_BONUSES

    score = 0

    # Academic merit — Academic Score Indicator scaled to 0-40
    academic_score = compute_academic_score(row, weights)
    score += (academic_score / 100) * 40

    # Financial need (max 30)
    income_score = max(0, (300000 - row["Parents_Income"]) / 300000) * 30
    score += income_score

    # Reservation bonus — user-configurable
    score += category_bonuses.get(row["Category"], 0)

    # Location disadvantage (max 8)
    score += 8 if row["Location"] == "rural" else 0

    # Age preference (max 5)
    score += max(0, (30 - row["Age"]) / 12) * 5

    return 1 if score >= 45 else 0


def load_and_prepare(path: str, weights=None, category_bonuses=None):
    df = pd.read_excel(path)

    # ── Synthesise per-grade percentages if not present ──────────────────────
    # If the dataset only has CGPA, derive approximate percentages by mapping
    # CGPA (0-10) → percentage (0-100) with minor random variation per grade
    # to simulate realistic variance. Replace this block once real data is available.
    grade_cols = ["class_8_pct", "class_9_pct", "class_10_pct", "class_11_pct", "class_12_pct"]
    for col in grade_cols:
        if col not in df.columns:
            base = df["CGPA"] * 10  # CGPA 7.5 → ~75%
            noise = np.random.normal(0, 4, size=len(df))
            df[col] = np.clip(base + noise, 30, 100).round(2)

    # Compute Academic Score Indicator for each student
    df["academic_score"] = df.apply(lambda r: compute_academic_score(r, weights), axis=1)

    # Eligibility labels
    df["eligible"] = df.apply(lambda r: determine_eligibility(r, weights, category_bonuses), axis=1)

    # ── Encode categoricals ──────────────────────────────────────────────────
    le_gender   = LabelEncoder()
    le_location = LabelEncoder()
    le_category = LabelEncoder()

    df["gender_enc"]   = le_gender.fit_transform(df["gender"])
    df["location_enc"] = le_location.fit_transform(df["Location"])
    df["category_enc"] = le_category.fit_transform(df["Category"])

    feature_cols = [
        "gender_enc", "location_enc", "category_enc",
        "class_8_pct", "class_9_pct", "class_10_pct", "class_11_pct", "class_12_pct",
        "academic_score",
        "Parents_Income", "Age",
    ]
    feature_names = [
        "Gender", "Location", "Category",
        "Class 8 %", "Class 9 %", "Class 10 %", "Class 11 %", "Class 12 %",
        "Academic Score",
        "Parents Income", "Age",
    ]

    X = df[feature_cols].values
    y = df["eligible"].values

    encoders = {
        "gender":   {v: int(le_gender.transform([v])[0])   for v in le_gender.classes_},
        "location": {v: int(le_location.transform([v])[0]) for v in le_location.classes_},
        "category": {v: int(le_category.transform([v])[0]) for v in le_category.classes_},
    }

    return X, y, encoders, feature_names, feature_cols, df


def compute_local_shap(model, X_sample: np.ndarray, n_features: int) -> np.ndarray:
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


def train(data_path: str = DATA_PATH, weights=None, category_bonuses=None):
    if weights is None:
        weights = DEFAULT_WEIGHTS

    if category_bonuses is None:
        category_bonuses = DEFAULT_CATEGORY_BONUSES

    # Normalise grade weights
    total = sum(weights.values())
    if abs(total - 1.0) > 1e-6:
        weights = {k: v / total for k, v in weights.items()}

    print("Loading data …")
    X, y, encoders, feature_names, feature_cols, df = load_and_prepare(data_path, weights, category_bonuses)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Random Forest …")
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    acc    = accuracy_score(y_test, y_pred)
    auc    = roc_auc_score(y_test, y_prob)

    print(f"\nAccuracy : {acc:.4f}")
    print(f"ROC-AUC  : {auc:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Not Eligible", "Eligible"]))

    # Sample SHAP examples
    sample_shapley = []
    for i in range(min(10, len(X_test))):
        contribs = compute_local_shap(model, X_test[i], len(feature_names))
        sample_shapley.append({
            "contributions": contribs.tolist(),
            "prediction":    int(y_pred[i]),
            "actual":        int(y_test[i]),
        })

    joblib.dump(model, f"{MODEL_DIR}/random_forest.joblib")

    meta = {
        "accuracy":               round(acc, 4),
        "roc_auc":                round(auc, 4),
        "feature_names":          feature_names,
        "feature_importances":    model.feature_importances_.tolist(),
        "encoders":               encoders,
        "training_samples":       len(X_train),
        "test_samples":           len(X_test),
        "eligibility_rate":       float(df["eligible"].mean()),
        "threshold":              45,
        "default_weights":        weights,
        "default_category_bonuses": category_bonuses,
        "sample_shap_examples":   sample_shapley,
        "stats": {
            "class_8_pct":  {"min": round(float(df["class_8_pct"].min()),  2), "max": round(float(df["class_8_pct"].max()),  2), "mean": round(float(df["class_8_pct"].mean()),  2)},
            "class_9_pct":  {"min": round(float(df["class_9_pct"].min()),  2), "max": round(float(df["class_9_pct"].max()),  2), "mean": round(float(df["class_9_pct"].mean()),  2)},
            "class_10_pct": {"min": round(float(df["class_10_pct"].min()), 2), "max": round(float(df["class_10_pct"].max()), 2), "mean": round(float(df["class_10_pct"].mean()), 2)},
            "class_11_pct": {"min": round(float(df["class_11_pct"].min()), 2), "max": round(float(df["class_11_pct"].max()), 2), "mean": round(float(df["class_11_pct"].mean()), 2)},
            "class_12_pct": {"min": round(float(df["class_12_pct"].min()), 2), "max": round(float(df["class_12_pct"].max()), 2), "mean": round(float(df["class_12_pct"].mean()), 2)},
            "academic_score": {"min": round(float(df["academic_score"].min()), 2), "max": round(float(df["academic_score"].max()), 2), "mean": round(float(df["academic_score"].mean()), 2)},
            "income": {"min": int(df["Parents_Income"].min()), "max": int(df["Parents_Income"].max()), "mean": round(float(df["Parents_Income"].mean()), 0)},
            "age":    {"min": int(df["Age"].min()),            "max": int(df["Age"].max()),            "mean": round(float(df["Age"].mean()), 2)},
        },
    }

    with open(f"{MODEL_DIR}/model_meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\nArtifacts saved to ./{MODEL_DIR}/")
    print(f"  random_forest.joblib")
    print(f"  model_meta.json")
    return model, meta


if __name__ == "__main__":
    train()
