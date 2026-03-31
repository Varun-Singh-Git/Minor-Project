"""
predict.py
==========
Single-image prediction pipeline.

Given the path to a raw leaf image, this script:
  1. Preprocesses the image
  2. Extracts the 10-D feature vector
  3. Loads the trained Decision Tree model
  4. Predicts the disease class
  5. Returns confidence score (probability estimate)
  6. Extracts and displays the decision path (IF-THEN rules)

Usage:
    python src/predict.py --image path/to/leaf.jpg

    Or import and call predict_image() from app.py
"""

import os
import sys
import argparse
import numpy as np

from sklearn.tree import DecisionTreeClassifier, _tree

sys.path.insert(0, os.path.dirname(__file__))
from feature_extraction import extract_features_from_path
from train import load_model

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
MODEL_PATH = "models/decision_tree_model.pkl"

FEATURE_NAMES = [
    "contrast", "energy", "homogeneity", "correlation",
    "red_mean", "green_mean", "blue_mean",
    "red_std",  "green_std",  "blue_std"
]


# ──────────────────────────────────────────────
# Extract decision path as human-readable rules
# ──────────────────────────────────────────────
def get_decision_rules(clf: DecisionTreeClassifier,
                        feature_vector: np.ndarray,
                        feature_names: list,
                        class_names: list) -> list:
    """
    Trace the decision path through the tree for a specific input sample
    and return each IF-THEN rule as a plain-English string.

    This is the key interpretability feature of the Decision Tree —
    every prediction is backed by auditable human-readable rules.

    Args:
        clf:            Trained DecisionTreeClassifier.
        feature_vector: 2D array of shape (1, n_features).
        feature_names:  List of feature name strings.
        class_names:    List of class label strings.

    Returns:
        List of rule strings, e.g.:
            ["contrast <= 0.427", "green_mean > 0.312", "→ Leaf Spot (confidence: 94.5%)"]
    """
    tree      = clf.tree_
    node_ids  = clf.decision_path(feature_vector).indices

    rules = []
    for i, node_id in enumerate(node_ids):
        # Check if this is a leaf node
        if tree.children_left[node_id] == _tree.TREE_LEAF:
            # Leaf: output predicted class and confidence
            class_counts = tree.value[node_id][0]
            total        = class_counts.sum()
            pred_class   = class_names[np.argmax(class_counts)]
            confidence   = (class_counts.max() / total) * 100
            rules.append(f"→ Prediction: {pred_class}  (confidence: {confidence:.1f}%)")
        else:
            # Internal node: output the feature split condition
            feat_idx   = tree.feature[node_id]
            threshold  = tree.threshold[node_id]
            feat_name  = feature_names[feat_idx]
            feat_value = feature_vector[0, feat_idx]

            # Determine which branch was taken
            if feat_value <= threshold:
                operator = "<="
            else:
                operator = ">"

            rules.append(f"IF {feat_name} {operator} {threshold:.4f}  "
                         f"(your value: {feat_value:.4f})")

    return rules


# ──────────────────────────────────────────────
# Main prediction function
# ──────────────────────────────────────────────
def predict_image(image_path: str, model_path: str = MODEL_PATH) -> dict:
    """
    Full prediction pipeline for a single leaf image.

    Args:
        image_path: Path to the raw input image.
        model_path: Path to the trained model pickle file.

    Returns:
        Dictionary with:
            predicted_class  : str   — disease class name
            confidence       : float — confidence percentage (0–100)
            all_probabilities: dict  — confidence per class
            decision_rules   : list  — human-readable IF-THEN rule path
            feature_values   : dict  — extracted feature values
    """
    # ── 1. Load model ──
    clf, label_encoder = load_model(model_path)
    class_names = list(label_encoder.classes_)

    # ── 2. Extract features ──
    feature_vector = extract_features_from_path(image_path)   # shape (1, 10)

    # ── 3. Predict class ──
    pred_encoded  = clf.predict(feature_vector)[0]
    predicted_class = label_encoder.inverse_transform([pred_encoded])[0]

    # ── 4. Get confidence (from leaf node class distribution) ──
    leaf_id      = clf.apply(feature_vector)[0]
    leaf_counts  = clf.tree_.value[leaf_id][0]
    total        = leaf_counts.sum()
    confidence   = float((leaf_counts.max() / total) * 100)

    all_probs = {
        class_names[i]: round(float(leaf_counts[i] / total * 100), 2)
        for i in range(len(class_names))
    }

    # ── 5. Get decision rules ──
    rules = get_decision_rules(clf, feature_vector, FEATURE_NAMES, class_names)

    # ── 6. Format feature values for display ──
    feature_values = {
        name: round(float(feature_vector[0, i]), 4)
        for i, name in enumerate(FEATURE_NAMES)
    }

    return {
        "predicted_class":   predicted_class,
        "confidence":        round(confidence, 2),
        "all_probabilities": all_probs,
        "decision_rules":    rules,
        "feature_values":    feature_values,
    }


# ──────────────────────────────────────────────
# Disease advice mapping
# ──────────────────────────────────────────────
DISEASE_ADVICE = {
    "Healthy": {
        "description": "The leaf shows no signs of disease.",
        "action":      "No treatment needed. Continue regular monitoring.",
        "severity":    "None"
    },
    "Leaf_Spot": {
        "description": "Circular necrotic spots detected on the leaf surface.",
        "action":      "Apply copper-based fungicide. Remove severely infected leaves. Avoid overhead irrigation.",
        "severity":    "Moderate"
    },
    "Leaf_Curl": {
        "description": "Upward or downward curling of leaf edges detected.",
        "action":      "Check for aphid or mite infestation. Apply appropriate insecticide/miticide.",
        "severity":    "Moderate"
    },
    "Slug_Damage": {
        "description": "Irregular holes and edge damage consistent with slug feeding.",
        "action":      "Use slug pellets around plant base. Remove debris where slugs hide.",
        "severity":    "Low-Moderate"
    },
    "Fruit_Disease": {
        "description": "Disease symptoms detected on fruit tissue.",
        "action":      "Remove infected fruits immediately. Apply fungicide spray. Improve air circulation.",
        "severity":    "High"
    },
}


def get_advice(predicted_class: str) -> dict:
    """Return treatment advice for a predicted disease class."""
    return DISEASE_ADVICE.get(predicted_class, {
        "description": "Unknown class",
        "action":      "Consult an agricultural expert.",
        "severity":    "Unknown"
    })


# ──────────────────────────────────────────────
# Entry point (CLI usage)
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Crop Disease Detection — Single Image Predictor"
    )
    parser.add_argument(
        "--image", "-i",
        required=True,
        help="Path to the leaf image to classify"
    )
    parser.add_argument(
        "--model", "-m",
        default=MODEL_PATH,
        help=f"Path to the model pickle file (default: {MODEL_PATH})"
    )
    args = parser.parse_args()

    if not os.path.exists(args.image):
        sys.exit(f"[ERROR] Image not found: {args.image}")

    print("=" * 55)
    print("  Crop Disease Detection — Prediction")
    print("=" * 55)

    result = predict_image(args.image, args.model)
    advice = get_advice(result["predicted_class"])

    print(f"\n📋 RESULT")
    print(f"   Predicted Class : {result['predicted_class']}")
    print(f"   Confidence      : {result['confidence']}%")
    print(f"   Severity        : {advice['severity']}")
    print(f"\n📝 Description  : {advice['description']}")
    print(f"💊 Recommended Action: {advice['action']}")

    print(f"\n📊 Confidence per class:")
    for cls, prob in sorted(result["all_probabilities"].items(),
                              key=lambda x: x[1], reverse=True):
        bar = "█" * int(prob / 5)
        print(f"   {cls:<20} {prob:5.1f}%  {bar}")

    print(f"\n🌳 Decision Path (IF-THEN Rules):")
    for i, rule in enumerate(result["decision_rules"], 1):
        indent = "   " + ("  " * min(i - 1, 6))
        print(f"{indent}{rule}")

    print(f"\n🔬 Extracted Feature Values:")
    for feat, val in result["feature_values"].items():
        print(f"   {feat:<20} {val}")
