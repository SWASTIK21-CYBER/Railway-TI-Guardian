"""
train.py
Trains the hybrid threat-detection model:
  Stage 1: Isolation Forest produces an anomaly score per record.
  Stage 2: Random Forest classifies normal/threat using the original
           features PLUS the anomaly score as an extra feature.

Run directly: python -m src.train
"""
import os
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from src.utils import load_config, resolve_path, log_event
from src.loader import load_nsl_kdd, add_binary_target
from src.processor import build_preprocessor, split_features_target


def train_and_save(config=None):
    if config is None:
        config = load_config()

    train_data, test_data = load_nsl_kdd(config)
    train_data = add_binary_target(train_data)
    test_data = add_binary_target(test_data)

    X_train, y_train = split_features_target(train_data)
    X_test, y_test = split_features_target(test_data)

    print("Preprocessing features...")
    preprocessor = build_preprocessor(config)
    X_train_encoded = preprocessor.fit_transform(X_train)
    X_test_encoded = preprocessor.transform(X_test)

    iso_cfg = config["model"]["isolation_forest"]
    print("Training Isolation Forest (Stage 1: anomaly scoring)...")
    iso_forest = IsolationForest(
        n_estimators=iso_cfg["n_estimators"],
        contamination=iso_cfg["contamination"],
        random_state=iso_cfg["random_state"],
        n_jobs=-1,
    )
    iso_forest.fit(X_train_encoded)

    train_scores = iso_forest.decision_function(X_train_encoded).reshape(-1, 1)
    test_scores = iso_forest.decision_function(X_test_encoded).reshape(-1, 1)

    X_train_hybrid = np.hstack((X_train_encoded, train_scores))
    X_test_hybrid = np.hstack((X_test_encoded, test_scores))

    rf_cfg = config["model"]["random_forest"]
    print("Training Random Forest (Stage 2: classification on hybrid features)...")
    rf_model = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_split=rf_cfg["min_samples_split"],
        class_weight=rf_cfg["class_weight"],
        random_state=rf_cfg["random_state"],
        n_jobs=-1,
    )
    rf_model.fit(X_train_hybrid, y_train)

    print("\nEvaluating on test set...")
    y_pred = rf_model.predict(X_test_hybrid)
    accuracy = accuracy_score(y_test, y_pred)
    conf = confusion_matrix(y_test, y_pred)

    print("\n=================== HYBRID MODEL EVALUATION ===================")
    print(f"Final Combined Accuracy: {accuracy * 100:.2f}%\n")
    print("Confusion Matrix:")
    print(f"  True Normals Correctly Flagged:    {conf[0][0]}")
    print(f"  False Alarms (Normal as Threat):   {conf[0][1]}")
    print(f"  Missed Threats (Threat as Normal): {conf[1][0]}")
    print(f"  True Threats Correctly Flagged:    {conf[1][1]}\n")
    print(classification_report(y_test, y_pred, target_names=["Normal (0)", "Threat (1)"]))
    print("===============================================================")

    # Save artifacts
    pre_path = resolve_path(config["paths"]["preprocessor"])
    iso_path = resolve_path(config["paths"]["isolation_forest"])
    rf_path = resolve_path(config["paths"]["hybrid_rf_model"])
    os.makedirs(os.path.dirname(pre_path), exist_ok=True)

    joblib.dump(preprocessor, pre_path)
    joblib.dump(iso_forest, iso_path)
    joblib.dump(rf_model, rf_path)
    print(f"\nModels saved to: {os.path.dirname(rf_path)}")

    log_event(
        f"Model trained. Accuracy={accuracy * 100:.2f}% "
        f"TP={conf[1][1]} TN={conf[0][0]} FP={conf[0][1]} FN={conf[1][0]}",
        log_path=resolve_path(config["paths"]["security_log"]),
    )

    return {"accuracy": accuracy, "confusion_matrix": conf.tolist()}


if __name__ == "__main__":
    train_and_save()
