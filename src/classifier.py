"""
classifier.py
Wraps the trained preprocessor + Isolation Forest + Random Forest
into a single HybridClassifier with one predict(event_dict) call.
"""
import os
import joblib
import numpy as np
import pandas as pd

from src.utils import load_config, resolve_path


class HybridClassifier:
    def __init__(self, config=None):
        self.config = config or load_config()
        pre_path = resolve_path(self.config["paths"]["preprocessor"])
        iso_path = resolve_path(self.config["paths"]["isolation_forest"])
        rf_path = resolve_path(self.config["paths"]["hybrid_rf_model"])

        for p in (pre_path, iso_path, rf_path):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Model file not found: {p}\n"
                    f"Run `python -m src.train` first to train and save the models."
                )

        self.preprocessor = joblib.load(pre_path)
        self.iso_forest = joblib.load(iso_path)
        self.rf_model = joblib.load(rf_path)

    def predict(self, event: dict):
        """
        event: dict with keys duration, protocol_type, service, flag,
               src_bytes, dst_bytes
        Returns: dict with verdict ("THREAT"/"NORMAL"), anomaly_score,
                 and confidence (probability of threat class).
        """
        df = pd.DataFrame([{
            "duration": event["duration"],
            "protocol_type": event["protocol_type"],
            "service": event["service"],
            "flag": event["flag"],
            "src_bytes": event["src_bytes"],
            "dst_bytes": event["dst_bytes"],
        }])

        encoded = self.preprocessor.transform(df)
        anomaly_score = self.iso_forest.decision_function(encoded)
        hybrid_features = np.hstack((encoded, anomaly_score.reshape(-1, 1)))

        prediction = self.rf_model.predict(hybrid_features)[0]
        proba = self.rf_model.predict_proba(hybrid_features)[0]
        threat_confidence = float(proba[1]) if len(proba) > 1 else float(prediction)

        return {
            "verdict": "THREAT" if prediction == 1 else "NORMAL",
            "anomaly_score": float(anomaly_score[0]),
            "confidence": threat_confidence,
            "severity": classify_severity(prediction == 1, threat_confidence),
        }


def classify_severity(is_threat: bool, confidence: float) -> str:
    """
    Maps a verdict + confidence score into a human-readable severity level
    for the dashboard (used to color-code and prioritize alerts visually).
    """
    if not is_threat:
        return "NORMAL"
    if confidence >= 0.85:
        return "CRITICAL"
    if confidence >= 0.65:
        return "HIGH"
    if confidence >= 0.5:
        return "MEDIUM"
    return "LOW"