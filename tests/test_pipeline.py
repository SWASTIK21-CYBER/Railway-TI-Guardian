"""
test_pipeline.py
Basic smoke tests for the training/inference pipeline.
Run with: pytest tests/test_pipeline.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config, resolve_path
from src.classifier import HybridClassifier
from src.detector import RRBLoadBalancer
from src.simulator import _generate_event


def test_config_loads():
    config = load_config()
    assert "rrb_lb" in config
    assert "model" in config


def test_classifier_predicts():
    config = load_config()
    rf_path = resolve_path(config["paths"]["hybrid_rf_model"])
    if not os.path.exists(rf_path):
        import pytest
        pytest.skip("Models not trained yet -- run `python -m src.train` first.")

    clf = HybridClassifier(config)
    event = _generate_event(is_attack=False)
    result = clf.predict(event)
    assert result["verdict"] in ("NORMAL", "THREAT")
    assert 0.0 <= result["confidence"] <= 1.0


def test_rrb_priority_ordering():
    config = load_config()
    rrb_cfg = config["rrb_lb"]
    high_priority_event = {"train_criticality": "high", "signal_urgency": "critical"}
    low_priority_event = {"train_criticality": "low", "signal_urgency": "normal"}

    rf_path = resolve_path(config["paths"]["hybrid_rf_model"])
    if not os.path.exists(rf_path):
        import pytest
        pytest.skip("Models not trained yet -- run `python -m src.train` first.")

    rrb = RRBLoadBalancer(config=config)
    high_score = rrb._compute_priority(high_priority_event)
    low_score = rrb._compute_priority(low_priority_event)
    # Lower number = higher priority (processed first)
    assert high_score < low_score
