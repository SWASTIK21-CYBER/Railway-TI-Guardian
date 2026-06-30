"""
dashboard_ui.py
Flask dashboard for Railway-TI-Guardian. Displays live simulated
railway traffic events, threat verdicts, and RRB-LB queue/load stats.

Run via main.py, or directly: python -m app.dashboard_ui
"""
import os
import sys
import threading
from collections import deque

from flask import Flask, render_template, jsonify

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils import load_config
from src.detector import RRBLoadBalancer
from src.simulator import RailwaySimulator

app = Flask(__name__)

config = load_config()
recent_results = deque(maxlen=50)
_results_lock = threading.Lock()

rrb = None
sim = None


def _on_result(result):
    with _results_lock:
        recent_results.appendleft(result)


def _on_event(event):
    rrb.submit(event)


def start_pipeline():
    """Starts RRB-LB workers and the railway event simulator. Called once."""
    global rrb, sim
    if rrb is not None:
        return
    rrb = RRBLoadBalancer(config=config, on_result=_on_result)
    rrb.start()

    sim_cfg = config["simulator"]
    sim = RailwaySimulator(
        on_event=_on_event,
        events_per_batch=sim_cfg["events_per_batch"],
        interval_seconds=sim_cfg["interval_seconds"],
        attack_ratio=sim_cfg["attack_ratio"],
    )
    sim.start()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/alerts")
def api_alerts():
    with _results_lock:
        data = list(recent_results)
    return jsonify(data)


@app.route("/api/stats")
def api_stats():
    stats = rrb.stats() if rrb else {"queue_length": 0, "processed": 0, "dropped": 0}
    with _results_lock:
        threats = sum(1 for r in recent_results if r.get("verdict") == "THREAT")
        normal = sum(1 for r in recent_results if r.get("verdict") == "NORMAL")
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0, "NORMAL": 0}
        for r in recent_results:
            sev = r.get("severity", "NORMAL")
            if sev in severity_counts:
                severity_counts[sev] += 1

        # Overall system status, driven by the most recent severity seen,
        # used to drive the signal-lamp indicator on the dashboard.
        if recent_results:
            latest_severities = [r.get("severity", "NORMAL") for r in list(recent_results)[:5]]
        else:
            latest_severities = []
        if any(s == "CRITICAL" for s in latest_severities):
            system_status = "RED"
        elif any(s in ("HIGH", "MEDIUM") for s in latest_severities):
            system_status = "AMBER"
        else:
            system_status = "GREEN"

    stats["threats_recent"] = threats
    stats["normal_recent"] = normal
    stats["severity_counts"] = severity_counts
    stats["system_status"] = system_status
    return jsonify(stats)


if __name__ == "__main__":
    start_pipeline()
    app.run(debug=False, port=5000, use_reloader=False)
