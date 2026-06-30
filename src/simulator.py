"""
simulator.py
Generates simulated railway network traffic events since no real
locomotive/station sensor feed is available. Produces a mix of normal
and attack-like traffic patterns (loosely modeled on NSL-KDD value
ranges) tagged with train_id, train_criticality, and signal_urgency
so RRB-LB has something meaningful to prioritize.
"""
import random
import time
import threading

TRAINS = [
    {"train_id": "EXP-101", "criticality": "high"},    # high-speed passenger express
    {"train_id": "REG-204", "criticality": "medium"},  # regional passenger
    {"train_id": "FRT-330", "criticality": "medium"},  # freight
    {"train_id": "YRD-009", "criticality": "low"},      # depot/yard shunt
    {"train_id": "SIG-CTRL", "criticality": "high"},    # signaling backbone node
]

SIGNAL_CONTEXTS = [
    {"context": "interlocking_control", "urgency": "critical"},
    {"context": "point_machine", "urgency": "elevated"},
    {"context": "track_circuit", "urgency": "elevated"},
    {"context": "telemetry", "urgency": "normal"},
    {"context": "diagnostics", "urgency": "normal"},
]

PROTOCOLS = ["tcp", "udp", "icmp"]
NORMAL_SERVICES = ["http", "ftp_data", "telnet", "private", "smtp"]
ATTACK_SERVICES = ["private", "eco_i", "finger", "remote_job", "urp_i"]
NORMAL_FLAGS = ["SF", "S1"]
ATTACK_FLAGS = ["S0", "REJ", "RSTO"]


def _generate_event(is_attack: bool) -> dict:
    train = random.choice(TRAINS)
    signal = random.choice(SIGNAL_CONTEXTS)

    if is_attack:
        event = {
            "duration": round(random.uniform(0, 2), 2),
            "protocol_type": random.choice(PROTOCOLS),
            "service": random.choice(ATTACK_SERVICES),
            "flag": random.choice(ATTACK_FLAGS),
            "src_bytes": random.randint(0, 50),
            "dst_bytes": random.randint(0, 10),
        }
    else:
        event = {
            "duration": round(random.uniform(0, 30), 2),
            "protocol_type": random.choice(PROTOCOLS),
            "service": random.choice(NORMAL_SERVICES),
            "flag": random.choice(NORMAL_FLAGS),
            "src_bytes": random.randint(100, 5000),
            "dst_bytes": random.randint(100, 5000),
        }

    event.update({
        "train_id": train["train_id"],
        "train_criticality": train["criticality"],
        "signal_context": signal["context"],
        "signal_urgency": signal["urgency"],
        "is_simulated_attack": is_attack,  # ground truth, for display only
        "timestamp": time.time(),
    })
    return event


class RailwaySimulator:
    """Continuously generates simulated events and feeds them to a sink callback."""

    def __init__(self, on_event, events_per_batch=5, interval_seconds=2, attack_ratio=0.3):
        self.on_event = on_event
        self.events_per_batch = events_per_batch
        self.interval_seconds = interval_seconds
        self.attack_ratio = attack_ratio
        self._running = False
        self._thread = None

    def _loop(self):
        while self._running:
            for _ in range(self.events_per_batch):
                is_attack = random.random() < self.attack_ratio
                event = _generate_event(is_attack)
                self.on_event(event)
            time.sleep(self.interval_seconds)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)