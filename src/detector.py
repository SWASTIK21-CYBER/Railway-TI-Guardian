"""
detector.py

Implements RRB-LB: Real-Time Reasoning & Preference Load Balancer.

This is the custom edge-deployed component described in the project:
it runs locally (no cloud dependency), accepts incoming traffic events
from train/station sensors, and decides PROCESSING ORDER based on:
  - train criticality  (how important is the train this event relates to)
  - signal urgency      (how time-sensitive the traffic context is)

It then dispatches events to a small worker pool that runs the
HybridClassifier (Isolation Forest + Random Forest) and logs the verdict.

Under heavy load, lowest-priority queued events are dropped
(load shedding) instead of crashing or stalling the edge node --
this is the "prevent localized computing crashes" requirement.
"""
import heapq
import itertools
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from src.utils import load_config, resolve_path, log_event
from src.classifier import HybridClassifier


@dataclass(order=True)
class _QueueItem:
    priority: int
    seq: int
    event: dict = field(compare=False)


class RRBLoadBalancer:
    """
    Real-Time Reasoning & Preference Load Balancer (RRB-LB).

    Usage:
        rrb = RRBLoadBalancer(on_result=my_callback)
        rrb.start()
        rrb.submit(event)   # event includes train_criticality + signal_urgency
        ...
        rrb.stop()
    """

    def __init__(self, config=None, on_result: Optional[Callable[[dict], None]] = None):
        self.config = config or load_config()
        rrb_cfg = self.config["rrb_lb"]

        self.criticality_weights = rrb_cfg["criticality_weights"]
        self.urgency_weights = rrb_cfg["urgency_weights"]
        self.max_queue_size = rrb_cfg["max_queue_size"]
        self.worker_threads = rrb_cfg["worker_threads"]

        self._heap = []
        self._counter = itertools.count()
        self._lock = threading.Lock()
        self._not_empty = threading.Condition(self._lock)
        self._running = False
        self._workers = []
        self._dropped_count = 0
        self._processed_count = 0

        self.on_result = on_result
        self.classifier = HybridClassifier(self.config)
        self.log_path = resolve_path(self.config["paths"]["security_log"])

    # ---------- Priority scoring ----------
    def _compute_priority(self, event: dict) -> int:
        """
        Lower number = higher priority (min-heap).
        Combines train criticality + signal urgency into a single score.
        """
        crit = self.criticality_weights.get(event.get("train_criticality", "low"), 1)
        urg = self.urgency_weights.get(event.get("signal_urgency", "normal"), 1)
        combined_weight = crit + urg
        # Invert so higher combined_weight => lower (higher-priority) number
        max_possible = max(self.criticality_weights.values()) + max(self.urgency_weights.values())
        return max_possible - combined_weight

    # ---------- Queue management ----------
    def submit(self, event: dict):
        """Add an event to the priority queue. Sheds lowest-priority load if full."""
        priority = self._compute_priority(event)
        with self._not_empty:
            if len(self._heap) >= self.max_queue_size:
                # Load shedding: drop the current lowest-priority (highest number) item
                # if the new event is more urgent than it.
                worst_priority = max(item.priority for item in self._heap)
                if priority < worst_priority:
                    worst_idx = max(range(len(self._heap)), key=lambda i: self._heap[i].priority)
                    dropped = self._heap.pop(worst_idx)
                    heapq.heapify(self._heap)
                    self._dropped_count += 1
                    log_event(
                        f"RRB-LB LOAD SHED: dropped lower-priority event "
                        f"(train={dropped.event.get('train_id')}) to make room for "
                        f"higher-priority event (train={event.get('train_id')})",
                        log_path=self.log_path,
                    )
                else:
                    self._dropped_count += 1
                    log_event(
                        f"RRB-LB LOAD SHED: queue full, dropped incoming low-priority "
                        f"event (train={event.get('train_id')})",
                        log_path=self.log_path,
                    )
                    return

            item = _QueueItem(priority=priority, seq=next(self._counter), event=event)
            heapq.heappush(self._heap, item)
            self._not_empty.notify()

    def _next(self, timeout=0.5):
        with self._not_empty:
            if not self._heap:
                self._not_empty.wait(timeout=timeout)
            if not self._heap:
                return None
            return heapq.heappop(self._heap)

    # ---------- Worker pool (simulates limited edge compute) ----------
    def _worker_loop(self, worker_id: int):
        while self._running:
            item = self._next()
            if item is None:
                continue
            event = item.event
            try:
                result = self.classifier.predict(event)
            except Exception as e:
                result = {"verdict": "ERROR", "anomaly_score": None, "confidence": None, "error": str(e)}

            result["event"] = event
            result["priority"] = item.priority
            result["worker_id"] = worker_id

            self._processed_count += 1

            if result["verdict"] == "THREAT":
                log_event(
                    f"[RRB-LB worker-{worker_id}] THREAT DETECTED | train={event.get('train_id')} "
                    f"criticality={event.get('train_criticality')} urgency={event.get('signal_urgency')} "
                    f"confidence={result.get('confidence', 0):.2f} action=BLOCK",
                    log_path=self.log_path,
                )
            else:
                log_event(
                    f"[RRB-LB worker-{worker_id}] normal traffic | train={event.get('train_id')} "
                    f"action=ALLOW",
                    log_path=self.log_path,
                )

            if self.on_result:
                self.on_result(result)

    def start(self):
        if self._running:
            return
        self._running = True
        self._workers = [
            threading.Thread(target=self._worker_loop, args=(i,), daemon=True)
            for i in range(self.worker_threads)
        ]
        for w in self._workers:
            w.start()
        log_event(
            f"RRB-LB started with {self.worker_threads} edge worker(s), "
            f"max queue size {self.max_queue_size}",
            log_path=self.log_path,
        )

    def stop(self):
        self._running = False
        for w in self._workers:
            w.join(timeout=2)

    def stats(self):
        with self._lock:
            return {
                "queue_length": len(self._heap),
                "processed": self._processed_count,
                "dropped": self._dropped_count,
            }
