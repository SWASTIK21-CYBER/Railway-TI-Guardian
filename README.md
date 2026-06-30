# Railway-TI-Guardian

A simulated **Railway Threat Intelligence and Detection** system. It detects
malicious network traffic targeting railway signaling/control systems using a
hybrid Machine Learning model, prioritized and dispatched at the edge by a
custom load balancer (**RRB-LB**).

## Architecture

```
NSL-KDD data --> preprocessing --> Isolation Forest (anomaly score)
                                          |
                                          v
                          Random Forest (hybrid features) --> verdict
                                          ^
                                          |
Simulated railway traffic ---> RRB-LB (priority queue, edge workers) 
                                          |
                                          v
                                 Flask Dashboard (live view)
```

### Models
- **Isolation Forest** — unsupervised anomaly scoring on encoded traffic features.
- **Random Forest** — supervised classifier trained on the original features
  plus the Isolation Forest anomaly score as an extra "hybrid" feature.
- Trained and evaluated on the **NSL-KDD** intrusion detection dataset.

### RRB-LB — Real-Time Reasoning & Preference Load Balancer
A custom-proposed edge algorithm (`src/detector.py`). It runs entirely
locally — no cloud dependency — simulating deployment inside a train
locomotive cabin or station edge server. Incoming events are placed in a
priority queue scored by:
- **Train criticality** (high-speed passenger / signaling backbone > freight > yard/depot)
- **Signal urgency** (interlocking control > track circuit/point machine > telemetry)

A small worker pool (simulating limited edge compute) processes the queue
highest-priority-first. If the queue fills up, RRB-LB **sheds load** by
dropping the lowest-priority queued events rather than letting the edge
node stall or crash.

### Simulator
Since there's no real locomotive/station sensor feed available,
`src/simulator.py` generates synthetic railway traffic events (tagged with
train ID, criticality, and signal context) with a configurable mix of
normal vs. attack-like patterns.

## Project Structure
```
Railway-TI-Guardian/
├── config/config.yaml       # all paths, model params, RRB-LB weights
├── data/                    # NSL-KDD KDDTrain+.txt / KDDTest+.txt
├── models/                  # saved .joblib models (created after training)
├── logs/security_logs.txt   # running event log
├── src/
│   ├── loader.py            # loads NSL-KDD data
│   ├── processor.py         # preprocessing pipeline (encode/scale)
│   ├── train.py             # trains + saves the hybrid model
│   ├── classifier.py        # loads models, exposes predict()
│   ├── detector.py          # RRB-LB priority queue + edge workers
│   ├── simulator.py         # generates fake railway traffic
│   └── utils.py             # config + logging helpers
├── app/
│   ├── dashboard_ui.py      # Flask app
│   └── templates/index.html # live dashboard UI
└── main.py                  # entrypoint: trains (if needed) + launches dashboard
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## Run

```bash
python main.py
```

Then open **http://127.0.0.1:5000** in your browser.

On first run this will train the hybrid model on NSL-KDD (~30-60 seconds)
and save the result to `models/`. On subsequent runs it loads the saved
models directly. The simulator and RRB-LB start automatically, and the
dashboard refreshes every 2 seconds showing live verdicts, train info, and
queue/load-shedding stats.

## Retraining
Delete the files in `models/` and re-run `python main.py`, or run training
directly:
```bash
python -m src.train
```
