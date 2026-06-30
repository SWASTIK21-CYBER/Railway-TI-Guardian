"""
main.py
Single entrypoint for Railway-TI-Guardian.

- Trains the hybrid model if saved model files don't exist yet.
- Launches the Flask dashboard, which starts the RRB-LB workers
  and the simulated railway traffic generator.

Run: python main.py
Then open http://127.0.0.1:5000 in your browser.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.utils import load_config, resolve_path


def ensure_models_trained(config):
    rf_path = resolve_path(config["paths"]["hybrid_rf_model"])
    iso_path = resolve_path(config["paths"]["isolation_forest"])
    pre_path = resolve_path(config["paths"]["preprocessor"])

    if all(os.path.exists(p) for p in (rf_path, iso_path, pre_path)):
        print("Trained models found. Skipping training.")
        return

    print("No trained models found. Training now (this may take a minute)...")
    from src.train import train_and_save
    train_and_save(config)


def main():
    config = load_config()
    ensure_models_trained(config)

    print("\nStarting Railway-TI-Guardian dashboard at http://127.0.0.1:5000 ...")
    from app.dashboard_ui import app, start_pipeline
    start_pipeline()
    app.run(debug=False, port=5000, use_reloader=False)


if __name__ == "__main__":
    main()
