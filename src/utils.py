"""
utils.py
Shared helpers: config loading, project paths, simple logger.
"""
import os
import yaml
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_config(config_path=None):
    """Load config/config.yaml and return as a dict."""
    if config_path is None:
        config_path = os.path.join(PROJECT_ROOT, "config", "config.yaml")
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def resolve_path(relative_path):
    """Resolve a path from config relative to the project root."""
    return os.path.join(PROJECT_ROOT, relative_path)


def log_event(message, log_path=None):
    """Append a timestamped line to the security log (and print it)."""
    if log_path is None:
        log_path = resolve_path("logs/security_logs.txt")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}"
    with open(log_path, "a") as f:
        f.write(line + "\n")
    print(line)
    return line
