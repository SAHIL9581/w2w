#!/usr/bin/env python3

import os
import json
import argparse
import sys
from joblib import load
import matplotlib.pyplot as plt

def setup_and_plot_scaler(config_path: str = "config.json"):
    """
    - Load JSON config      
    - Print & load std_scaler_path    
    - Set config['finetuning']['model_params']['in_channels']    
    - Print loaded feature count    
    - Plot feature means & scales    
    """
    # ——— Load config ———
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)

    # ——— Get scaler path ———
    try:
        scaler_path = config["paths"]["std_scaler_path"]
    except KeyError:
        raise KeyError("Missing 'std_scaler_path' under 'paths' in config")

    print(f"Standard scaler path from config: {scaler_path}")

    # ——— Load scaler ———
    if not os.path.isfile(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    scaler = load(scaler_path)

    # ——— Set in_channels and print feature count ———
    # Ensure the nested dicts exist
    config.setdefault("finetuning", {})
    config["finetuning"].setdefault("model_params", {})
    # Here’s the key line you requested:
    config["finetuning"]["model_params"]["in_channels"] = scaler.n_features_in_
    print(f"Loaded scaler with {scaler.n_features_in_} features.")

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Set model in_channels from scaler and plot its stats"
    )
    parser.add_argument(
        "-c", "--config",
        default="config.json",
        help="Path to JSON config (default: config.json)"
    )
    args = parser.parse_args()

    try:
        updated_config = setup_and_plot_scaler(args.config)
    except Exception as e:
        print(f"[ERROR] {e}", file=sys.stderr)
        sys.exit(1)
