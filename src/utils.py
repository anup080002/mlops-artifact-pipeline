import json, pathlib

def load_config(cfg_path="config/config.json"):
    with open(cfg_path) as f:
        return json.load(f)
