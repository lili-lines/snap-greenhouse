"""
for loading the config from a YAML & making it as a dict
"""
from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"

cfg = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))