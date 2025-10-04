import os
import yaml

def load_yaml(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def module_path(*parts) -> str:
    return os.path.join(os.path.dirname(__file__), *parts)
