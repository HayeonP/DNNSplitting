import pickle
import sys

def load_task_set_list(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    if not isinstance(data, dict):
        raise ValueError("Unexpected pickle format (expected dict).")

    task_set_list = data.get("task_set_list", [])
    return data, task_set_list


def load_yaml(path):
    try:
        import yaml  # type: ignore
    except Exception:
        yaml = None

    if yaml is not None:
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}

    config = {}
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if ":" not in line:
                continue
            key, value = line.split(":", 1)
            config[key.strip()] = value.strip()
    return config


def log_debug(enabled, *args, **kwargs):
    if enabled:
        print(*args, **kwargs)


def log_info(*args, **kwargs):
    print(*args, **kwargs)


def log_error(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)
