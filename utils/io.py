import os, json, time
from pathlib import Path

def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)
    return path

def save_json(obj, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

class SimpleLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        ensure_dir(os.path.dirname(log_path))
        self._fp = open(log_path, "a", encoding="utf-8")
    def log(self, msg: str):
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line)
        self._fp.write(line + "\n")
        self._fp.flush()
    def close(self):
        self._fp.close()
