r"""
Settings that persist between launches, in GUI\config.json.

The nunif location is asked for once and reused everywhere. Everything else has
a default under the GUI folder, so a fresh copy of this folder runs with nothing
configured beyond that one path.
"""
from __future__ import annotations

import json
import os
import sys
import threading
from os import path


GUI_DIR = path.dirname(path.dirname(path.abspath(__file__)))
CONFIG_PATH = path.join(GUI_DIR, "config.json")
DATA_DIR = path.join(GUI_DIR, "data")

DEFAULTS = {
    "nunif_dir": "",            # the repo root that contains nunif\ and iw3\
    "python_exe": "",           # blank = derive from nunif_dir
    "download_dir": "",         # blank = <GUI>\data\downloads
    "dataset_dir": "",          # blank = <GUI>\data\dataset
    "models_dir": "",           # blank = <GUI>\data\models
    "pexels_key": "",
    "pixabay_key": "",
    "media_type": "video",      # video | image
    "theme": "midnight",
    "page": 1,
    "collect": {},              # page 1 form state
    "prep": {},                 # page 2 form state
    "train": {},                # page 3 form state
    "live_run": "",             # page 4 selected run
    "install_lowres": True,     # page 5: also install the low-res inpainting patch
    "install_border": True,     # page 5: also install the screen-edge fix
}

_lock = threading.Lock()
_cache = None


def _read():
    if not path.exists(CONFIG_PATH):
        return {}
    try:
        with open(CONFIG_PATH, encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception as e:                                           # noqa: BLE001
        print(f"config.json unreadable ({e}); starting from defaults", file=sys.stderr)
        return {}


def load():
    global _cache
    with _lock:
        if _cache is None:
            merged = dict(DEFAULTS)
            merged.update(_read())
            _cache = merged
        return dict(_cache)


def save(patch):
    """Merge a partial update and write it out. Returns the full config."""
    global _cache
    with _lock:
        cur = dict(DEFAULTS)
        cur.update(_read())
        if _cache:
            cur.update(_cache)
        for k, v in (patch or {}).items():
            if k in ("collect", "prep", "train") and isinstance(v, dict) and isinstance(cur.get(k), dict):
                merged = dict(cur[k])
                merged.update(v)
                cur[k] = merged
            else:
                cur[k] = v
        _cache = cur
        tmp = CONFIG_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(cur, f, indent=2)
        os.replace(tmp, CONFIG_PATH)
        return dict(cur)


# ---- derived paths -------------------------------------------------------

def _first_dir(*candidates):
    for c in candidates:
        if c and path.isdir(c):
            return path.normpath(c)
    return ""


def resolve_nunif(raw):
    """Accept either the repository root or the install root and return the
    repository root -- the folder holding `nunif\\` and `iw3\\`, which is what
    every tool expects as its working directory."""
    if not raw:
        return ""
    raw = path.normpath(path.expanduser(raw.strip().strip('"')))
    for cand in (raw, path.join(raw, "nunif")):
        if path.isdir(path.join(cand, "nunif")) and path.isdir(path.join(cand, "iw3")):
            return path.normpath(cand)
    return ""


def python_for(nunif_dir, override=""):
    """nunif's own python -- the one with torch in it. A plain `python` on PATH
    is almost never the right interpreter on a portable install."""
    if override and path.exists(override):
        return path.normpath(override)
    if nunif_dir:
        root = path.dirname(path.normpath(nunif_dir))
        for rel in (("python", "python.exe"), ("python", "bin", "python3"),
                    ("venv", "Scripts", "python.exe"), ("venv", "bin", "python")):
            p = path.join(root, *rel)
            if path.exists(p):
                return path.normpath(p)
    return ""


def detect_nunif():
    """Best-effort guesses, so most people never have to type a path."""
    here = GUI_DIR
    guesses = []
    # ...\<install>\New_Trainer\GUI  ->  ...\<install>\nunif
    for up in (2, 3, 4):
        base = here
        for _ in range(up):
            base = path.dirname(base)
        guesses.append(path.join(base, "nunif"))
        guesses.append(base)
    guesses += [
        r"C:\nunif-windows\nunif", r"D:\nunif-windows\nunif",
        r"E:\nunif-windows\nunif",
        path.expanduser("~/nunif"),
    ]
    for g in guesses:
        r = resolve_nunif(g)
        if r:
            return r
    return ""


def paths(cfg=None):
    cfg = cfg or load()
    nunif = resolve_nunif(cfg.get("nunif_dir", ""))
    out = {
        "gui_dir": GUI_DIR,
        "nunif_dir": nunif,
        "python_exe": python_for(nunif, cfg.get("python_exe", "")),
        "tools_dir": path.join(GUI_DIR, "tools"),
        "download_dir": path.normpath(cfg.get("download_dir") or path.join(DATA_DIR, "downloads")),
        "dataset_dir": path.normpath(cfg.get("dataset_dir") or path.join(DATA_DIR, "dataset")),
        "models_dir": path.normpath(cfg.get("models_dir") or path.join(DATA_DIR, "models")),
    }
    out["ready"] = bool(out["nunif_dir"] and out["python_exe"])
    return out


def ensure_data_dirs(p):
    for k in ("download_dir", "dataset_dir", "models_dir"):
        try:
            os.makedirs(p[k], exist_ok=True)
        except OSError:
            pass
