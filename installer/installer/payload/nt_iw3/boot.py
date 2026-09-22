r"""
Turn on whichever pieces were installed. Called once, on `import iw3`.

Everything is optional and independent, and each piece is wrapped in its own
try/except: a piece that fails prints one line to stderr and iw3 carries on
without it. Nothing here can stop iw3 from starting.

What is enabled comes from config.json next to this package, written by the
installer, so re-running the installer changes it without touching any code.
"""
from __future__ import annotations

import json
import os
import sys
from os import path


HERE = path.dirname(path.abspath(__file__))
ROOT = path.dirname(HERE)                      # <nunif>\nt_inpaint
CONFIG = path.join(ROOT, "config.json")

_done = False


def settings():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        # No config: behave as if everything present is wanted.
        return {"model": True, "mask_patch": True, "lowres": True, "window": True,
                "border": True}


def install(verbose=None):
    global _done
    if _done:
        return
    _done = True
    if os.environ.get("NT_INPAINT_DISABLE"):
        return
    cfg = settings()
    if verbose is None:
        verbose = bool(os.environ.get("NT_INPAINT_VERBOSE"))
    on = []

    if cfg.get("model", True):
        try:
            import nt_iw3.models        # noqa: F401  -- the import IS the registration
            on.append("architectures")
        except Exception as e:          # noqa: BLE001
            print(f"nt_iw3: the custom architectures are not available "
                  f"({type(e).__name__}: {e}); models trained with them will not load",
                  file=sys.stderr)

    if cfg.get("mask_patch", True):
        try:
            from nt_iw3 import iw3_mask
            if iw3_mask.install():
                on.append("mask patch")
        except Exception as e:          # noqa: BLE001
            print(f"nt_iw3: the mask patch could not be applied "
                  f"({type(e).__name__}: {e}); the custom model will look much "
                  f"like the stock one", file=sys.stderr)

    if cfg.get("lowres", True):
        try:
            from nt_iw3 import lowres
            if lowres.install():
                on.append("low-res inpainting")
        except Exception as e:          # noqa: BLE001
            print(f"nt_iw3: the low-res inpainting patch could not be applied "
                  f"({type(e).__name__}: {e}); Inpaint Max Width will behave the "
                  f"way it does in stock iw3", file=sys.stderr)

    # After lowres, so the window wrapper ends up outside it and the model
    # sees the frames it is asked for whichever path the call takes.
    if cfg.get("window", True):
        try:
            from nt_iw3 import window
            if window.install():
                on.append("window reuse")
        except Exception as e:          # noqa: BLE001
            print(f"nt_iw3: the window-reuse patch could not be applied "
                  f"({type(e).__name__}: {e}); the video model will keep running "
                  f"every frame twice, which is what stock iw3 does", file=sys.stderr)

    if cfg.get("border", False):
        try:
            from nt_iw3 import border
            if border.install():
                on.append("screen-edge fix")
        except Exception as e:          # noqa: BLE001
            print(f"nt_iw3: the screen-edge fix could not be applied "
                  f"({type(e).__name__}: {e}); 'Preserve Screen Border' will keep "
                  f"doing nothing with forward_inpaint, as in stock iw3", file=sys.stderr)

    if verbose and on:
        print("nt_iw3: " + ", ".join(on) + " active", file=sys.stderr)
    return on
