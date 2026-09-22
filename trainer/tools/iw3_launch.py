r"""
Start an iw3 entry point with ntrainer's architectures registered.

  iw3-gui.bat            -- the normal iw3 GUI
  iw3-desktop-gui.bat    -- the desktop streaming GUI
  iw3-player-gui.bat     -- the player
  iw3-cli.bat ...        -- the command line

Why this exists
---------------
`load_model()` stores only the architecture *name* in a .pth and looks it up in
nunif's model registry (`nunif/models/register.py`). A name is registered when
the module defining it is imported, and iw3 imports `iw3.models` only -- it has
no idea `ntrainer` exists. Loading one of our checkpoints in a stock iw3 process
therefore fails with

    ValueError: Unknown model name: inpaint.nt_video_inpaint_v2_b

This launcher imports `ntrainer.models` first (that import *is* the
registration) and then runs the requested iw3 module exactly as `python -m`
would, so nothing under nunif/ is modified and the stock launchers still work
untouched for stock models.

  python tools/iw3_launch.py iw3.gui [args...]
  python tools/iw3_launch.py --check          # registration + config check only
"""
from __future__ import annotations

import os
import runpy
import sys
from os import path


_HERE = path.dirname(path.abspath(__file__))
_ROOT = path.dirname(_HERE)                      # ...\New_Trainer
_NUNIF = path.join(path.dirname(_ROOT), "nunif")  # ...\nunif-windows\nunif

OUR_MODELS = (
    "inpaint.nt_inpaint_v2", "inpaint.nt_inpaint_v2_s",
    "inpaint.nt_inpaint_v2_b", "inpaint.nt_inpaint_v2_l",
    "inpaint.nt_video_inpaint_v2", "inpaint.nt_video_inpaint_v2_s",
    "inpaint.nt_video_inpaint_v2_b", "inpaint.nt_video_inpaint_v2_l",
)


def extras_installed():
    """True when the iw3 extras (Install on page 5, or the standalone installer)
    are in this Python. Their hook registers the newer architectures -- the
    ones with window reuse -- and applies the mask patch on `import iw3`, so
    this launcher must not register the trainer's own copies on top."""
    if os.environ.get("NT_INPAINT_DISABLE"):
        return False
    import importlib.util
    try:
        return importlib.util.find_spec("nt_iw3_register") is not None
    except (ImportError, ValueError):
        return False


def register():
    """Make nunif's registry know our architectures."""
    for p in (_NUNIF, _ROOT):
        if path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    if extras_installed():
        import iw3  # noqa: F401  -- the installed hook does everything on this import
        from nunif.models import get_model_names
        names = set(get_model_names())
        missing = [n for n in OUR_MODELS if n not in names]
        if missing:
            raise RuntimeError(f"the installed iw3 extras did not register: {missing}")
        return sorted(n for n in OUR_MODELS if n in names)
    import ntrainer.models  # noqa: F401  -- the import is the registration
    from nunif.models import get_model_names
    names = set(get_model_names())
    missing = [n for n in OUR_MODELS if n not in names]
    if missing:
        raise RuntimeError(f"these architectures did not register: {missing}")
    # Registering the architectures is only half of it: without the mask patch
    # iw3 hands our models a mask a tenth the width of the one they were trained
    # on, and the output is indistinguishable from the stock model.
    try:
        import ntrainer.iw3_mask
        ntrainer.iw3_mask.install()
    except Exception as e:                                          # noqa: BLE001
        print(f"warning: the inpaint mask patch could not be applied "
              f"({type(e).__name__}: {e}); our models will run against iw3's own "
              f"mask and will look much like the stock model", file=sys.stderr)
    return sorted(n for n in OUR_MODELS if n in names)


def check():
    """Report what iw3 will see, without starting a GUI."""
    registered = register()
    print("registered architectures:")
    for n in registered:
        print(f"  {n}")

    # The site-packages hook is what makes the *stock* iw3 launchers work too.
    import site
    hook = None
    for d in site.getsitepackages() + [site.getusersitepackages()]:
        pth, mod = path.join(d, "ntrainer.pth"), path.join(d, "ntrainer_iw3_register.py")
        if path.exists(pth) and path.exists(mod):
            hook = d
            break
    print()
    if hook:
        print(f"auto-register hook: installed in {hook}")
        print("                    the stock iw3-gui.bat works too")
        try:
            text = open(path.join(hook, "ntrainer_iw3_register.py"), encoding="utf-8").read()
        except OSError:
            text = ""
        if "iw3_mask" not in text:
            print("                    OLD VERSION -- it registers the architectures but")
            print("                    not the mask patch, so our models will look like")
            print("                    the stock one. Press Install in the GUI again.")
    else:
        print("auto-register hook: NOT installed -- only the New_Trainer\\iw3-*.bat")
        print("                    launchers can load our models")

    if extras_installed():
        print("\niw3 extras        : installed (nt_iw3) -- mask patch, window reuse and")
        print("                    the rest come from there; see page 5 of the GUI")
    else:
        import ntrainer.iw3_mask as _mask
        print(f"\nmask patch        : {'active' if _mask.install() else 'disabled'}"
              f"  (stretch threshold {_mask.MaskParams.max_grad}, "
              f"min run {_mask.MaskParams.min_run}px, grow {_mask.MaskParams.grow}px)")

    from iw3.inpaint_utils import INPAINT_MODELS, INPAINT_CONFIG_FILE
    print(f"\ninpaint_models.yml: {INPAINT_CONFIG_FILE}")
    print(f"{'name':<28} {'kind':<6} target")
    ok = True
    for name, entry in INPAINT_MODELS.items():
        for kind in ("image", "video"):
            target = entry[kind]
            if target.lower().startswith(("http://", "https://")):
                mark = "url"
            elif path.exists(target):
                mark = f"{path.getsize(target) / 1e6:.0f} MB"
            else:
                mark, ok = "MISSING", False
            print(f"{name:<28} {kind:<6} [{mark}] {target}")
    print()
    if not ok:
        print("one or more local paths above do not exist -- fix inpaint_models.yml")
        return 1
    print("all good. Pick the model by name in the GUI's 'Inpainting Model' box.")
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    if argv[0] == "--check":
        return check()

    module = argv[0]
    register()
    # iw3 resolves its own relative paths (pretrained_models, hub cache) against
    # the repository root, so behave like the stock launchers' `pushd %NUNIF_DIR%`.
    if path.isdir(_NUNIF):
        os.chdir(_NUNIF)
    sys.argv = [module] + argv[1:]
    runpy.run_module(module, run_name="__main__", alter_sys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
