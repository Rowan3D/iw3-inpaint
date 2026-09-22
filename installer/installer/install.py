r"""
Install the inpainting extras into a plain iw3 / nunif install.

    install.bat                       (double-click; this is what it runs)
    python installer\install.py --nunif "C:\nunif-windows\nunif"
    python installer\install.py --uninstall

What gets installed, and where
------------------------------
Everything lands in ONE folder inside the nunif install:

    <nunif>\nt_inpaint\
        nt_iw3\            the code: model architectures, mask patch, low-res patch
        models\            the .pth that was shipped with this installer
        config.json        which pieces are switched on
        uninstall.bat      removes all of it

plus two small files in the Python that runs iw3:

    <python>\Lib\site-packages\nt_iw3.pth            one line: import nt_iw3_register
    <python>\Lib\site-packages\nt_iw3_register.py    points at the folder above

and one entry appended to iw3's model list:

    <nunif>\iw3\inpaint_models.yml                   (backed up first)

**No file belonging to nunif or iw3 is modified** apart from that model list,
which is a user config file iw3 ships empty. Both patches are applied at run
time by wrapping functions in memory, so an iw3 update cannot collide with them
and cannot leave a half-patched source tree behind.

Why the site-packages hook
--------------------------
A checkpoint stores only the NAME of its architecture; nunif looks that name up
in a registry filled by importing the module that defines it. iw3 imports its
own models and nothing else, so without the hook, loading the model fails with
"Unknown model name". A line starting with `import` inside a .pth file runs at
interpreter start, which is the one place that works for every iw3 launcher --
the GUI, the CLI and the desktop app alike. The hook itself imports nothing
heavy until something imports `iw3`.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from os import path

HERE = path.dirname(path.abspath(__file__))
PACK = path.dirname(HERE)
PAYLOAD = path.join(HERE, "payload")
MODELS_IN = path.join(PACK, "models")

FOLDER = "nt_inpaint"                 # <nunif>\nt_inpaint
HOOK_PY = "nt_iw3_register.py"
HOOK_PTH = "nt_iw3.pth"

# The trainer GUI used to write its own hook, which registers the (older)
# architectures from the trainer folder and applies the mask patch a second
# time. With this one installed it is redundant and it conflicts: whichever
# registers last wins, and the trainer's classes do not do window reuse.
LEGACY_HOOKS = ("ntrainer.pth", "ntrainer_iw3_register.py")

IS_WINDOWS = os.name == "nt"


# --------------------------------------------------------------------------
# finding things
# --------------------------------------------------------------------------
def is_nunif(folder):
    """The repository root: the folder that holds both nunif\\ and iw3\\."""
    return bool(folder) and path.isdir(path.join(folder, "nunif")) \
        and path.isdir(path.join(folder, "iw3"))


def resolve_nunif(raw):
    """Accept the repo root, the install root, or a path with quotes on it."""
    if not raw:
        return ""
    raw = path.normpath(path.expandvars(path.expanduser(raw.strip().strip('"').strip("'"))))
    for cand in (raw, path.join(raw, "nunif"), path.dirname(raw)):
        if is_nunif(cand):
            return path.normpath(cand)
    return ""


def detect_nunif():
    """Look where it usually is, so most people never type a path."""
    seen, out = set(), []

    def add(p):
        p = resolve_nunif(p)
        if p and p.lower() not in seen:
            seen.add(p.lower())
            out.append(p)

    add(os.environ.get("NUNIF_DIR", ""))
    # the installer unpacked inside the install
    here = PACK
    for _ in range(4):
        add(here)
        add(path.join(here, "nunif"))
        here = path.dirname(here)
    for base in ("C:\\", "D:\\", "E:\\", "F:\\", path.expanduser("~"),
                 path.join(path.expanduser("~"), "Desktop"),
                 path.join(path.expanduser("~"), "Downloads"),
                 path.join(path.expanduser("~"), "Documents")):
        if not path.isdir(base):
            continue
        for name in ("nunif-windows", "nunif", "iw3", "nunif_windows",
                     "nunif-windows-package"):
            add(path.join(base, name))
            add(path.join(base, name, "nunif"))
    return out


def python_for(nunif_dir):
    """The interpreter that has torch: nunif's own, not whatever is on PATH."""
    root = path.dirname(path.normpath(nunif_dir))
    names = (("python", "python.exe"), ("python", "bin", "python3"),
             ("venv", "Scripts", "python.exe"), ("venv", "bin", "python"),
             ("env", "Scripts", "python.exe"))
    for rel in names:
        p = path.join(root, *rel)
        if path.isfile(p):
            return path.normpath(p)
    return ""


def site_packages_of(python_exe):
    """Ask that interpreter where its site-packages is, rather than guessing."""
    code = ("import site,sys,json;"
            "d=[p for p in (site.getsitepackages() if hasattr(site,'getsitepackages') else [])"
            " if p.endswith(('site-packages','dist-packages'))];"
            "print(json.dumps(d or [p for p in sys.path if p.endswith("
            "('site-packages','dist-packages'))]))")
    try:
        out = subprocess.run([python_exe, "-c", code], capture_output=True,
                             text=True, timeout=120)
        dirs = json.loads(out.stdout.strip() or "[]")
    except Exception:                                               # noqa: BLE001
        return ""
    for d in dirs:
        if path.isdir(d) and os.access(d, os.W_OK):
            return path.normpath(d)
    return path.normpath(dirs[-1]) if dirs else ""


def yml_path(nunif_dir):
    """Where iw3 reads its model list from (NUNIF_HOME moves it)."""
    home = os.environ.get("NUNIF_HOME")
    if home:
        return path.join(path.expanduser(home), "iw3", "inpaint_models.yml")
    return path.join(nunif_dir, "iw3", "inpaint_models.yml")


def shipped_models():
    if not path.isdir(MODELS_IN):
        return []
    return sorted(path.join(MODELS_IN, f) for f in os.listdir(MODELS_IN)
                  if f.lower().endswith(".pth"))


# --------------------------------------------------------------------------
# talking to the person
# --------------------------------------------------------------------------
def say(*a):
    print(*a, flush=True)


def rule(title=""):
    say("\n" + (f"--- {title} " + "-" * max(0, 68 - len(title))) if title else "-" * 72)


def ask(prompt, default=""):
    try:
        got = input(prompt).strip()
    except EOFError:
        return default
    return got or default


def choose(prompt, options, default=1):
    """options: [(key, text), ...] -> the chosen key."""
    for i, (_k, text) in enumerate(options, 1):
        say(f"   [{i}] {text}")
    while True:
        got = ask(f"\n{prompt} [{default}]: ", str(default))
        if got.isdigit() and 1 <= int(got) <= len(options):
            return options[int(got) - 1][0]
        say("   Type one of the numbers above.")


def ask_nunif(preset=""):
    found = detect_nunif()
    if preset:
        p = resolve_nunif(preset)
        if p:
            return p
        say(f"\n{preset}\n   ...is not a nunif folder (no nunif\\ and iw3\\ inside it).")
    if found:
        rule("Found an iw3 install")
        opts = [(p, p) for p in found[:5]] + [("", "somewhere else - let me type it")]
        picked = choose("Which one?", opts, 1)
        if picked:
            return picked
    while True:
        rule("Where is iw3 installed?")
        say("   The folder that has  nunif\\  and  iw3\\  inside it, for example")
        say("   C:\\nunif-windows\\nunif")
        got = ask("\n   Path (or blank to give up): ")
        if not got:
            return ""
        p = resolve_nunif(got)
        if p:
            return p
        say("\n   That folder does not have nunif\\ and iw3\\ in it. Try the one above it?")


# --------------------------------------------------------------------------
# the work
# --------------------------------------------------------------------------
def copy_payload(dest):
    pkg_src = path.join(PAYLOAD, "nt_iw3")
    pkg_dst = path.join(dest, "nt_iw3")
    if path.isdir(pkg_dst):
        shutil.rmtree(pkg_dst, ignore_errors=True)
    shutil.copytree(pkg_src, pkg_dst,
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return pkg_dst


HOOK_TEMPLATE = '''r"""
Makes iw3 aware of the extras installed in __ROOT__.

Written by the iw3 inpaint installer. Two files do the job:

    nt_iw3.pth           ->  import nt_iw3_register
    nt_iw3_register.py   ->  this file

A line in a .pth file that starts with `import` runs when the interpreter
starts, so this is loaded in every process before anything imports iw3. It does
NOT import torch at startup -- that would slow down every python and pip
command in this install. Instead it installs a small import hook that does
nothing at all until something imports the `iw3` package, and only then loads
the extras.

Delete these two files to turn everything off. NT_INPAINT_DISABLE=1 does the
same thing temporarily.
"""
import os
import sys

TARGET = "iw3"
ROOT = os.environ.get("NT_INPAINT_DIR") or r"__ROOT__"


def _boot():
    try:
        if os.path.isdir(ROOT) and ROOT not in sys.path:
            sys.path.append(ROOT)
        import nt_iw3.boot
        nt_iw3.boot.install()
    except Exception as e:                                          # noqa: BLE001
        print(f"nt_iw3_register: the iw3 inpaint extras are NOT active "
              f"({type(e).__name__}: {e}). Expected them in {ROOT}",
              file=sys.stderr)


class _Loader:
    def __init__(self, inner):
        self._inner = inner

    def create_module(self, spec):
        return self._inner.create_module(spec)

    def exec_module(self, module):
        self._inner.exec_module(module)
        _boot()

    def __getattr__(self, name):
        return getattr(self._inner, name)


class _Finder:
    """Fires once, on `import iw3`, then removes itself from sys.meta_path."""

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
            return None
        try:
            sys.meta_path.remove(self)
        except ValueError:
            return None
        # Ask the rest of sys.meta_path, not PathFinder directly: another
        # extra (a separate installer) may have its own hook waiting for the
        # same import, and going straight to PathFinder would skip it.
        spec = None
        for finder in list(sys.meta_path):
            find = getattr(finder, "find_spec", None)
            if find is not None:
                spec = find(fullname, path, target)
                if spec is not None:
                    break
        if spec is None or spec.loader is None:
            return spec
        spec.loader = _Loader(spec.loader)
        return spec


def install():
    if os.environ.get("NT_INPAINT_DISABLE"):
        return False
    if TARGET in sys.modules:
        _boot()
        return True
    if not any(isinstance(f, _Finder) for f in sys.meta_path):
        sys.meta_path.insert(0, _Finder())
    return True


install()
'''


def write_hook(site_dir, root):
    with open(path.join(site_dir, HOOK_PY), "w", encoding="utf-8") as f:
        f.write(HOOK_TEMPLATE.replace("__ROOT__", root))
    with open(path.join(site_dir, HOOK_PTH), "w", encoding="utf-8") as f:
        f.write("import nt_iw3_register\n")


def remove_legacy_hooks(site_dir):
    """Take out the trainer's old hook, if present. Returns what was removed."""
    gone = []
    for f in LEGACY_HOOKS:
        fp = path.join(site_dir, f)
        if path.isfile(fp):
            try:
                os.remove(fp)
                gone.append(f)
            except OSError:
                pass
    return gone


def read_arch(python_exe, model_path):
    """(arch name, parameter count) straight out of the checkpoint."""
    code = ("import sys,json,torch;"
            "d=torch.load(sys.argv[1],map_location='cpu',weights_only=False);"
            "sd=d.get('state_dict') or {};"
            "print(json.dumps({'name':d.get('name',''),"
            "'params':int(sum(v.numel() for v in sd.values() if hasattr(v,'numel'))) }))")
    try:
        out = subprocess.run([python_exe, "-c", code, model_path],
                             capture_output=True, text=True, timeout=300)
        return json.loads(out.stdout.strip().splitlines()[-1])
    except Exception:                                               # noqa: BLE001
        return {"name": "", "params": 0}


def yml_entry(name, model_path, kind):
    return (f"\n# added by the iw3 inpaint installer\n"
            f"{name}:\n  {kind}: {model_path}\n")


def _strip_entries(text, names):
    """Remove `name:` and the indented lines under it, for each name given.

    Written by hand rather than with a YAML library because the file is the
    user's, often hand-edited and commented, and a load/dump round trip would
    throw all of that away.
    """
    out, skipping = [], False
    for line in text.splitlines(True):
        if any(line.startswith(f"{n}:") for n in names if n):
            skipping = True
            continue
        if skipping:
            # an entry is its key plus the indented lines under it
            if line.strip() == "" or line[:1] in (" ", "\t"):
                continue
            skipping = False
        if line.strip() == "# added by the iw3 inpaint installer":
            continue
        out.append(line)
    return "".join(out)


def _read_yml(fp):
    if not path.isfile(fp):
        os.makedirs(path.dirname(fp), exist_ok=True)
        return ""
    with open(fp, encoding="utf-8") as f:
        text = f.read()
    backup = fp + ".before-nt-inpaint"
    if not path.isfile(backup):
        shutil.copyfile(fp, backup)
    return text


def update_yml(fp, name, model_path, kind, drop=()):
    """Append (or replace) one entry, keeping the rest of the file untouched."""
    text = _strip_entries(_read_yml(fp), [name] + list(drop))
    text = text.rstrip("\n") + "\n" + yml_entry(name, model_path, kind)
    with open(fp, "w", encoding="utf-8") as f:
        f.write(text)


def drop_from_yml(fp, names):
    """Take our entries back out -- used when a re-install does not include the
    model. Leaving them would show an entry in iw3 that cannot be loaded."""
    names = [n for n in names if n]
    if not names or not path.isfile(fp):
        return False
    text = _read_yml(fp)
    new = _strip_entries(text, names).rstrip("\n") + "\n"
    if new == text:
        return False
    with open(fp, "w", encoding="utf-8") as f:
        f.write(new)
    return True


def previous_config(root):
    try:
        with open(path.join(root, "config.json"), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


UNINSTALL_BAT = """@echo off
title Remove the iw3 inpaint extras
echo.
echo   This removes:
echo     __ROOT__
echo     __SITE__\\nt_iw3.pth
echo     __SITE__\\nt_iw3_register.py
echo.
echo   Your iw3 install is not touched otherwise. The entry in
echo   inpaint_models.yml is left alone; delete it by hand if you want it gone
echo   (there is a copy of the original beside it, named .before-nt-inpaint).
echo.
set /p OK=  Type Y to remove:
if /i not "%OK%"=="Y" exit /b 0
del /q "__SITE__\\nt_iw3.pth" 2>nul
del /q "__SITE__\\nt_iw3_register.py" 2>nul
cd /d "%~dp0.."
rmdir /s /q "__ROOT__"
echo.
echo   Removed. Restart iw3.
pause
"""


def do_install(args):
    rule("iw3 inpainting extras")
    external = bool(args.external_model)
    models = [] if external else shipped_models()
    say(f"   installer : {PACK}")
    if external:
        say(f"   model     : managed by the trainer ({args.name or 'no entry named'})")
    else:
        say(f"   model     : {path.basename(models[0]) if models else '(none shipped)'}"
            + (f" and {len(models) - 1} more" if len(models) > 1 else ""))

    nunif = ask_nunif(args.nunif)
    if not nunif:
        say("\nNothing was changed.")
        return 1
    say(f"\n   iw3        : {nunif}")

    python_exe = args.python or python_for(nunif)
    if not python_exe:
        say("\n   Could not find the Python that runs iw3 (looked for python\\python.exe")
        say("   and venv\\ next to the nunif folder). Pass it with --python.")
        return 1
    say(f"   python     : {python_exe}")

    site_dir = site_packages_of(python_exe)
    if not site_dir or not path.isdir(site_dir):
        say("\n   That Python did not report a usable site-packages folder.")
        return 1
    say(f"   packages   : {site_dir}")

    # ---- what to install ----
    rule("What would you like to install?")
    say("   The low-res patch works with any inpainting model, including the one")
    say("   iw3 ships with. The trained model needs its own mask handling, which")
    say("   comes with it.\n")
    what = args.only or choose("Install", [
        ("both", "Everything  (recommended)"),
        ("lowres", "Only the low-res inpainting patch"
                   "  -  full-size output from a smaller, faster pass"),
        ("model", "Only the trained inpainting model"
                  "  -  better filling at high divergence, and faster on video"),
    ], 1)
    want_model = what in ("both", "model") and (bool(models) or external)
    want_lowres = what in ("both", "lowres")
    if what in ("both", "model") and not models and not external:
        say("\n   No .pth was shipped in this installer's models\\ folder, so there"
            "\n   is no model to install. Carrying on with the patch only.")

    # ---- optional: screen edges ----
    rule("Stretched screen edges (optional)")
    say("   At high divergence iw3 fills the left and right edges of each eye by")
    say("   smearing the outermost column of the picture - about 80 px at")
    say("   divergence 16 on a 1920-wide video. iw3's 'Preserve Screen Border'")
    say("   option is meant to prevent that, but forward_inpaint ignores it.")
    say("   This makes it work: tick the box and the depth fades to the screen")
    say("   plane at the edges instead, so there is nothing left to smear.")
    say("   Nothing changes while the box is unticked.\n")
    if args.border:
        want_border = args.border == "yes"
    else:
        want_border = choose("Install the screen-edge fix", [
            ("yes", "Yes  (recommended)"),
            ("no", "No"),
        ], 1) == "yes"

    model_path = models[0] if models else ""
    if want_model and not external and len(models) > 1:
        rule("Which model?")
        model_path = choose("Model", [(m, path.basename(m)) for m in models], 1)

    name = args.name or (path.splitext(path.basename(model_path))[0] if model_path else "")
    name = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
    kind = args.kind or "video"

    # ---- copy ----
    rule("Installing")
    root = path.join(nunif, FOLDER)
    previous = previous_config(root)
    os.makedirs(root, exist_ok=True)
    copy_payload(root)
    say(f"   code       -> {path.join(root, 'nt_iw3')}")

    installed_model = ""
    if want_model and external:
        # the trainer freezes the weights and writes its own yml entries (it
        # keeps several side by side, one per install name); all this install
        # has to do is make iw3 able to run them
        say(f"   model      :  managed by the trainer - '{name}' is checked below")
    elif want_model:
        info = read_arch(python_exe, model_path)
        arch = info.get("name", "")
        if arch and not arch.startswith("inpaint."):
            say(f"\n   {path.basename(model_path)} says its architecture is '{arch}',")
            say("   which is not an inpainting model. Stopping rather than writing")
            say("   an entry iw3 cannot load.")
            return 1
        kind = "image" if ("video" not in arch) else "video"
        models_dir = path.join(root, "models")
        os.makedirs(models_dir, exist_ok=True)
        installed_model = path.join(models_dir, path.basename(model_path))
        shutil.copyfile(model_path, installed_model)
        mb = os.path.getsize(installed_model) / 1e6
        say(f"   model      -> {installed_model}  ({mb:.0f} MB"
            + (f", {info['params'] / 1e6:.1f}M parameters" if info.get("params") else "")
            + ")")
        if arch:
            say(f"   arch       :  {arch}  ({kind} model)")

        fp = yml_path(nunif)
        # a previous install under a different name would otherwise be left
        # in the list pointing at a model this install no longer has
        update_yml(fp, name, installed_model, kind,
                   drop=[previous.get("entry", "")])
        say(f"   model list -> {fp}")
        say(f"   entry      :  {name}")
    elif previous.get("entry"):
        # installing without the model, over an install that had one: take the
        # old entry out, or iw3 offers a model whose architecture is no longer
        # registered and fails with "Unknown model name" when you pick it.
        if drop_from_yml(yml_path(nunif), [previous["entry"]]):
            say(f"   model list :  removed the earlier '{previous['entry']}' entry")
        stale = path.join(root, "models")
        if path.isdir(stale):
            shutil.rmtree(stale, ignore_errors=True)
            say("   model      :  removed the earlier model file")

    write_hook(site_dir, root)
    say(f"   hook       -> {path.join(site_dir, HOOK_PY)}")
    if want_model:
        gone = remove_legacy_hooks(site_dir)
        if gone:
            say(f"   old hook   :  removed the trainer's earlier hook ({', '.join(gone)}),")
            say("                 which this one replaces")

    with open(path.join(root, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"model": want_model, "mask_patch": want_model,
                   # only our architectures advertise WINDOW_CAPABLE, so window
                   # reuse has nothing to do without the model -- but it is
                   # harmless either way, and switching it off with the model
                   # keeps config.json an honest description of what is on.
                   "window": want_model,
                   "lowres": want_lowres,
                   "border": want_border,
                   # a trainer install owns its entries; keep whatever entry an
                   # earlier standalone install made, so a later re-run of the
                   # standalone installer can still clean it up
                   "entry": (previous.get("entry", "") if external else name)
                            if want_model else "",
                   "model_file": (previous.get("model_file", "") if external
                                  else installed_model),
                   "managed_by": "trainer" if external else "installer",
                   "installed": datetime.now().isoformat(timespec="seconds")},
                  f, indent=1)
    with open(path.join(root, "uninstall.bat"), "w", encoding="utf-8") as f:
        f.write(UNINSTALL_BAT.replace("__ROOT__", root).replace("__SITE__", site_dir))

    # ---- prove it ----
    rule("Checking it works")
    ok = verify(python_exe, nunif, name if want_model else "",
                want_model, want_lowres, want_window=want_model,
                want_border=want_border, kind=kind)

    rule("Done" if ok else "Done, with warnings")
    if want_lowres:
        say("   * Low-res inpainting is on. In iw3, set 'Inpaint Max Width' to")
        say("     1280 or 1920 with a *_inpaint method: the model runs at that")
        say("     width, the video stays full size.")
    if want_model:
        say(f"   * Pick '{name}' in iw3's 'Inpainting Model' box. It only appears")
        say("     when Method is forward_inpaint, mlbw_l2_inpaint or monobw_inpaint.")
        say("   * Window reuse is on with it: on video the model runs each frame")
        say("     once instead of twice, for about a third less time and VRAM.")
    if want_border:
        say("   * Screen-edge fix is on. Tick 'Preserve Screen Border' in iw3 to use")
        say("     it with forward_inpaint (it already works for the mlbw methods).")
    say(f"\n   To remove all of it: {path.join(root, 'uninstall.bat')}")
    say("\n   Close iw3 and open it again before you use any of this.")
    return 0 if ok else 2


VERIFY_CODE = r"""
import json, sys
result = {}
try:
    import iw3            # the hook fires here
    from nunif.models import get_model_names
    names = set(get_model_names())
    result["arch"] = any(n.startswith("inpaint.nt_") for n in names)
    import iw3.forward_warp as fw
    result["mask"] = "iw3_mask" in getattr(fw.fix_layered_holes, "__module__", "") \
        or "install" in getattr(fw.fix_layered_holes, "__qualname__", "")
    from iw3.base_inpaint import BaseImageInpaint, BaseVideoInpaint
    result["lowres"] = bool(getattr(BaseVideoInpaint.infer, "_nt_lowres", False)
                            and getattr(BaseImageInpaint.infer, "_nt_lowres", False))
    from iw3.base_inpaint import FrameQueue
    result["window"] = bool(getattr(BaseVideoInpaint._inpaint_single, "_nt_window", False)
                            and getattr(FrameQueue.add, "_nt_window", False))
    from iw3.forward_inpaint import ForwardInpaintImage, ForwardInpaintVideo
    result["border"] = bool(getattr(ForwardInpaintImage.apply_warp, "_nt_border", False)
                            and getattr(ForwardInpaintVideo.apply_warp, "_nt_border", False))
    entry = sys.argv[1] if len(sys.argv) > 1 else ""
    if entry:
        from iw3.inpaint_utils import INPAINT_MODELS, load_video_inpaint_model, \
            load_image_inpaint_model
        result["entry"] = entry in INPAINT_MODELS
        if result["entry"]:
            kind = sys.argv[2] if len(sys.argv) > 2 else "video"
            load = load_video_inpaint_model if kind == "video" else load_image_inpaint_model
            m = load(entry, device_id=-1)
            result["loads"] = True
            result["model_name"] = getattr(m, "name", "?")
except Exception as e:
    result["error"] = f"{type(e).__name__}: {e}"
print("NT_RESULT " + json.dumps(result))
"""


def verify(python_exe, nunif, entry, want_model, want_lowres, want_window=False,
           want_border=False, kind="video"):
    try:
        out = subprocess.run([python_exe, "-c", VERIFY_CODE, entry,
                              kind], cwd=nunif, capture_output=True,
                             text=True, timeout=900)
    except Exception as e:                                          # noqa: BLE001
        say(f"   could not run the check ({e})")
        return False
    line = [ln for ln in out.stdout.splitlines() if ln.startswith("NT_RESULT ")]
    if not line:
        say("   the check did not report back. Output was:")
        for ln in (out.stdout + out.stderr).strip().splitlines()[-12:]:
            say("     " + ln)
        return False
    got = json.loads(line[0][len("NT_RESULT "):])
    ok = True

    def check(cond, good, bad):
        nonlocal ok
        say(("   [ok]   " if cond else "   [!!]  ") + (good if cond else bad))
        ok = ok and cond

    if got.get("error"):
        say("   [!!]  " + got["error"])
        return False
    if want_model:
        check(got.get("arch"), "the custom architectures are registered",
              "the architectures did NOT register - the model will not load")
        check(got.get("mask"), "the mask patch is active",
              "the mask patch is NOT active - the model would look like the stock one")
        check(got.get("entry"), f"iw3 can see the '{entry}' entry",
              f"iw3 cannot see the '{entry}' entry in its model list")
        check(got.get("loads"), f"the model file loads ({got.get('model_name', '?')})",
              "the model file did NOT load")
    if want_window:
        check(got.get("window"), "window reuse is active (the video model runs "
                                 "each frame once, not twice)",
              "window reuse is NOT active - it would still work, just slower")
    if want_lowres:
        check(got.get("lowres"), "low-res inpainting is active",
              "the low-res patch is NOT active")
    if want_border:
        check(got.get("border"), "the screen-edge fix is active "
                                 "(Preserve Screen Border works with forward_inpaint)",
              "the screen-edge fix is NOT active")
    return ok


def do_check(args):
    """Verify an existing install against its own config.json, changing nothing."""
    nunif = resolve_nunif(args.nunif) if args.nunif else ask_nunif("")
    if not nunif:
        say("   no nunif folder given")
        return 1
    python_exe = args.python or python_for(nunif)
    root = path.join(nunif, FOLDER)
    cfg = previous_config(root)
    rule("Checking the iw3 install")
    say(f"   iw3        : {nunif}")
    if not cfg:
        say(f"   [!!]  nothing is installed yet ({root} has no config.json)")
        return 2
    site_dir = site_packages_of(python_exe) if python_exe else ""
    hook = bool(site_dir) and all(path.isfile(path.join(site_dir, f))
                                  for f in (HOOK_PY, HOOK_PTH))
    say(("   [ok]   " if hook else "   [!!]  ") + ("the site-packages hook is in place"
        if hook else f"the site-packages hook is missing from {site_dir or '?'}"))
    legacy = [f for f in LEGACY_HOOKS if site_dir and path.isfile(path.join(site_dir, f))]
    if legacy and cfg.get("model"):
        say("   [!!]  the trainer's old hook is still there too - install again to remove it")
    entry = args.name or cfg.get("entry", "")
    ok = verify(python_exe, nunif, entry if cfg.get("model") else "",
                bool(cfg.get("model")), bool(cfg.get("lowres")),
                want_window=bool(cfg.get("window")), want_border=bool(cfg.get("border")),
                kind=args.kind or "video")
    return 0 if (ok and hook and not (legacy and cfg.get("model"))) else 2


def do_uninstall(args):
    nunif = ask_nunif(args.nunif)
    if not nunif:
        return 1
    root = path.join(nunif, FOLDER)
    python_exe = args.python or python_for(nunif)
    site_dir = site_packages_of(python_exe) if python_exe else ""
    for f in (HOOK_PTH, HOOK_PY):
        fp = path.join(site_dir, f) if site_dir else ""
        if fp and path.isfile(fp):
            os.remove(fp)
            say(f"   removed {fp}")
    if path.isdir(root):
        shutil.rmtree(root, ignore_errors=True)
        say(f"   removed {root}")
    say("\n   The entry in inpaint_models.yml was left alone; a copy of the file")
    say("   as it was before is beside it, named .before-nt-inpaint.")
    say("\n   Restart iw3.")
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nunif", default="", help="path to the folder holding nunif\\ and iw3\\")
    ap.add_argument("--python", default="", help="the interpreter that runs iw3")
    ap.add_argument("--name", default="", help="name for the entry in iw3's model list")
    ap.add_argument("--only", choices=["both", "lowres", "model"], default="",
                    help="skip the menu and install this")
    ap.add_argument("--border", choices=["yes", "no"], default="",
                    help="skip the question and install (or not) the screen-edge fix")
    ap.add_argument("--external-model", action="store_true",
                    help="(used by the trainer) enable everything the model needs, "
                         "but leave the model file and the yml entry to the caller")
    ap.add_argument("--kind", choices=["video", "image"], default="",
                    help="with --external-model / --check: which loader to test the entry with")
    ap.add_argument("--check", action="store_true",
                    help="verify the current install and change nothing")
    ap.add_argument("--uninstall", action="store_true")
    args = ap.parse_args()
    try:
        if args.check:
            return do_check(args)
        return do_uninstall(args) if args.uninstall else do_install(args)
    except KeyboardInterrupt:
        say("\n\nStopped. Nothing further was changed.")
        return 1


if __name__ == "__main__":
    code = main()
    if IS_WINDOWS and sys.stdout.isatty():
        try:
            input("\nPress Enter to close. ")
        except EOFError:
            pass
    sys.exit(code)
