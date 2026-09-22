r"""
Give nunif's Python what torch.compile needs, or say exactly why it cannot.

  install_compile_support.py --check     report, change nothing
  install_compile_support.py             download and install

torch.compile goes through triton, and triton compiles a small C helper the
first time it talks to CUDA. That needs two things every normal Python install
has and the **embeddable** build does not:

    <python>\Include\Python.h
    <python>\libs\python3XX.lib

nunif ships the embeddable build, so on a stock install compiling fails with
"include file 'Python.h' not found" and, before that, a warning about
python3XX.lib. Neither is a PyPI package, so the dependency step cannot fetch
them: they come from CPython itself.

The official NuGet package for CPython carries both, for the exact patch
version, and is the least invasive source -- nothing is installed, nothing is
registered, two folders are unpacked beside the interpreter. Delete them to
undo it.

This script must run with nunif's own Python, so the version and destination
are read from the interpreter itself rather than guessed.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import sysconfig
import tempfile
import urllib.error
import urllib.request
import zipfile
from os import path


TIMEOUT = 120
UA = "nunif-inpaint-trainer"


def targets():
    """(<python dir>, the header we need, the import library we need)."""
    root = path.dirname(path.abspath(sys.executable))
    tag = f"python{sys.version_info.major}{sys.version_info.minor}"
    return root, path.join(root, "Include", "Python.h"), path.join(root, "libs", f"{tag}.lib")


def status():
    root, header, lib = targets()
    return {
        "python": sys.executable,
        "version": ".".join(str(v) for v in sys.version_info[:3]),
        "root": root,
        "header": header, "has_header": path.isfile(header),
        "lib": lib, "has_lib": path.isfile(lib),
        "embeddable": bool([f for f in os.listdir(root) if f.endswith("._pth")]),
    }


def report(st):
    print(f"python     : {st['python']}")
    print(f"version    : {st['version']}" + ("  (embeddable build)" if st["embeddable"] else ""))
    print(f"Python.h   : {'found' if st['has_header'] else 'MISSING'}  {st['header']}")
    print(f"import lib : {'found' if st['has_lib'] else 'MISSING'}  {st['lib']}")
    if st["has_header"] and st["has_lib"]:
        print("\ntorch.compile has what it needs on this install.")
    else:
        print("\ntorch.compile cannot build its CUDA helper without these.")


def candidates(version):
    return [
        f"https://api.nuget.org/v3-flatcontainer/python/{version}/python.{version}.nupkg",
        f"https://globalcdn.nuget.org/packages/python.{version}.nupkg",
    ]


def download(version, dest):
    last = None
    for url in candidates(version):
        print(f"fetching   : {url}")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r, open(dest, "wb") as f:
                shutil.copyfileobj(r, f)
            size = path.getsize(dest)
            if size < 1024 * 1024:
                raise OSError(f"that is only {size} bytes, not a CPython package")
            print(f"downloaded : {size / 1e6:.1f} MB")
            return True
        except (urllib.error.URLError, OSError) as e:
            print(f"             failed ({e})")
            last = e
    print("\nCould not download it. The package is CPython's own, published on nuget.org;\n"
          "if this machine cannot reach that host, install Python "
          f"{version} from python.org\n"
          "and copy its Include\\ and libs\\ folders into the folder printed above.")
    return False


def extract(archive, root, force=False):
    """Find the header and the import library wherever they live in the archive.

    Deliberately not hard-coded to `tools/include` and `tools/libs`: searching
    for the two files by name means a change in how the package is laid out
    shows up as "not found in the archive" rather than as a silent no-op.
    """
    tag = f"python{sys.version_info.major}{sys.version_info.minor}"
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        header = next((n for n in names if n.lower().endswith("/python.h")), None)
        lib = next((n for n in names if n.lower().endswith(f"/{tag}.lib")), None)
        if not header or not lib:
            print(f"error      : the archive has no {'Python.h' if not header else tag + '.lib'}; "
                  f"it is not the package this expects")
            return False
        inc_root = header[:header.lower().rindex("/python.h")]
        lib_root = lib[:lib.lower().rindex(f"/{tag}.lib")]
        print(f"archive    : headers under {inc_root}/, libraries under {lib_root}/")

        done = 0
        for name in names:
            if name.endswith("/"):
                continue
            if name.startswith(inc_root + "/"):
                rel = name[len(inc_root) + 1:]
                out = path.join(root, "Include", *rel.split("/"))
            elif name.startswith(lib_root + "/"):
                rel = name[len(lib_root) + 1:]
                out = path.join(root, "libs", *rel.split("/"))
            else:
                continue
            if path.exists(out) and not force:
                continue
            os.makedirs(path.dirname(out), exist_ok=True)
            with z.open(name) as src, open(out, "wb") as dst:
                shutil.copyfileobj(src, dst)
            done += 1
    print(f"installed  : {done} file(s)")
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--check", action="store_true", help="report and change nothing")
    p.add_argument("--force", action="store_true", help="overwrite files that are already there")
    args = p.parse_args()

    st = status()
    report(st)
    if args.check:
        return 0
    if st["has_header"] and st["has_lib"] and not args.force:
        print("\nNothing to do.")
        return 0
    if os.name != "nt":
        print("\nThis only applies to the Windows builds of Python; nothing to do here.")
        return 0

    root = st["root"]
    if not os.access(root, os.W_OK):
        print(f"\nerror      : cannot write to {root}")
        return 1

    print()
    with tempfile.TemporaryDirectory() as tmp:
        pkg = path.join(tmp, "python.nupkg")
        if not download(st["version"], pkg):
            return 1
        if not extract(pkg, root, force=args.force):
            return 1

    after = status()
    print()
    report(after)
    if after["has_header"] and after["has_lib"]:
        print("\nTick 'Compile the model' and start a short run. The trainer tests the "
              "compiler\nat startup now, so if anything is still wrong it says so and trains "
              "without it.")
        return 0
    print("\nSomething is still missing -- see above.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
