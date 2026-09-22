r"""
Download a chosen number of native-4K images from 4KLSDB.

  https://huggingface.co/datasets/SingleBicycle/4KLSDB   (CC-BY-4.0)
  129,484 images, every one >= 3840px on at least one side, curated across
  natural landscapes, urban scenes, people, food, CGI and artwork, and already
  filtered with Laplacian/Sobel edge detection to drop blurry or flat content.

The full repo is 2.67 TB, but most of that is low-resolution copies we do not
want.  Three splits -- train_x4 / x8 / x16 -- contain the SAME 129,484 HR
images, each paired with a different LR copy.  x16's LR is 1/256 the pixels, so
it is the cheapest route to the HR data.

This walks the shards one at a time: download a shard, write out its HR images,
delete the shard, repeat until `--count` is reached.  Peak disk is therefore one
shard (~4 GB) plus the images kept, not 2.67 TB.  The `hr` column holds the
original encoded bytes, so images are written verbatim with no re-encode.

Resumable: finished shards are recorded in _fetch_state.json, so re-running
continues where it stopped.

Requires:  pip install huggingface_hub pyarrow
Run via ..\fetch_images.bat
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from os import path


REPO = "SingleBicycle/4KLSDB"
STATE = "_fetch_state.json"
MAGIC = [(b"\xff\xd8\xff", ".jpg"), (b"\x89PNG\r\n\x1a\n", ".png"),
         (b"RIFF", ".webp"), (b"II*\x00", ".tif"), (b"MM\x00*", ".tif")]


def ext_of(buf):
    for sig, e in MAGIC:
        if buf[:len(sig)] == sig:
            return e
    return ".bin"


def human(n):
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{n:.1f} PB"


def load_state(out):
    p = path.join(out, STATE)
    if path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:                                        # noqa: BLE001
            pass
    return {"done_shards": [], "saved": 0, "bytes": 0}


def save_state(out, st):
    with open(path.join(out, STATE), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output", type=str, required=True, help="where to write the images")
    p.add_argument("--count", type=int, default=20000,
                   help="how many images to end up with (0 = the whole 129,484)")
    p.add_argument("--split", type=str, default="train_x16",
                   choices=["train_x16", "train_x8", "train_x4"],
                   help="all three hold the same HR images; x16 carries the least "
                        "low-res baggage, so it downloads fastest")
    p.add_argument("--min-width", type=int, default=1920,
                   help="skip images narrower than this (our warp needs >=1920 to reach "
                        "div-16 hole widths)")
    p.add_argument("--min-quality", type=float, default=0.0,
                   help="4KLSDB's own quality_score, 0 to disable")
    p.add_argument("--min-aesthetic", type=float, default=0.0,
                   help="4KLSDB's own aesthetic_score, 0 to disable")
    p.add_argument("--max-disk-gb", type=float, default=0.0,
                   help="stop once the output folder reaches this size (0 = no limit)")
    p.add_argument("--tmp", type=str, default=None,
                   help="scratch dir for shards (default: <output>/_tmp). Needs ~5 GB.")
    p.add_argument("--keep-shards", action="store_true",
                   help="don't delete each shard after extracting (needs 2.67 TB for all)")
    p.add_argument("--shard-limit", type=int, default=0,
                   help="stop after this many shards (0 = until --count is met)")
    p.add_argument("--list", action="store_true", help="list the shards and exit")
    args = p.parse_args()

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import pyarrow.parquet as pq
    except ImportError as e:
        # nunif ships its own embeddable python. A bare `pip` in a normal prompt
        # installs into the SYSTEM python instead and reports success, so name
        # the interpreter explicitly.
        print(f"missing dependency: {e}\n", file=sys.stderr)
        print(f"nunif uses its own python, not the one on your PATH:", file=sys.stderr)
        print(f"  {sys.executable}\n", file=sys.stderr)
        print(f"install into THAT one, either by running:", file=sys.stderr)
        print(f"  New_Trainer\\install_deps.bat\n", file=sys.stderr)
        print(f"or directly:", file=sys.stderr)
        print(f'  "{sys.executable}" -m pip install huggingface_hub pyarrow\n',
              file=sys.stderr)
        return 1

    base = os.environ.get("NT_CWD") or os.getcwd()
    out = path.abspath(path.join(base, args.output))
    os.makedirs(out, exist_ok=True)
    tmp = path.abspath(path.join(base, args.tmp)) if args.tmp else path.join(out, "_tmp")
    os.makedirs(tmp, exist_ok=True)

    print(f"repo   : {REPO}  (CC-BY-4.0)")
    print(f"output : {out}")
    files = [f for f in list_repo_files(REPO, repo_type="dataset")
             if f.startswith(f"data/{args.split}-") and f.endswith(".parquet")]
    files.sort()
    if not files:
        print(f"no shards found for split {args.split}", file=sys.stderr)
        return 1
    print(f"split  : {args.split}  ({len(files)} shards, ~630 images each)")
    if args.list:
        for f in files:
            print("  ", f)
        return 0

    st = load_state(out)
    done = set(st["done_shards"])
    target = args.count or 129484
    print(f"target : {target} images"
          + (f"  (resuming: {st['saved']} already saved, {len(done)} shards done)"
             if st["saved"] else ""))
    print()

    t0 = time.time()
    saved = st["saved"]
    nbytes = st["bytes"]
    skipped = 0
    shards_this_run = 0

    try:
        for fi, fn in enumerate(files):
            if saved >= target:
                break
            if fn in done:
                continue
            if args.shard_limit and shards_this_run >= args.shard_limit:
                print(f"reached --shard-limit {args.shard_limit}")
                break
            if args.max_disk_gb and nbytes / 1e9 >= args.max_disk_gb:
                print(f"reached --max-disk-gb {args.max_disk_gb}")
                break

            print(f"[shard {fi + 1}/{len(files)}] {path.basename(fn)} downloading...",
                  flush=True)
            t_dl = time.time()
            local = hf_hub_download(REPO, fn, repo_type="dataset",
                                    local_dir=tmp, cache_dir=path.join(tmp, ".cache"))
            dl_bytes = path.getsize(local)
            dl_secs = time.time() - t_dl

            cols = ["hr", "file_name", "width", "height", "quality_score", "aesthetic_score"]
            pf = pq.ParquetFile(local)
            have = set(pf.schema_arrow.names)
            cols = [c for c in cols if c in have]
            kept = 0
            for rg in range(pf.num_row_groups):          # row groups keep RAM bounded
                tbl = pf.read_row_group(rg, columns=cols).to_pylist()
                for row in tbl:
                    if saved >= target:
                        break
                    w = row.get("width") or 0
                    if args.min_width and w and w < args.min_width:
                        skipped += 1
                        continue
                    if args.min_quality and (row.get("quality_score") or 0) < args.min_quality:
                        skipped += 1
                        continue
                    if args.min_aesthetic and (row.get("aesthetic_score") or 0) < args.min_aesthetic:
                        skipped += 1
                        continue
                    img = row.get("hr")
                    buf = img.get("bytes") if isinstance(img, dict) else None
                    if not buf:
                        skipped += 1
                        continue
                    stem = path.splitext(path.basename(
                        (row.get("file_name") or f"{fi:05d}_{saved:07d}")))[0]
                    dst = path.join(out, f"{stem}{ext_of(buf)}")
                    if path.exists(dst):
                        dst = path.join(out, f"{stem}_{saved:07d}{ext_of(buf)}")
                    with open(dst, "wb") as f:
                        f.write(buf)                     # original bytes, no re-encode
                    saved += 1
                    kept += 1
                    nbytes += len(buf)
                if saved >= target:
                    break

            if not args.keep_shards:
                try:
                    os.remove(local)
                except OSError:
                    pass
                shutil.rmtree(path.join(tmp, ".cache"), ignore_errors=True)

            done.add(fn)
            shards_this_run += 1
            st.update(done_shards=sorted(done), saved=saved, bytes=nbytes)
            save_state(out, st)
            el = time.time() - t0
            rate = saved / el if el else 0
            print(f"    +{kept} images ({human(dl_bytes)} shard in {dl_secs:.0f}s) | "
                  f"total {saved}/{target}, {human(nbytes)} on disk | "
                  f"{rate:.1f} img/s"
                  + (f" | eta {(target - saved) / rate / 60:.0f} min" if rate else ""))
    except KeyboardInterrupt:
        print("\ninterrupted -- progress saved, re-run to continue")
    finally:
        if not args.keep_shards:
            shutil.rmtree(tmp, ignore_errors=True)
        st.update(done_shards=sorted(done), saved=saved, bytes=nbytes)
        save_state(out, st)

    el = time.time() - t0
    print(f"\n{saved} images, {human(nbytes)} on disk"
          + (f", {skipped} skipped by filters" if skipped else "")
          + f", in {el / 60:.1f} min")
    print(f"\nnext:\n  scan_dataset.bat \"{out}\"\n"
          f"  make_dataset.bat \"{out}\" <dataset-output-folder>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
