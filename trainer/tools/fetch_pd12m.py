r"""
Download high-resolution images from PD12M.

  https://huggingface.co/datasets/Spawning/PD12M   (CDLA-Permissive-2.0)
  12.4M public-domain / CC0 images from Europeana, Wikimedia Commons and the
  Smithsonian.

Why this source alongside 4KLSDB:

  * **No rate limit.** Spawning re-hosts every image on S3
    (pd12m.s3.us-west-2.amazonaws.com), so this is Wikimedia Commons content
    WITHOUT Wikimedia's throttling. Their own tutorial recommends parallel bulk
    download.
  * **Deep focus.** 4KLSDB was filtered by aesthetic score (Q-Align top 80%),
    which systematically favours shallow depth of field -- lovely photos, but
    a blurred background is the worst possible ground truth for an inpainter,
    because the only correct answer is mush. Museum, archival and
    documentation photography is overwhelmingly deep-focus.
  * **Clean licence.** CDLA-Permissive-2.0, commercial use fine, versus
    4KLSDB's LAION-derived provenance.

Metadata is 122 small parquet files carrying id/url/width/height/caption, so
resolution filtering happens before a single image byte is fetched.

Requires:  huggingface_hub, pyarrow  (run install_deps.bat)
Run via ..\fetch_images.bat <out> --source pd12m
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import sys
import threading
import time
from os import path


REPO = "Spawning/PD12M"

# PD12M is museum-sourced, so a large share is photographs OF flat things:
# paintings, prints, manuscripts, and specimens shot on a white sweep. Those are
# useless here -- a photographed painting has no parallax, so it generates no
# disocclusion mask, and a specimen on white has no background to reconstruct.
# Note this is NOT the same as excluding 2D art or CGI as subject matter; 4KLSDB
# supplies those as real scenes. This drops flat reproductions.
#
# Captions are auto-generated and name the medium directly ("The image shows a
# painting of...", "...against a white background"), so a keyword pass over the
# metadata filters them out before a single image is downloaded.
CAPTION_BLOCK = {
    "artwork": ["painting", "drawing", "engraving", "lithograph", "etching", "woodcut",
                "sketch", "illustration", "watercolor", "watercolour", "a print of",
                "portrait of", "coat of arms", "emblem", "heraldic", "tapestry",
                "stained glass", "mosaic", "fresco"],
    "document": ["manuscript", "document", "title page", "handwritten", "typewritten",
                 "newspaper", "certificate", "a map of", "map showing", "postage stamp",
                 "banknote", "a book", "book cover", "sheet music", "diagram", "blueprint",
                 "letter written", "printed text", "a page of", "pages of"],
    "flat_bg": ["white background", "plain background", "black background",
                "solid background", "neutral background", "gray background",
                "grey background", "blank background", "studio background"],
    # Archival scans: they pass every other test (real scene, real parallax) but
    # are soft, grainy and low-contrast, so they teach the model to reconstruct
    # mush. Spotted in the --survey output, where 3 of 5 "kept" samples were
    # "an old photo" / "an old black and white photo".
    # NB "black and white photo", not "black and white" -- the latter also
    # matches "a white and black goat".
    "archival": ["black and white photo", "black-and-white photo", "an old photo",
                 "old black and white", "sepia", "daguerreotype", "tintype",
                 "vintage photo", "historical photo", "faded photo"],
    "specimen": ["specimen", "fossil", "herbarium", "taxidermy", "preserved", "mounted on a pin",
                 "under a microscope", "a coin", "a medal", "an artifact", "a vase",
                 "a ceramic", "pottery", "porcelain"],
}

# Content balancing. PD12M skews heavily to nature (iNaturalist is 16% on its own,
# and Wikimedia Commons is full of landscape and wildlife), which leaves urban
# scenes and people under-represented -- exactly the content with the most
# parallax and the most man-made detail to reconstruct.
#
# Order matters: the first match wins, so the scarce categories are tested first.
# A photo of people in a city counts as "people", not "nature", because that is
# the quota we are trying to fill.
# Terms are bare nouns, not "a noun": matching is word-boundary anchored, so
# "car" cannot hit "carpet" or "scar", and writing "a car" would miss "a red car".
CATEGORIES = [
    ("people", ["man", "men", "woman", "women", "person", "people", "child", "children",
                "boy", "boys", "girl", "girls", "crowd", "wearing", "soldier", "soldiers",
                "worker", "workers", "dancer", "dancers", "musician", "musicians",
                "player", "players", "posing"]),
    ("urban", ["street", "streets", "city", "cityscape", "building", "buildings",
               "architecture", "town", "skyline", "bridge", "road", "plaza", "alley",
               "downtown", "urban", "house", "houses", "church", "cathedral", "tower",
               "castle", "station", "market", "shop", "shops", "storefront", "traffic",
               "wall", "walls", "fence", "sidewalk", "pavement", "rooftop", "roof",
               "village", "ruins", "monument", "courtyard", "harbor", "harbour"]),
    ("interior", ["room", "interior", "kitchen", "bedroom", "living room", "office",
                  "hall", "hallway", "staircase", "stairs", "indoor", "indoors", "table",
                  "chair", "chairs", "desk", "window", "windows", "door", "doorway",
                  "ceiling", "furniture", "shelf", "shelves"]),
    ("transport", ["car", "cars", "train", "trains", "aircraft", "airplane", "plane",
                   "boat", "boats", "ship", "ships", "bicycle", "bike", "motorcycle",
                   "bus", "tram", "truck", "locomotive", "railway", "railroad",
                   "helicopter", "vehicle", "vehicles"]),
    ("nature", ["tree", "trees", "forest", "mountain", "mountains", "river", "lake",
                "flower", "flowers", "plant", "plants", "bird", "birds", "insect",
                "animal", "animals", "beach", "ocean", "sea", "sky", "clouds", "grass",
                "leaf", "leaves", "landscape", "garden", "park", "waterfall", "rock",
                "rocks", "snow", "desert", "butterfly", "beetle", "moss", "fern",
                "wildlife", "valley", "canyon", "meadow", "field", "fields"]),
]
DEFAULT_MIX = {"people": 25, "urban": 30, "interior": 10, "transport": 10,
               "nature": 20, "other": 5}
ALL_CATS = [c for c, _ in CATEGORIES] + ["other"]


_RE_CACHE = {}


def _matches(text, terms):
    """Word-boundary match, not substring.

    Plain `in` is a trap here: "team" matches inside "steam locomotive", which
    filed a train under "people". Every term is anchored on word boundaries so
    "park" cannot match "parking" and "a coin" cannot match "a coincidence".
    """
    if not terms:
        return False
    # Keyed by identity for speed (this runs per row per category), and the cache
    # holds a reference to `terms` itself so the id can never be recycled onto a
    # different list while its regex is still cached.
    hit = _RE_CACHE.get(id(terms))
    if hit is None or hit[0] is not terms:
        rx = re.compile("|".join(r"\b" + re.escape(t.strip()) + r"\b" for t in terms))
        _RE_CACHE[id(terms)] = (terms, rx)
    else:
        rx = hit[1]
    return rx.search(text) is not None


def classify(caption):
    c = (caption or "").lower()
    for name, terms in CATEGORIES:
        if _matches(c, terms):
            return name
    return "other"


def parse_mix(spec):
    """`people=25,urban=30,...` -> absolute quotas later. Values are weights, so
    they do not have to add up to 100; they get normalised."""
    if not spec:
        return dict(DEFAULT_MIX)
    mix = {}
    for part in re.split(r"[,\s]+", spec.strip()):
        if not part:
            continue
        k, _, v = part.partition("=")
        k = k.strip().lower()
        if k not in ALL_CATS:
            raise SystemExit(f"--mix: unknown category {k!r}; choose from "
                             f"{', '.join(ALL_CATS)}")
        try:
            mix[k] = max(0.0, float(v))
        except ValueError:
            raise SystemExit(f"--mix: {part!r} is not category=number")
    if not mix or sum(mix.values()) <= 0:
        raise SystemExit("--mix: all weights are zero")
    return mix


def make_quota(mix, target):
    total = sum(mix.values())
    return {c: (target * mix.get(c, 0.0) / total) for c in ALL_CATS}


def select_balanced(cands, need, counts, quota):
    """Take `need` rows, always from whichever category is furthest below its
    quota. Categories already at quota are skipped entirely, which is what stops
    nature from eating the whole run -- PD12M has far more of it than anything
    else, so unbalanced selection is ~60% nature.

    `cands` must already be shuffled and carry `_cat`. Returns [] if every
    category with rows left is full; the caller treats that as a stall.
    """
    by = {c: [] for c in ALL_CATS}
    for r in cands:
        by[r["_cat"]].append(r)
    picked = []
    while len(picked) < need:
        best_c, best_room = None, 0.0
        for c in ALL_CATS:
            if not by[c]:
                continue
            room = quota[c] - counts.get(c, 0)
            if room > best_room:
                best_c, best_room = c, room
        if best_c is None:
            break
        picked.append(by[best_c].pop())
        counts[best_c] = counts.get(best_c, 0) + 1
    return picked


# Institutions whose collections are overwhelmingly one of the above.
SOURCE_BLOCK = ["art museum", "portrait gallery", "design museum", "NMNH",
                "museum of art", "national gallery", "library", "archive"]
STATE = "_fetch_state.json"
UA = "iw3-inpaint-trainer/1.0 (dataset collection for a personal ML project)"


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
    return {"done_meta": [], "saved": 0, "bytes": 0}


def save_state(out, st):
    with open(path.join(out, STATE), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--count", type=int, default=20000)
    p.add_argument("--min-width", type=int, default=1920)
    p.add_argument("--min-height", type=int, default=1080)
    p.add_argument("--max-disk-gb", type=float, default=0.0)
    p.add_argument("--workers", type=int, default=16,
                   help="parallel downloads. S3 has no meaningful rate limit, so this can "
                        "be high; 16-32 is reasonable on a home connection.")
    p.add_argument("--timeout", type=float, default=30.0)
    p.add_argument("--shuffle", action="store_true", default=True,
                   help="shuffle within each metadata file so you get a spread of sources "
                        "rather than one institution's collection in order")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--no-content-filter", action="store_true",
                   help="keep paintings, documents, specimens and white-background object "
                        "shots (they are excluded by default: flat reproductions have no "
                        "parallax, so they generate no mask)")
    p.add_argument("--keep", type=str, nargs="*", default=[],
                   choices=sorted(CAPTION_BLOCK),
                   help="re-allow specific blocked categories, e.g. --keep artwork")
    p.add_argument("--exclude-caption", type=str, nargs="*", default=[],
                   help="extra caption keywords to reject")
    p.add_argument("--require-caption", type=str, nargs="*", default=[],
                   help="only keep captions containing at least one of these")
    p.add_argument("--exclude-source", type=str, nargs="*", default=[],
                   help="extra source substrings to reject")
    p.add_argument("--include-source", type=str, nargs="*", default=[],
                   help="only keep these sources (substring match), e.g. iNaturalist")
    p.add_argument("--mix", type=str, default="",
                   help="target content mix, e.g. "
                        "\"people=25,urban=30,interior=10,transport=10,nature=20,other=5\". "
                        "Weights are relative, they need not sum to 100. Default: "
                        + ",".join(f"{k}={v}" for k, v in DEFAULT_MIX.items()))
    p.add_argument("--no-balance", action="store_true",
                   help="take whatever comes (PD12M is ~60%% nature that way)")
    p.add_argument("--survey", type=int, default=0, metavar="N",
                   help="read N metadata files, report what sources and content types are "
                        "in there, and exit. Costs no image bandwidth.")
    p.add_argument("--list", action="store_true")
    args = p.parse_args()

    mix = parse_mix(args.mix)
    blocked_terms = []
    if not args.no_content_filter:
        for cat, terms in CAPTION_BLOCK.items():
            if cat not in args.keep:
                blocked_terms += terms
    blocked_terms += [t.lower() for t in args.exclude_caption]
    blocked_srcs = ([] if args.no_content_filter else list(SOURCE_BLOCK)) \
        + list(args.exclude_source)
    blocked_srcs = [t.lower() for t in blocked_srcs]
    required = [t.lower() for t in args.require_caption]
    want_srcs = [t.lower() for t in args.include_source]

    def content_ok(row):
        src = (row.get("source") or "").lower()
        if want_srcs and not any(t in src for t in want_srcs):
            return False
        if any(t in src for t in blocked_srcs):
            return False
        cap = (row.get("caption") or "").lower()
        if required and not _matches(cap, required):
            return False
        return not _matches(cap, blocked_terms)

    try:
        from huggingface_hub import hf_hub_download, list_repo_files
        import pyarrow.parquet as pq
        import requests
    except ImportError as e:
        print(f"missing dependency: {e}\n", file=sys.stderr)
        print(f"nunif uses its own python, not the one on your PATH:", file=sys.stderr)
        print(f"  {sys.executable}\n", file=sys.stderr)
        print(f"run:  New_Trainer\\install_deps.bat\n", file=sys.stderr)
        return 1

    base = os.environ.get("NT_CWD") or os.getcwd()
    out = path.abspath(path.join(base, args.output))
    os.makedirs(out, exist_ok=True)
    meta_dir = path.join(out, "_meta")
    os.makedirs(meta_dir, exist_ok=True)

    print(f"repo   : {REPO}  (CDLA-Permissive-2.0, public domain / CC0)")
    print(f"output : {out}")
    metas = sorted(f for f in list_repo_files(REPO, repo_type="dataset")
                   if f.startswith("metadata/") and f.endswith(".parquet"))
    if not metas:
        print("no metadata files found", file=sys.stderr)
        return 1
    print(f"metadata: {len(metas)} parquet files")
    if args.list:
        for f in metas:
            print("  ", f)
        return 0

    if args.survey:
        import collections
        srcs = collections.Counter()
        cats = collections.Counter()
        content = collections.Counter()
        n_big = n_ok = n_rows = 0
        sample_ok, sample_bad = [], []
        for mf in metas[:args.survey]:
            local = hf_hub_download(REPO, mf, repo_type="dataset", local_dir=meta_dir)
            tbl = pq.read_table(local)
            names = set(tbl.schema.names)
            rows = tbl.select([c for c in ("width", "height", "caption", "source", "url")
                               if c in names]).to_pylist()
            try:
                os.remove(local)
            except OSError:
                pass
            n_rows += len(rows)
            for r in rows:
                if (r.get("width") or 0) < args.min_width or \
                        (r.get("height") or 0) < args.min_height:
                    continue
                n_big += 1
                srcs[(r.get("source") or "?")] += 1
                cap = (r.get("caption") or "").lower()
                hit = [c for c, terms in CAPTION_BLOCK.items()
                       if any(t in cap for t in terms)]
                for c in hit:
                    cats[c] += 1
                if content_ok(r):
                    n_ok += 1
                    content[classify(r.get("caption"))] += 1
                    if len(sample_ok) < 5:
                        sample_ok.append((r.get("source"), (r.get("caption") or "")[:95]))
                elif len(sample_bad) < 5:
                    sample_bad.append((r.get("source"), (r.get("caption") or "")[:95]))

        print(f"\nsurveyed {args.survey} metadata file(s), {n_rows} rows")
        print(f"  {n_big} at >= {args.min_width}x{args.min_height} "
              f"({n_big / max(n_rows, 1) * 100:.1f}%)")
        print(f"  {n_ok} survive the content filter "
              f"({n_ok / max(n_big, 1) * 100:.1f}% of those)\n")
        print(f"  content types present (of the {n_big} big enough):")
        for c, n in cats.most_common():
            print(f"    {n:7d} ({n / max(n_big, 1) * 100:5.1f}%)  {c}")
        mix_total = sum(mix.values())
        print(f"\n  scene mix of the {n_ok} kept (natural) vs --mix target:")
        for c in ALL_CATS:
            n = content.get(c, 0)
            want = mix.get(c, 0.0) / mix_total * 100
            print(f"    {n:7d} ({n / max(n_ok, 1) * 100:5.1f}%)  {c:10s} target {want:5.1f}%")
        print(f"\n  top sources:")
        for src, n in srcs.most_common(20):
            mark = "  <- blocked" if any(t in src.lower() for t in blocked_srcs) else ""
            print(f"    {n:7d} ({n / max(n_big, 1) * 100:5.1f}%)  {src[:55]}{mark}")
        print(f"\n  KEPT examples:")
        for src, cap in sample_ok:
            print(f"    [{(src or '?')[:28]:28s}] {cap}")
        print(f"\n  REJECTED examples:")
        for src, cap in sample_bad:
            print(f"    [{(src or '?')[:28]:28s}] {cap}")
        print(f"\n  tune with --keep / --exclude-caption / --include-source, or turn the "
              f"whole thing off with --no-content-filter")
        return 0

    st = load_state(out)
    done = set(st["done_meta"])
    target = args.count
    saved, nbytes = st["saved"], st["bytes"]
    cat_counts = {c: int(st.get("cats", {}).get(c, 0)) for c in ALL_CATS}
    balance = not args.no_balance
    quota = make_quota(mix, target)
    stalls = 0
    print(f"target : {target} images, >= {args.min_width}x{args.min_height}"
          + (f"  (resuming from {saved})" if saved else ""))
    if balance:
        mt = sum(mix.values())
        print("mix    : " + "  ".join(f"{c} {mix.get(c, 0) / mt * 100:.0f}%"
                                      for c in ALL_CATS if mix.get(c, 0) > 0))
    else:
        print("mix    : off (--no-balance)")
    print(f"workers: {args.workers} parallel downloads from S3 (no rate limit)\n")

    rng = random.Random(args.seed)
    sess = requests.Session()
    sess.headers["User-Agent"] = UA
    lock = threading.Lock()
    t0 = time.time()
    stop = threading.Event()
    counters = {"ok": saved, "fail": 0, "bytes": nbytes}
    n_content_rejected = 0

    def worker(q):
        s = requests.Session()
        s.headers["User-Agent"] = UA
        while not stop.is_set():
            try:
                url, name, cat = q.get(timeout=1)
            except queue.Empty:
                return
            try:
                r = s.get(url, timeout=args.timeout)
                if r.status_code == 200 and r.content:
                    dst = path.join(out, name)
                    with open(dst, "wb") as f:
                        f.write(r.content)
                    with lock:
                        counters["ok"] += 1
                        counters["bytes"] += len(r.content)
                        cat_counts[cat] = cat_counts.get(cat, 0) + 1
                        if counters["ok"] >= target:
                            stop.set()
                else:
                    with lock:
                        counters["fail"] += 1
            except Exception:                                    # noqa: BLE001
                with lock:
                    counters["fail"] += 1
            finally:
                q.task_done()

    try:
        for mi, mf in enumerate(metas):
            if counters["ok"] >= target or stop.is_set():
                break
            if mf in done:
                continue
            if args.max_disk_gb and counters["bytes"] / 1e9 >= args.max_disk_gb:
                print(f"reached --max-disk-gb {args.max_disk_gb}")
                break

            local = hf_hub_download(REPO, mf, repo_type="dataset", local_dir=meta_dir)
            tbl = pq.read_table(local)
            names = set(tbl.schema.names)
            cols = [c for c in ("url", "width", "height", "id", "caption", "source")
                    if c in names]
            rows = tbl.select(cols).to_pylist()
            try:
                os.remove(local)
            except OSError:
                pass

            big = [r for r in rows
                   if (r.get("width") or 0) >= args.min_width
                   and (r.get("height") or 0) >= args.min_height
                   and r.get("url")]
            cands = [r for r in big if content_ok(r)]
            n_content_rejected += len(big) - len(cands)
            if args.shuffle:
                rng.shuffle(cands)
            need = max(0, target - counters["ok"])
            n_filtered = len(cands)
            filtered = cands
            if balance:
                for r in filtered:
                    r["_cat"] = classify(r.get("caption"))
                # Quota is measured against images actually SAVED, not queued, so
                # download failures do not silently eat a category's share.
                cands = select_balanced(filtered, need, dict(cat_counts), quota)
                if n_filtered and not cands:
                    stalls += 1
                    if stalls >= 3:
                        # Every category with rows left has been at quota for three
                        # files running: either the mix is unreachable in this
                        # corpus or the target is nearly met. Either way, stop
                        # starving the run.
                        balance = False
                        print("    [balance] every category at quota for 3 files "
                              "running -- filling the remainder unbalanced")
                        cands = filtered[:need]
                else:
                    stalls = 0
            else:
                cands = filtered[:need]
            print(f"[meta {mi + 1}/{len(metas)}] {path.basename(mf)}: "
                  f"{len(rows)} rows -> {len(big)} at >= {args.min_width}px -> "
                  f"{n_filtered} after content filter -> {len(cands)} queued",
                  flush=True)
            if not cands:
                done.add(mf)
                continue

            q = queue.Queue(maxsize=args.workers * 4)
            threads = [threading.Thread(target=worker, args=(q,), daemon=True)
                       for _ in range(args.workers)]
            for t in threads:
                t.start()
            for r in cands:
                if stop.is_set():
                    break
                url = r["url"]
                ext = path.splitext(url.split("?")[0])[1].lower() or ".jpg"
                if len(ext) > 5:
                    ext = ".jpg"
                q.put((url, f"pd12m_{r.get('id') or abs(hash(url))}{ext}",
                       r.get("_cat") or classify(r.get("caption"))))
            q.join()
            stop_local = stop.is_set()
            for t in threads:
                t.join(timeout=2)

            done.add(mf)
            st.update(done_meta=sorted(done), saved=counters["ok"],
                      bytes=counters["bytes"], cats=dict(cat_counts))
            save_state(out, st)
            el = time.time() - t0
            rate = (counters["ok"] - saved) / el if el else 0
            print(f"    total {counters['ok']}/{target}, {human(counters['bytes'])} on disk, "
                  f"{counters['fail']} failed | {rate:.1f} img/s"
                  + (f" | eta {(target - counters['ok']) / rate / 60:.0f} min"
                     if rate > 0 and not stop_local else ""))
    except KeyboardInterrupt:
        stop.set()
        print("\ninterrupted -- progress saved, re-run to continue")
    finally:
        stop.set()
        st.update(done_meta=sorted(done), saved=counters["ok"],
                  bytes=counters["bytes"], cats=dict(cat_counts))
        save_state(out, st)

    el = time.time() - t0
    print(f"\n{counters['ok']} images, {human(counters['bytes'])} on disk, "
          f"{counters['fail']} failed, in {el / 60:.1f} min")
    got = sum(cat_counts.values())
    if got:
        mt = sum(mix.values())
        print("\nscene mix:")
        for c in ALL_CATS:
            n = cat_counts.get(c, 0)
            print(f"  {n:7d} ({n / got * 100:5.1f}%)  {c:10s} "
                  f"target {mix.get(c, 0.0) / mt * 100:5.1f}%")
        print("  (category is inferred from the caption; --mix to change, "
              "--no-balance to turn off)")
    if n_content_rejected:
        print(f"{n_content_rejected} rejected by the content filter "
              f"(paintings / documents / specimens / white-background object shots)")
    print(f"\nnext:\n  scan_dataset.bat \"{out}\"\n"
          f"  make_dataset.bat \"{out}\" <dataset-output> --min-detail 0.02")
    return 0


if __name__ == "__main__":
    sys.exit(main())
