r"""
Download royalty-free stock video, balanced across the same scene categories as
the image set.

  fetch_videos.bat <out> --count 400
  fetch_videos.bat <out> --count 400 --source pexels
  fetch_videos.bat <out> --probe                 check the keys work, print one raw result

Sources (both need a free API key, see --pexels-key / --pixabay-key):

  pexels    https://api.pexels.com/videos/search   200 req/hour, 20k/month.
            Has genuine 4K. Licence: Pexels licence, free for commercial use.
  pixabay   https://pixabay.com/api/videos/        ~100 req/minute.
            Licence: Pixabay content licence, free for commercial use.

Neither is rate-limited on the *file* downloads, only on the search calls, and
one search call returns up to 80 (Pexels) or 200 (Pixabay) results -- so the
metadata is never the bottleneck.

Balancing works differently from the image downloader. There the corpus was
fixed and captions were classified; here the category IS the query, so the mix
is controlled by how many results are requested per category. Same default mix
as `fetch_pd12m.py`, for the same reason: left alone, stock video skews to
nature and abstract b-roll, and it is urban scenes and people that carry the
parallax and the man-made detail worth reconstructing.

A warning worth reading: stock video is heavily shallow depth-of-field, slow
motion and drone. Shallow DoF is the same problem 4KLSDB had for the image set --
an out-of-focus background under the mask can only teach mush. Drone and dolly
shots, on the other hand, are the best possible material here, because they are
exactly parallax. `make_video_dataset.bat --scan` and the hole-width stats at the
end of a run are how you check what you actually got.

NOTE: this GUI copy differs from New_Trainer\tools\fetch_videos.py in one
place -- see MIN_VIDEO_BYTES and the `aborted` flag in the download worker. The
original promotes a half-written .part file to its final name when the run stops
mid-download, which leaves 0-byte and truncated .mp4 files behind.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import sys
import threading
import time
from os import path

_TOOLS = path.dirname(path.abspath(__file__))
if _TOOLS not in sys.path:
    sys.path.insert(0, _TOOLS)

from fetch_pd12m import DEFAULT_MIX, ALL_CATS, human, parse_mix   # noqa: E402


STATE = "_fetch_videos_state.json"
# Any real clip is far bigger than this; anything under it is a failed or
# interrupted transfer, whatever the server said.
MIN_VIDEO_BYTES = 64 * 1024

UA = "iw3-inpaint-trainer/1.0 (dataset collection for a personal ML project)"

# The category IS the query here. Terms are ordered so the first few are the
# highest-parallax framings -- a street at eye level has far more disocclusion
# than a flat-on facade.
QUERIES = {
    "people": [
        "people walking street", "crowd walking", "pedestrians crossing", "man walking city",
        "woman walking street", "people market", "children playing outside",
        "construction worker", "street musician", "people cafe", "commuters station",
    ],
    "urban": [
        "walking through city street", "city street traffic", "old town alley",
        "downtown buildings", "city at night street", "european street", "narrow alley",
        "market street", "bridge city", "rooftop city view", "village street",
        "train station interior", "parking garage", "industrial area", "construction site",
    ],
    "interior": [
        "walking through house", "living room interior", "kitchen interior",
        "office interior", "cafe interior", "museum walk", "library interior",
        "staircase interior", "warehouse interior", "shop interior",
    ],
    "transport": [
        "driving pov street", "train passing", "car driving city", "bicycle riding street",
        "bus city", "tram street", "boat harbor", "motorcycle riding", "airport airplane",
        "truck highway",
    ],
    "nature": [
        "walking forest path", "forest trees", "mountain landscape", "river flowing",
        "beach waves", "park trees", "garden flowers", "waterfall", "desert landscape",
        "snow forest",
    ],
    "other": [
        "drone flying over", "timelapse city", "close up hands", "workshop tools",
        "food preparation", "sports action",
    ],
}


def load_state(out):
    p = path.join(out, STATE)
    if path.exists(p):
        try:
            with open(p, encoding="utf-8") as f:
                return json.load(f)
        except Exception:                                        # noqa: BLE001
            pass
    return {"seen": [], "saved": 0, "bytes": 0, "cats": {}}


def save_state(out, st):
    with open(path.join(out, STATE), "w", encoding="utf-8") as f:
        json.dump(st, f, indent=2)


def best_file(entry, source, min_width, max_width):
    """Largest variant at or under max_width, and at least min_width.

    Written to tolerate schema drift: Pexels gives a list of dicts, Pixabay a
    dict of named sizes, and both have gained keys over time. Anything with a
    usable width and a link is a candidate.
    """
    cands = []
    if source == "pexels":
        for f in entry.get("video_files") or []:
            link = f.get("link")
            w = f.get("width") or 0
            if link and w:
                cands.append((int(w), int(f.get("height") or 0), link, f.get("fps")))
    else:
        vids = entry.get("videos") or {}
        items = vids.values() if isinstance(vids, dict) else vids
        for f in items:
            if not isinstance(f, dict):
                continue
            link = f.get("url")
            w = f.get("width") or 0
            if link and w:
                cands.append((int(w), int(f.get("height") or 0), link, None))
    cands = [c for c in cands if c[0] >= min_width and (not max_width or c[0] <= max_width)]
    if not cands:
        return None
    return max(cands, key=lambda c: c[0])


def pexels_size(min_width):
    """Pexels `size` is a MINIMUM (large=4K, medium=FullHD, small=HD), so it has to
    track --min-width: pinning "medium" would keep asking for >=FullHD even when
    the caller wants 4K, and would over-filter when they want less."""
    if min_width >= 3840:
        return "large"
    if min_width >= 1920:
        return "medium"
    return "small"


def search(session, source, key, query, page, per_page, min_width, timeout):
    if source == "pexels":
        r = session.get("https://api.pexels.com/videos/search",
                        params={"query": query, "per_page": per_page, "page": page,
                                "orientation": "landscape", "size": pexels_size(min_width)},
                        headers={"Authorization": key}, timeout=timeout)
        if r.status_code == 429:
            return None, "rate limited"
        r.raise_for_status()
        return r.json().get("videos") or [], None
    else:
        r = session.get("https://pixabay.com/api/videos/",
                        params={"key": key, "q": query, "per_page": per_page, "page": page,
                                "min_width": min_width, "safesearch": "true",
                                "order": "popular"},
                        timeout=timeout)
        if r.status_code == 429:
            return None, "rate limited"
        r.raise_for_status()
        return r.json().get("hits") or [], None


def flag_reject(entry, source, allow_ai=False, allow_optout=False, allow_low=False):
    """Per-asset flags Pixabay publishes. Returns a reason string, or None to keep.

    `noAiTraining` is the creator saying they do not want the asset used to train
    models. That is exactly what this corpus is for, so it is honoured by default.
    `isAiGenerated` is filtered for a different reason: generated footage has
    invented geometry and its own temporal artefacts, which is the opposite of
    the parallax ground truth this dataset exists to provide.
    """
    if source != "pixabay":
        return None
    if entry.get("noAiTraining") and not allow_optout:
        return "creator opted out of AI training"
    if entry.get("isAiGenerated") and not allow_ai:
        return "AI generated"
    if entry.get("isLowQuality") and not allow_low:
        return "flagged low quality"
    return None


def entry_duration(entry):
    try:
        return float(entry.get("duration") or 0)
    except (TypeError, ValueError):
        return 0.0


def main():
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--output", type=str, required=True)
    p.add_argument("--count", type=int, default=400, help="videos to download")
    p.add_argument("--source", type=str, nargs="*", default=["pexels", "pixabay"],
                   choices=["pexels", "pixabay"])
    p.add_argument("--pexels-key", type=str, default=os.environ.get("PEXELS_API_KEY", ""))
    p.add_argument("--pixabay-key", type=str, default=os.environ.get("PIXABAY_API_KEY", ""))
    p.add_argument("--min-width", type=int, default=1920)
    p.add_argument("--max-width", type=int, default=3840,
                   help="skip variants wider than this; 4K files get large fast")
    p.add_argument("--min-duration", type=float, default=5.0,
                   help="seconds. A clip needs seq/fps seconds plus slack to yield even "
                        "one training clip")
    p.add_argument("--max-duration", type=float, default=60.0)
    p.add_argument("--max-disk-gb", type=float, default=0.0)
    p.add_argument("--mix", type=str, default="",
                   help="scene mix, e.g. \"people=25,urban=30,...\". Default matches "
                        "fetch_images.bat: " + ",".join(f"{k}={v}" for k, v in DEFAULT_MIX.items()))
    p.add_argument("--no-balance", action="store_true")
    p.add_argument("--workers", type=int, default=6,
                   help="parallel downloads. Video files are big; 6-12 is plenty")
    p.add_argument("--timeout", type=float, default=60.0)
    p.add_argument("--per-page", type=int, default=0, help="0 = source maximum")
    p.add_argument("--seed", type=int, default=71)
    p.add_argument("--ignore-no-ai-training", action="store_true",
                   help="download Pixabay assets whose creator set noAiTraining. Off by "
                        "default: that flag is the creator declining exactly this use")
    p.add_argument("--allow-ai-generated", action="store_true",
                   help="keep assets flagged isAiGenerated. Off by default: generated "
                        "footage has invented geometry and its own temporal artefacts")
    p.add_argument("--allow-low-quality", action="store_true",
                   help="keep assets Pixabay flags isLowQuality")
    p.add_argument("--probe", action="store_true",
                   help="do one search per configured source, print what came back and "
                        "exit. Run this first -- it is the only way to confirm the keys "
                        "and the response shape from your machine")
    args = p.parse_args()

    try:
        import requests
    except ImportError:
        print("missing dependency: requests\n", file=sys.stderr)
        print(f"nunif uses its own python, not the one on your PATH:\n  {sys.executable}\n",
              file=sys.stderr)
        print("run:  New_Trainer\\install_deps.bat", file=sys.stderr)
        return 1

    keys = {"pexels": args.pexels_key.strip(), "pixabay": args.pixabay_key.strip()}
    sources = [s for s in args.source if keys[s]]
    missing = [s for s in args.source if not keys[s]]
    if missing:
        print(f"no API key for {', '.join(missing)} -- pass --{missing[0]}-key KEY "
              f"or set {missing[0].upper()}_API_KEY", file=sys.stderr)
    if not sources:
        return 2

    base = os.environ.get("NT_CWD") or os.getcwd()
    out = path.abspath(path.join(base, args.output))
    os.makedirs(out, exist_ok=True)

    sess = requests.Session()
    sess.headers["User-Agent"] = UA
    per_page = {"pexels": 80, "pixabay": 200}

    if args.probe:
        for s in sources:
            pp = args.per_page or per_page[s]
            print(f"\n=== {s} ===")
            try:
                rows, err = search(sess, s, keys[s], "city street", 1, min(pp, 5),
                                   args.min_width, args.timeout)
                if err:
                    print(f"  {err}")
                    continue
                print(f"  {len(rows)} result(s)")
                if rows:
                    e = rows[0]
                    f = best_file(e, s, args.min_width, args.max_width)
                    print(f"  id={e.get('id')} duration={entry_duration(e)}s")
                    print(f"  best file >= {args.min_width}px: "
                          + (f"{f[0]}x{f[1]} fps={f[3]}" if f else "NONE (lower --min-width)"))
                    print("  raw keys:", sorted(e.keys()))
                    flags = {k: e.get(k) for k in
                             ("noAiTraining", "isAiGenerated", "isLowQuality")
                             if k in e}
                    if flags:
                        print(f"  per-asset flags: {flags}  (honoured unless "
                              f"--ignore-no-ai-training / --allow-ai-generated)")
            except Exception as ex:                              # noqa: BLE001
                print(f"  FAILED: {type(ex).__name__}: {ex}")
        return 0

    mix = parse_mix(args.mix)
    st = load_state(out)
    seen = set(st["seen"])
    counts = {c: int(st.get("cats", {}).get(c, 0)) for c in ALL_CATS}
    saved, nbytes = st["saved"], st["bytes"]
    target = args.count
    total_w = sum(mix.values())
    quota = {c: target * mix.get(c, 0.0) / total_w for c in ALL_CATS}

    print(f"output : {out}")
    print(f"target : {target} videos, >= {args.min_width}px, "
          f"{args.min_duration:g}-{args.max_duration:g}s"
          + (f"  (resuming from {saved})" if saved else ""))
    print("sources: " + ", ".join(sources))
    if not args.no_balance:
        print("mix    : " + "  ".join(f"{c} {mix.get(c, 0) / total_w * 100:.0f}%"
                                      for c in ALL_CATS if mix.get(c, 0) > 0))
    print()

    rng = random.Random(args.seed)
    lock = threading.Lock()
    stop = threading.Event()
    counters = {"ok": saved, "fail": 0, "bytes": nbytes}
    q = queue.Queue(maxsize=args.workers * 3)
    finished = threading.Event()

    def worker():
        s = requests.Session()
        s.headers["User-Agent"] = UA
        while True:
            try:
                url, name, cat = q.get(timeout=0.5)
            except queue.Empty:
                # An empty queue means the producer is mid API call, NOT that
                # there is no more work. Exiting here killed every worker during
                # the first search, after which the producer blocked forever on
                # a full queue and nothing downloaded at all.
                if finished.is_set() or stop.is_set():
                    return
                continue
            try:
                with s.get(url, timeout=args.timeout, stream=True) as r:
                    if r.status_code != 200:
                        raise IOError(f"http {r.status_code}")
                    expect = int(r.headers.get("Content-Length") or 0)
                    tmp = path.join(out, name + ".part")
                    n = 0
                    aborted = False
                    with open(tmp, "wb") as f:
                        for chunk in r.iter_content(1 << 20):
                            if stop.is_set():
                                # The target was reached, or the run was
                                # cancelled, while this file was still arriving.
                                # It is a fragment, not a video.
                                aborted = True
                                break
                            f.write(chunk)
                            n += len(chunk)
                    # Only a complete file earns its final name. Promoting the
                    # .part unconditionally is what left 0-byte and truncated
                    # .mp4 files in the output folder -- always the last few of
                    # a run, because that is when stop gets set.
                    if aborted:
                        raise IOError("cancelled mid-download")
                    if n < MIN_VIDEO_BYTES:
                        raise IOError(f"suspiciously small ({n} bytes)")
                    if expect and n < expect:
                        raise IOError(f"truncated ({n} of {expect} bytes)")
                    os.replace(tmp, path.join(out, name))
                with lock:
                    counters["ok"] += 1
                    counters["bytes"] += n
                    counts[cat] = counts.get(cat, 0) + 1
                    if counters["ok"] >= target:
                        stop.set()
            except Exception:                                    # noqa: BLE001
                with lock:
                    counters["fail"] += 1
                try:
                    os.remove(path.join(out, name + ".part"))
                except OSError:
                    pass
            finally:
                q.task_done()

    threads = [threading.Thread(target=worker, daemon=True) for _ in range(args.workers)]
    for t in threads:
        t.start()

    def put(item):
        """Never block indefinitely: if the workers have stopped (target reached,
        Ctrl+C) the producer must notice rather than wait on a full queue."""
        while not stop.is_set():
            try:
                q.put(item, timeout=0.5)
                return True
            except queue.Full:
                continue
        return False

    # round-robin: always query the category furthest below its quota
    pages = {}
    rejected = {}
    dry = set()
    exhausted = set()
    t0 = time.time()
    try:
        while counters["ok"] < target and not stop.is_set():
            live = [c for c in ALL_CATS if c not in exhausted and QUERIES.get(c)]
            if not live:
                print("no categories left with results")
                break
            if args.no_balance:
                cat = rng.choice(live)
            else:
                cat = max(live, key=lambda c: quota[c] - counts.get(c, 0))
                if quota[cat] - counts.get(cat, 0) <= 0:
                    cat = rng.choice(live)

            source = sources[len(pages) % len(sources)]
            qtext = rng.choice(QUERIES[cat])
            pkey = (source, cat, qtext)
            pages[pkey] = pages.get(pkey, 0) + 1
            page = pages[pkey]
            pp = args.per_page or per_page[source]

            try:
                rows, err = search(sess, source, keys[source], qtext, page, pp,
                                   args.min_width, args.timeout)
            except Exception as ex:                              # noqa: BLE001
                print(f"  [{source}] {qtext!r}: {type(ex).__name__}: {ex}")
                time.sleep(2)
                continue
            if err:
                print(f"  [{source}] {err}; waiting 30s")
                time.sleep(30)
                continue
            if not rows:
                # this query is used up; the category is only done when every
                # query for it, on every source, has run dry
                dry.add(pkey)
                if all((s, cat, qq) in dry for s in sources for qq in QUERIES[cat]):
                    exhausted.add(cat)
                    print(f"  [{cat}] out of results on every query")
                continue

            # Select first, print, then queue. Queuing blocks while the workers
            # drain (video files are large), so printing afterwards made the tool
            # look dead for minutes at a time.
            batch = []
            for e in rows:
                uid = f"{source}_{e.get('id')}"
                if uid in seen:
                    continue
                why = flag_reject(e, source, allow_ai=args.allow_ai_generated,
                                  allow_optout=args.ignore_no_ai_training,
                                  allow_low=args.allow_low_quality)
                if why:
                    rejected[why] = rejected.get(why, 0) + 1
                    seen.add(uid)
                    continue
                dur = entry_duration(e)
                if dur and not (args.min_duration <= dur <= args.max_duration):
                    continue
                f = best_file(e, source, args.min_width, args.max_width)
                if not f:
                    continue
                seen.add(uid)
                if not args.no_balance and counts.get(cat, 0) + len(batch) >= quota[cat] + 1:
                    break
                batch.append((f[2], f"{uid}.mp4", cat))
            print(f"  [{source}] {cat:9s} {qtext[:34]:34s} p{page} -> {len(rows)} hits, "
                  f"{len(batch)} queued | {counters['ok']}/{target} saved, "
                  f"{human(counters['bytes'])}", flush=True)
            queued = 0
            for item in batch:
                if stop.is_set() or not put(item):
                    break
                queued += 1

            if args.max_disk_gb and counters["bytes"] / 1e9 >= args.max_disk_gb:
                print(f"reached --max-disk-gb {args.max_disk_gb}")
                break
            st.update(seen=sorted(seen), saved=counters["ok"], bytes=counters["bytes"],
                      cats=dict(counts))
            save_state(out, st)
            time.sleep(0.3 if source == "pixabay" else 1.0)   # stay under the rate limits
    except KeyboardInterrupt:
        print("\ninterrupted -- progress saved, re-run to continue")
    finally:
        # let whatever is already queued finish, then stop the workers
        finished.set()
        for t in threads:
            t.join(timeout=max(30.0, args.timeout))
        stop.set()
        for t in threads:
            t.join(timeout=5)
        st.update(seen=sorted(seen), saved=counters["ok"], bytes=counters["bytes"],
                  cats=dict(counts))
        save_state(out, st)

    el = time.time() - t0
    print(f"\n{counters['ok']} videos, {human(counters['bytes'])} on disk, "
          f"{counters['fail']} failed, in {el / 60:.1f} min")
    if rejected:
        print("skipped by per-asset flag: "
              + ", ".join(f"{n} {k}" for k, n in sorted(rejected.items())))
    got = sum(counts.values())
    if got:
        print("\nscene mix:")
        for c in ALL_CATS:
            n = counts.get(c, 0)
            print(f"  {n:6d} ({n / got * 100:5.1f}%)  {c:10s} "
                  f"target {mix.get(c, 0.0) / total_w * 100:5.1f}%")
    print(f"\nnext:\n  make_video_dataset.bat \"{out}\" <dataset-out> --scan\n"
          f"  make_video_dataset.bat \"{out}\" <dataset-out>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
