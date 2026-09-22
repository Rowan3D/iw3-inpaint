r"""
Clip extraction from video files.

Sequential decode with PyAV (which nunif already depends on, so it is present in
nunif's python). Sequential rather than seeking: seeking to arbitrary timestamps
in long-GOP H.264/HEVC either lands on the wrong frame or forces a decode from
the previous keyframe anyway, so it buys nothing and loses robustness.

Two knobs matter for what the temporal model actually learns:

* `fps` -- the *sampled* rate. Consecutive frames of 60fps footage are nearly
  identical, so a clip sampled at 60fps teaches the temporal blocks that nothing
  ever moves. Frames are decimated to `fps` (nagadomi caps at 30 for the same
  reason).
* `skip` -- frames dropped between clips, so successive clips from one video are
  not near-duplicates of each other.
* `stride` -- an extra multiplier on top of the fps decimation. 1 keeps every
  sampled frame; 2 keeps every second one, doubling the motion between
  consecutive frames of a clip without changing how many frames a clip has.
* `start` -- where in the video to begin. Skipping the first seconds avoids
  titles and fade-ins, and a random start makes repeated passes over the same
  library pick different material.

NOTE: this GUI copy adds `stride` and `start`; New_Trainer\ntrainer\video_io.py
has neither.
"""
from __future__ import annotations

import os
import random
from os import path


VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mpg", ".mpeg", ".wmv", ".ts"}


def list_videos(root):
    out = []
    for dirpath, _, names in os.walk(root):
        for n in sorted(names):
            if path.splitext(n)[1].lower() in VIDEO_EXT:
                out.append(path.join(dirpath, n))
    out.sort()
    return out


def probe(filename):
    """(width, height, fps, n_frames_estimate, duration_seconds) or None."""
    import av
    try:
        with av.open(filename) as c:
            if not c.streams.video:
                return None
            s = c.streams.video[0]
            fps = float(s.average_rate or s.guessed_rate or 0) or 0.0
            dur = float(s.duration * s.time_base) if s.duration else (
                float(c.duration / 1e6) if c.duration else 0.0)
            n = s.frames or (int(dur * fps) if fps and dur else 0)
            return (s.codec_context.width, s.codec_context.height, fps, n, dur)
    except Exception:                                            # noqa: BLE001
        return None


def start_offset(filename, start, seq, fps, skip, stride, max_clips, rng=None):
    """Decoded-frame index to begin at, for a `start` of a number of seconds or
    "random".

    A random start must leave enough video behind it to actually produce the
    clips that were asked for, so the latest allowed start is the end minus what
    the run needs. Without that guard a random start near the end silently
    yields nothing and the video looks like it failed.
    """
    info = probe(filename)
    if info is None:
        return 0
    _, _, src_fps, n_frames, dur = info
    if not start:
        return 0
    step = max(1, int(round((src_fps / fps) if (fps and src_fps and src_fps > fps) else 1))) * max(1, stride)
    wanted = max(1, max_clips or 1)
    need = (seq * wanted + skip * max(0, wanted - 1)) * step
    if not n_frames:
        n_frames = int((dur or 0) * (src_fps or 0))
    latest = max(0, int(n_frames) - need)
    if isinstance(start, str) and start.strip().lower() == "random":
        if latest <= 0:
            return 0
        return (rng or random).randint(0, latest)
    try:
        seconds = float(start)
    except (TypeError, ValueError):
        return 0
    if seconds <= 0 or not src_fps:
        return 0
    return min(int(seconds * src_fps), latest)


def iter_clips(filename, seq=12, fps=30.0, skip=24, max_clips=0, min_width=0,
               stride=1, start=0, rng=None):
    """Yield (seq, H, W, 3) uint8 arrays of consecutive frames, decimated to ~`fps`.

    uint8 ndarray rather than PIL on purpose. `TF.to_tensor()` on twelve 4K PIL
    images costs 10.9 s and allocates 1.19 GB of CPU float32; the same frames as
    stacked uint8 are 0.30 GB and the float conversion belongs on the GPU.

    `skip` sampled frames are dropped between clips. Stops after `max_clips`
    (0 = until the file ends). `stride` multiplies the sampling step, so
    stride=2 takes every second sampled frame. `start` is a number of seconds to
    skip first, or the string "random".
    """
    import av
    import numpy as np

    info = probe(filename)
    if info is None:
        return
    w, h, src_fps, _, _ = info
    if min_width and w < min_width:
        return
    step = 1
    if fps and src_fps and src_fps > fps:
        step = max(1, int(round(src_fps / fps)))
    step *= max(1, int(stride or 1))

    begin = start_offset(filename, start, seq, fps, skip, stride, max_clips, rng=rng)

    produced = 0
    buf = []
    skipping = 0
    i = -1
    with av.open(filename) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        for frame in container.decode(stream):
            i += 1
            if i < begin:
                continue
            if (i - begin) % step:
                continue
            if skipping > 0:
                skipping -= 1
                continue
            buf.append(frame.to_ndarray(format="rgb24"))
            if len(buf) < seq:
                continue
            yield np.stack(buf)
            produced += 1
            buf = []
            skipping = skip
            if max_clips and produced >= max_clips:
                return
