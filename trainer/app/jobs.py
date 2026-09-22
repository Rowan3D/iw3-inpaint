r"""
Runs one long tool at a time as a subprocess and reports progress back to the
page.

Progress comes from the tool's own stdout. Each job type registers a regex that
pulls "done / total" out of a line it already prints, so the bundled tools stay
byte-identical to New_Trainer's and can be re-copied over at any time. When a
tool prints nothing parseable, a counter callback (files on disk, say) is used
instead, and failing that the bar just runs indeterminate.

ETA is a least-squares fit over the last 60 points at which progress actually
changed -- not the last 60 polls -- rather than total*elapsed/done, because
these jobs do not start at a steady rate: the first minute of a download is API
calls with nothing saved, and an average that includes it reads far too
pessimistic for the next hour.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import threading
import time
from collections import deque


IS_WINDOWS = os.name == "nt"

# nunif's training loop prints a tqdm bar for the train pass and another for the
# eval pass, and nothing else about what is happening inside an epoch. An epoch
# is minutes long, so without this the only feedback is a bar that jumps once
# every few minutes.
#   " 45%|####      | 90/200 [00:30<00:37,  2.95it/s]"
TQDM_RE = re.compile(
    r"(?P<pct>\d+)%\|[^|]*\|\s*(?P<n>\d+)/(?P<total>\d+)"
    r"\s*\[(?P<elapsed>[^<\]]*)<(?P<left>[^,\]]*),\s*(?P<rate>[^\]]+)\]")
# the loop prints a bare " train" / " eval" line before each bar
STAGE_RE = re.compile(r"^\s*(train|eval)\s*$")
EPOCH_RE = re.compile(r"^\s*epoch:\s*(\d+)")


def _fmt_eta(seconds):
    if seconds is None or seconds < 0 or seconds != seconds:      # NaN-safe
        return ""
    seconds = int(seconds)
    if seconds < 90:
        return f"{seconds}s"
    m, s = divmod(seconds, 60)
    if m < 90:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m"


class Job:
    def __init__(self, name, argv, cwd=None, env=None, progress_re=None,
                 final_re=None, counter=None, total=None, label="", preamble=None):
        self.name = name
        self.argv = [str(a) for a in argv]
        self.cwd = cwd
        self.env = env
        self.label = label
        self.preamble = list(preamble or [])   # logged before the command line
        self.progress_re = re.compile(progress_re) if progress_re else None
        # Several tools print their running total only once per API page or
        # batch, so the last few items finish after the final status line. The
        # summary line they print at the end carries the true count -- without
        # it the bar sticks just short of the end (47 of 50) on a job that
        # actually completed.
        self.final_re = re.compile(final_re) if final_re else None
        self.counter = counter          # callable -> int, polled every 2s
        self.total = total

        self.proc = None
        self.lines = deque(maxlen=4000)
        self.started = None
        self.finished = None
        self.returncode = None
        self.done = 0
        self.extra = ""                 # free-text status, e.g. "4.5 GB on disk"
        self.error = ""
        self._samples = deque(maxlen=60)
        self._regex_hits = 0
        # what is happening inside the current unit of work
        self.sub = {"stage": "", "epoch": None, "n": 0, "total": 0,
                    "percent": None, "rate": "", "left": ""}
        self._lock = threading.Lock()
        self._stopping = False

    # ---- lifecycle -------------------------------------------------------
    def start(self):
        env = dict(os.environ)
        env.update(self.env or {})
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        creation = 0
        if IS_WINDOWS:
            creation = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
                subprocess, "CREATE_NO_WINDOW", 0)
        try:
            self.proc = subprocess.Popen(
                self.argv, cwd=self.cwd or None, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True, encoding="utf-8", errors="replace",
                bufsize=1, creationflags=creation)
        except OSError as e:
            self.error = f"could not start: {e}"
            self.finished = time.time()
            self.returncode = -1
            self._log(f"[gui] {self.error}")
            return self
        self.started = time.time()
        for line in self.preamble:
            self._log("[gui] " + line)
        self._log("[gui] " + " ".join(
            a if " " not in a else f'"{a}"' for a in self.argv))
        threading.Thread(target=self._pump, daemon=True).start()
        if self.counter:
            threading.Thread(target=self._poll_counter, daemon=True).start()
        return self

    def _log(self, line):
        with self._lock:
            self.lines.append(line.rstrip("\n"))

    def _pump(self):
        try:
            for line in self.proc.stdout:
                line = line.rstrip("\n")
                # tqdm redraws with \r; keep only the final state of the line
                if "\r" in line:
                    line = line.rsplit("\r", 1)[-1]
                if line.strip():
                    self._log(line)
                    self._parse(line)
        except Exception as e:                                    # noqa: BLE001
            self._log(f"[gui] output stopped: {e}")
        finally:
            self.returncode = self.proc.wait()
            self.finished = time.time()
            if self.returncode not in (0, None) and not self._stopping:
                self.error = self.error or f"exited with code {self.returncode}"
            self._log(f"[gui] finished, exit code {self.returncode}")

    def _parse(self, line):
        self._parse_sub(line)
        if self.final_re:
            f = self.final_re.search(line)
            if f:
                self._regex_hits += 1
                self._apply(f.groupdict(), allow_total=False)
                return
        if not self.progress_re:
            return
        m = self.progress_re.search(line)
        if not m:
            return
        self._regex_hits += 1
        self._apply(m.groupdict())

    def _parse_sub(self, line):
        """Track the inner progress bar, independent of the job's own progress."""
        m = EPOCH_RE.match(line)
        if m:
            try:
                self.sub["epoch"] = int(m.group(1))
            except ValueError:
                pass
            self.sub.update(stage="", n=0, total=0, percent=None, rate="", left="")
            return
        m = STAGE_RE.match(line)
        if m:
            self.sub.update(stage=m.group(1), n=0, total=0, percent=None,
                            rate="", left="")
            return
        m = TQDM_RE.search(line)
        if m:
            try:
                n, total = int(m.group("n")), int(m.group("total"))
            except ValueError:
                return
            self.sub.update(n=n, total=total,
                            percent=(100.0 * n / total) if total else None,
                            rate=m.group("rate").strip(),
                            left=m.group("left").strip())

    def _apply(self, g, allow_total=True):
        try:
            if g.get("done") is not None:
                self._set_done(int(float(g["done"])))
            if allow_total and g.get("total"):
                self.total = int(float(g["total"]))
            if g.get("extra"):
                self.extra = g["extra"].strip()
        except (TypeError, ValueError):
            pass

    def _poll_counter(self):
        while self.finished is None:
            # the tool's own numbers win as soon as it prints any; the folder
            # count is only here for tools that report nothing parseable, and
            # it would otherwise mix in files from earlier runs
            try:
                if self._regex_hits == 0:
                    n = self.counter()
                else:
                    n = None
                if n is not None:
                    self._set_done(int(n))
            except Exception:                                     # noqa: BLE001
                pass
            time.sleep(2.0)

    def _set_done(self, n):
        if n < self.done:
            return
        # Only record when the number actually MOVES. The counter polls every
        # two seconds, so appending on every poll filled the 60-sample window
        # with two minutes of history -- and a job whose unit of work takes
        # seven minutes then had its ETA fitted to whether an epoch happened to
        # tick inside those two minutes. That came out about five times too
        # optimistic on a 3x200 epoch run. Deduplicated, the same 60 samples
        # span the last 60 epochs, which is what the fit was meant to see.
        if self._samples and n == self._samples[-1][1]:
            self.done = n
            return
        self.done = n
        self._samples.append((time.time(), n))

    # ---- reporting -------------------------------------------------------
    def eta(self):
        """Least squares over the recent samples -> seconds remaining."""
        if not self.total or self.done <= 0 or self.done >= self.total:
            return None
        pts = [p for p in self._samples if p[1] > 0]
        if len(pts) < 3:
            return None
        t0 = pts[0][0]
        xs = [p[0] - t0 for p in pts]
        ys = [float(p[1]) for p in pts]
        n = len(pts)
        mx, my = sum(xs) / n, sum(ys) / n
        denom = sum((x - mx) ** 2 for x in xs)
        if denom <= 0:
            return None
        rate = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / denom  # per second
        if rate <= 1e-9:
            return None
        return (self.total - self.done) / rate

    def running(self):
        return self.proc is not None and self.finished is None

    def stop(self):
        self._stopping = True
        if not self.running():
            return False
        self._log("[gui] stopping...")
        return self._kill()

    def _kill(self):
        proc = self.proc
        if proc is None or proc.poll() is not None:
            return True                       # already gone; nothing to kill
        try:
            if IS_WINDOWS:
                # the tools spawn worker threads and dataloader child processes;
                # /T takes the whole tree down rather than orphaning them
                subprocess.run(["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                               capture_output=True,
                               creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            else:
                proc.terminate()
        except Exception as e:                                    # noqa: BLE001
            self._log(f"[gui] stop failed: {e}")
            return False
        return True

    def state(self, tail=200):
        with self._lock:
            lines = list(self.lines)[-tail:]
        elapsed = (self.finished or time.time()) - (self.started or time.time())
        eta = self.eta()
        pct = None
        if self.total:
            pct = max(0.0, min(100.0, 100.0 * self.done / self.total))
        if self.finished is not None and self.returncode == 0:
            # The job succeeded, so the bar is full whatever the last number
            # was: a tool may legitimately finish under its target (it ran out
            # of results), and a job with nothing countable at all -- pip, say
            # -- has no total to divide by. Either way, leaving the bar part
            # drawn on a finished job reads as a failure.
            pct = 100.0
        return {
            "name": self.name,
            "label": self.label,
            "running": self.running(),
            "done": self.done,
            "total": self.total,
            "percent": pct,
            "eta": _fmt_eta(eta),
            "eta_seconds": eta,
            "elapsed": _fmt_eta(elapsed),
            "extra": self.extra,
            "error": self.error,
            "returncode": self.returncode,
            "sub": dict(self.sub),
            "lines": lines,
        }


class ChainJob(Job):
    """Several commands run one after another in a single slot.

    Training is several 200-epoch phases plus a final copy, and the user should
    see one bar for the whole thing, not have to babysit each phase. A step that
    fails stops the chain -- there is no point running phase 3 on a phase 2 that
    crashed.

    `steps` is a list of dicts: {"argv": [...], "label": "..."} for a command, or
    {"call": fn, "label": "..."} for a bit of Python (the final rename).
    """

    def __init__(self, name, steps, cwd=None, env=None, counter=None, total=None,
                 label="", preamble=None, extend=None):
        super().__init__(name, argv=["<chain>"], cwd=cwd, env=env, counter=counter,
                         total=total, label=label, preamble=preamble)
        self.steps = list(steps)
        self.step_index = 0
        # Called with this job after every successful step. It may append or
        # insert steps -- the loop below reads self.steps as it goes, so a step
        # added while the chain is running is picked up. That is how "keep
        # training past the plan" can be switched on mid-run.
        self.extend = extend

    def start(self):
        self.started = time.time()
        for line in self.preamble:
            self._log("[gui] " + line)
        threading.Thread(target=self._run_chain, daemon=True).start()
        if self.counter:
            threading.Thread(target=self._poll_counter, daemon=True).start()
        return self

    def running(self):
        return self.started is not None and self.finished is None

    def _run_chain(self):
        rc = 0
        try:
            for i, step in enumerate(self.steps):
                if self._stopping:
                    break
                self.step_index = i
                self._log(f"[gui] --- step {i + 1} of {len(self.steps)}: "
                          f"{step.get('label', '')} ---")
                if step.get("call"):
                    try:
                        msg = step["call"]()
                        if msg:
                            self._log("[gui] " + str(msg))
                    except Exception as e:                        # noqa: BLE001
                        self.error = f"{step.get('label', 'step')}: {e}"
                        self._log("[gui] " + self.error)
                        rc = 1
                        break
                    continue
                rc = self._run_one(step["argv"])
                if rc != 0:
                    if not self._stopping:
                        self.error = (f"step {i + 1} ({step.get('label', '')}) "
                                      f"exited with code {rc}")
                    break
                if self.extend and not self._stopping:
                    try:
                        note = self.extend(self)
                        if note:
                            self._log("[gui] " + str(note))
                    except Exception as e:                        # noqa: BLE001
                        self._log(f"[gui] could not extend the chain: {e}")
        finally:
            self.returncode = rc
            self.finished = time.time()
            self._log(f"[gui] all steps finished, exit code {rc}")

    def _run_one(self, argv):
        env = dict(os.environ)
        env.update(self.env or {})
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONIOENCODING"] = "utf-8"
        creation = 0
        if IS_WINDOWS:
            creation = subprocess.CREATE_NEW_PROCESS_GROUP | getattr(
                subprocess, "CREATE_NO_WINDOW", 0)
        argv = [str(a) for a in argv]
        self._log("[gui] " + " ".join(a if " " not in a else f'"{a}"' for a in argv))
        try:
            self.proc = subprocess.Popen(
                argv, cwd=self.cwd or None, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL, text=True, encoding="utf-8",
                errors="replace", bufsize=1, creationflags=creation)
        except OSError as e:
            self.error = f"could not start: {e}"
            self._log("[gui] " + self.error)
            return -1
        # Stop() kills whatever self.proc points at. Between two chain steps
        # that is the PREVIOUS, already-dead process, so a Stop landing in the
        # gap -- or during the Popen above -- killed nothing and left this new
        # process running with the GUI showing it as stopped. Re-check now that
        # self.proc is the live one.
        if self._stopping:
            self._log("[gui] stop arrived while this step was starting")
            self._kill()
        try:
            for line in self.proc.stdout:
                line = line.rstrip("\n")
                if "\r" in line:
                    line = line.rsplit("\r", 1)[-1]
                if line.strip():
                    self._log(line)
                    self._parse(line)
        except Exception as e:                                    # noqa: BLE001
            self._log(f"[gui] output stopped: {e}")
        return self.proc.wait()

    def state(self, tail=200):
        st = super().state(tail=tail)
        st["step"] = self.step_index + 1
        st["steps"] = len(self.steps)
        lab = self.steps[self.step_index].get("label", "") if self.steps else ""
        # The page used to work the run number out from step_index itself, which
        # is wrong whenever a chain is restarted part-way: step 1 of the new
        # chain can be phase 3 of the training. The builder knows the real phase
        # numbers, so it puts them in the label and the page just shows it.
        st["steplabel"] = lab
        if st["running"] and lab:
            st["label"] = f"{self.label} \u2014 {lab}"
        return st


class JobManager:
    """One job per slot name; starting a slot that is busy is refused rather
    than silently killing the running one."""

    def __init__(self):
        self.jobs = {}
        self._lock = threading.Lock()

    def start(self, name, cls=Job, **kw):
        with self._lock:
            cur = self.jobs.get(name)
            if cur is not None and cur.running():
                raise RuntimeError(f"{name} is already running")
            job = cls(name, **kw)
            self.jobs[name] = job
        job.start()
        return job

    def get(self, name):
        return self.jobs.get(name)

    def stop(self, name):
        job = self.jobs.get(name)
        return bool(job and job.stop())

    def any_running(self):
        return [n for n, j in self.jobs.items() if j.running()]

    def state(self, name, tail=200):
        job = self.jobs.get(name)
        return job.state(tail=tail) if job else None


def count_files(folder, exts=None, recursive=False):
    """Fallback progress source: how many output files exist so far."""
    def _count():
        if not os.path.isdir(folder):
            return 0
        n = 0
        if recursive:
            for _, _, names in os.walk(folder):
                for f in names:
                    if not exts or os.path.splitext(f)[1].lower() in exts:
                        n += 1
        else:
            for f in os.listdir(folder):
                if not exts or os.path.splitext(f)[1].lower() in exts:
                    n += 1
        return n
    return _count


def count_dirs(folder):
    def _count():
        if not os.path.isdir(folder):
            return 0
        return sum(1 for f in os.listdir(folder)
                   if os.path.isdir(os.path.join(folder, f)))
    return _count
