/* iw3 inpaint trainer -- GUI front end.
   No build step, no libraries: the file is served as-is so it can be read and
   edited in place. */
"use strict";

const $  = id => document.getElementById(id);
const $$ = sel => Array.from(document.querySelectorAll(sel));

let STATE = null;         // last /api/state
let STATE_PATHS = null;   // the derived folders as of the last save
let POLL = null;          // job poll timer
const JOB_SLOTS = {};     // strip element -> job name currently shown

/* ------------------------------------------------------------------ util */

async function api(url, body) {
  const opt = body === undefined
    ? { cache: "no-store" }
    : { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body || {}) };
  const res = await fetch(url, opt);
  let data = {};
  try { data = await res.json(); } catch (e) { /* empty body */ }
  if (!res.ok) throw new Error(data.error || `${res.status} ${res.statusText}`);
  return data;
}

let toastTimer = null;
function toast(msg, bad) {
  const t = $("toast");
  t.textContent = msg;
  t.classList.toggle("bad", !!bad);
  t.classList.remove("hidden");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => t.classList.add("hidden"), bad ? 6000 : 3200);
}

function setStatus(el, text, kind) {
  el.textContent = text || "";
  el.classList.remove("ok", "bad");
  if (kind) el.classList.add(kind);
}

function human(bytes) {
  let n = bytes, u = ["B", "KB", "MB", "GB", "TB"], i = 0;
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return `${n.toFixed(1)} ${u[i]}`;
}

/* ------------------------------------------------- tooltips (550ms hover) */

let tipTimer = null, tipTarget = null, tipShown = false;

function showTip(el) {
  const text = el.getAttribute("data-tip");
  if (!text) return;
  const tip = $("tip");
  tipShown = true;
  tip.textContent = text;
  tip.classList.remove("hidden");
  const r = el.getBoundingClientRect(), t = tip.getBoundingClientRect();
  let x = r.left, y = r.bottom + 8;
  if (x + t.width > innerWidth - 10) x = innerWidth - t.width - 10;
  if (y + t.height > innerHeight - 10) y = r.top - t.height - 8;
  tip.style.left = Math.max(10, x) + "px";
  tip.style.top = Math.max(10, y) + "px";
}
function hideTip() {
  clearTimeout(tipTimer);
  tipTimer = null;
  tipTarget = null;
  tipShown = false;
  $("tip").classList.add("hidden");
}
document.addEventListener("mouseover", e => {
  const el = e.target.closest("[data-tip]");
  if (!el || el === tipTarget) return;
  // Pages that poll (the job strips, the live view) rebuild their markup every
  // couple of seconds, which swaps the node under a stationary cursor for an
  // identical one. Restarting the 550ms wait each time meant the tip on those
  // controls never appeared at all. Same text, cursor never left: keep it up.
  const rebuilt = tipShown && tipTarget && !tipTarget.isConnected &&
                  tipTarget.getAttribute("data-tip") === el.getAttribute("data-tip");
  tipTarget = el;
  if (rebuilt) { showTip(el); return; }
  clearTimeout(tipTimer);
  tipShown = false;
  $("tip").classList.add("hidden");
  tipTimer = setTimeout(() => { if (tipTarget === el) showTip(el); }, 550);
});
document.addEventListener("mouseout", e => {
  // mouseout bubbles, so it also fires every time the pointer crosses from one
  // child of the tooltipped element to another -- a label to its own input, a
  // button to the text inside it. Hiding on that made the tip vanish a split
  // second after it appeared on nearly every control that has children, which
  // is most of them. Only hide once the pointer has actually left.
  if (!tipTarget || !tipTarget.contains(e.target)) return;
  const to = e.relatedTarget;
  if (to && tipTarget.contains(to)) return;
  hideTip();
});
document.addEventListener("scroll", e => {
  if (!tipTarget) return;
  // The job log scrolls itself to the bottom on every poll -- about once a
  // second, for as long as anything is running. Hiding on any scroll at all
  // therefore killed the tooltip a moment after it appeared whenever a job was
  // on screen, which is most of the time. React only when whatever scrolled
  // actually moved the element the tip belongs to.
  const t = e.target;
  const moved = t === document || t === document.documentElement ||
                t === document.body || (t.contains && t.contains(tipTarget));
  if (!moved) return;
  if (tipShown) showTip(tipTarget); else hideTip();
}, true);
// A tip hanging over the thing you just clicked is in the way.
document.addEventListener("click", hideTip, true);

/* ----------------------------------------------------------------- themes */

const THEMES = [
 {
  "id": "midnight",
  "name": "Midnight",
  "blurb": "The default. Neutral slate with a blue accent.",
  "dark": true,
  "swatch": [
   "#14161a",
   "#5aa9ff",
   "#ffb454",
   "#e7e9ee"
  ]
 },
 {
  "id": "tokyonight",
  "name": "Tokyo Night",
  "blurb": "Soft blue-violet night sky. Hugely popular in Neovim and VS Code.",
  "dark": true,
  "swatch": [
   "#24283b",
   "#7aa2f7",
   "#ff9e64",
   "#c0caf5"
  ]
 },
 {
  "id": "dracula",
  "name": "Dracula",
  "blurb": "The most ported theme there is. Purple and pink on dark slate.",
  "dark": true,
  "swatch": [
   "#282a36",
   "#bd93f9",
   "#ffb86c",
   "#f8f8f2"
  ]
 },
 {
  "id": "nord",
  "name": "Nord",
  "blurb": "Arctic and low-saturation. Calm blues, easy for long sessions.",
  "dark": true,
  "swatch": [
   "#2e3440",
   "#88c0d0",
   "#ebcb8b",
   "#d8dee9"
  ]
 },
 {
  "id": "mocha",
  "name": "Catppuccin Mocha",
  "blurb": "Pastel and soothing, the community favourite dark flavour.",
  "dark": true,
  "swatch": [
   "#1e1e2e",
   "#89b4fa",
   "#fab387",
   "#cdd6f4"
  ]
 },
 {
  "id": "gruvbox",
  "name": "Gruvbox Dark",
  "blurb": "Retro-groove warmth. Earthy tones, deliberately low contrast.",
  "dark": true,
  "swatch": [
   "#282828",
   "#83a598",
   "#fe8019",
   "#ebdbb2"
  ]
 },
 {
  "id": "onedark",
  "name": "One Dark",
  "blurb": "Atom's signature look, still everywhere via One Dark Pro.",
  "dark": true,
  "swatch": [
   "#282c34",
   "#61afef",
   "#d19a66",
   "#abb2bf"
  ]
 },
 {
  "id": "rosepine",
  "name": "Ros\u00e9 Pine",
  "blurb": "Muted rose and gold on deep plum. Soho vibes.",
  "dark": true,
  "swatch": [
   "#191724",
   "#c4a7e7",
   "#f6c177",
   "#e0def4"
  ]
 },
 {
  "id": "everforest",
  "name": "Everforest",
  "blurb": "Green-based and low contrast, built to be easy on the eyes.",
  "dark": true,
  "swatch": [
   "#2d353b",
   "#7fbbb3",
   "#e69875",
   "#d3c6aa"
  ]
 },
 {
  "id": "latte",
  "name": "Catppuccin Latte",
  "blurb": "The light pastel flavour. Bright without being harsh.",
  "dark": false,
  "swatch": [
   "#eff1f5",
   "#1e66f5",
   "#fe640b",
   "#4c4f69"
  ]
 },
 {
  "id": "solarized",
  "name": "Solarized Light",
  "blurb": "Ethan Schoonover's precision light scheme. Warm paper tone.",
  "dark": false,
  "swatch": [
   "#fdf6e3",
   "#268bd2",
   "#cb4b16",
   "#657b83"
  ]
 }
];

function applyTheme(id, persist) {
  if (!THEMES.some(t => t.id === id)) id = "midnight";
  document.documentElement.dataset.theme = id;
  try { localStorage.setItem("nt_theme", id); } catch (e) { /* private mode */ }
  $$("#themeList .theme").forEach(b => b.classList.toggle("on", b.dataset.id === id));
  if (persist) api("/api/config", { theme: id }).catch(() => {});
}

function renderThemes() {
  const cur = document.documentElement.dataset.theme || "midnight";
  $("themeList").innerHTML = THEMES.map(t => `
    <button class="theme ${t.id === cur ? "on" : ""}" data-id="${t.id}"
            data-tip="${t.blurb} (${t.dark ? "dark" : "light"})">
      <span class="sw">${t.swatch.map(c => `<i style="background:${c}"></i>`).join("")}</span>
      <span class="nm"><b>${t.name}</b><span>${t.dark ? "dark" : "light"}</span></span>
    </button>`).join("");
  $$("#themeList .theme").forEach(b =>
    b.addEventListener("click", () => applyTheme(b.dataset.id, true)));
}

/* ------------------------------------------------------------ navigation */

function showPage(n) {
  // Navigating away from an open setup panel used to drop whatever was in the
  // path box. Settle it first.
  const leavingSetup = !$("setup").classList.contains("hidden");
  $$(".tab").forEach(t => t.classList.toggle("on", t.dataset.page === String(n)));
  for (let i = 1; i <= 5; i++) $("page" + i).classList.toggle("hidden", i !== n);
  $("setup").classList.add("hidden");
  if (leavingSetup && typeof checkNunifPath === "function") checkNunifPath(true, true);
  api("/api/config", { page: n }).catch(() => {});
  if (n === 4 && typeof pollLive === "function") pollLive(true);
  if (n === 5 && typeof refreshModels === "function") refreshModels();
}
$$(".tab").forEach(t => t.addEventListener("click", () => showPage(+t.dataset.page)));

function showSetup(on) {
  $("setup").classList.toggle("hidden", !on);
  for (let i = 1; i <= 5; i++) $("page" + i).classList.add("hidden");
  if (!on) showPage(1);
  // Say straight away whether the saved path is still good, rather than
  // waiting for a keystroke or for Save to be pressed.
  else if (typeof checkNunifPath === "function") checkNunifPath(true);
}
$("setupBtn").addEventListener("click", () => showSetup(true));

/* --------------------------------------------------------- folder picker */

let pickTarget = null;

async function openPicker(inputId, start) {
  pickTarget = inputId;
  $("picker").classList.remove("hidden");
  setStatus($("pickErr"), "");
  await loadPicker(start || $(inputId).value || "");
}
async function loadPicker(p) {
  let d;
  try { d = await api("/api/browse", { path: p }); }
  catch (e) { setStatus($("pickErr"), e.message, "bad"); return; }
  $("pickPath").value = d.path || "";
  $("pickDrives").innerHTML = (d.drives || [])
    .map(x => `<button class="ghost" data-go="${x}">${x}</button>`).join("");
  const rows = [];
  if (d.parent) rows.push(`<li class="up" data-go="${d.parent}">.. (up one level)</li>`);
  (d.dirs || []).forEach(name => {
    const full = (d.path.endsWith("\\") || d.path.endsWith("/"))
      ? d.path + name : d.path + (d.path.includes("\\") ? "\\" : "/") + name;
    rows.push(`<li data-go="${full.replace(/"/g, "&quot;")}">${name}</li>`);
  });
  $("pickList").innerHTML = rows.join("") ||
    `<li style="cursor:default;color:#6d7688">no subfolders here</li>`;
  setStatus($("pickErr"), d.error || "", d.error ? "bad" : "");
}
$("picker").addEventListener("click", e => {
  const go = e.target.closest("[data-go]");
  if (go) { loadPicker(go.dataset.go); return; }
  if (e.target === $("picker")) closePicker();
});
$("pickPath").addEventListener("keydown", e => {
  if (e.key === "Enter") loadPicker($("pickPath").value);
});
function closePicker() { $("picker").classList.add("hidden"); pickTarget = null; }
$("pickClose").addEventListener("click", closePicker);
$("pickUse").addEventListener("click", () => {
  if (pickTarget) {
    $(pickTarget).value = $("pickPath").value;
    $(pickTarget).dispatchEvent(new Event("change", { bubbles: true }));
  }
  closePicker();
});
document.addEventListener("click", e => {
  const b = e.target.closest("button.browse");
  if (b) openPicker(b.dataset.target);
});
$("browseNunif").addEventListener("click", () => openPicker("nunifPath"));

/* -------------------------------------------------------------- job strip */

function stripFor(name) {
  return document.querySelector(`.jobstrip[data-job="${name}"]`);
}

function renderJob(el, st) {
  // A slot the server knows nothing about yet comes back as a bare stub; show
  // nothing for it rather than an empty bar labelled with the internal name.
  const neverRan = !st || (!st.running && st.returncode === undefined
                           && !(st.lines || []).length);
  if (neverRan) { el.classList.add("hidden"); return; }
  el.classList.remove("hidden");
  const pct = st.percent;
  const known = pct !== null && pct !== undefined;
  const bits = [];
  if (st.total) bits.push(`${st.done} of ${st.total}`);
  else if (st.done) bits.push(`${st.done}`);
  if (st.extra) bits.push(st.extra);
  if (st.elapsed) bits.push(`${st.elapsed} elapsed`);

  let right = "";
  if (st.running && st.eta) right = `about ${st.eta} left`;
  else if (st.running) right = "working...";
  else if (st.error) right = st.error;
  else if (st.returncode === 0) right = "done";

  el.innerHTML = `
    <div class="jobhead">
      <b>${st.label || st.name}${known ? " &middot; " + pct.toFixed(0) + "%" : ""}</b>
      <span>${bits.join(" &middot; ")}</span>
      <span class="${st.error ? "status bad" : ""}">${right}</span>
    </div>
    <div class="bar ${st.running && !known ? "indet" : ""}"><i style="width:${known ? pct : 0}%"></i></div>
    <div class="joblog"></div>`;
  const log = el.querySelector(".joblog");
  log.innerHTML = (st.lines || []).map(l => {
    const cls = l.startsWith("[gui]") ? "gui"
      : /error|traceback|failed|missing dependency/i.test(l) ? "err" : "";
    return `<div class="${cls}">${l.replace(/[<>&]/g, c =>
      ({ "<": "&lt;", ">": "&gt;", "&": "&amp;" }[c]))}</div>`;
  }).join("");
  log.scrollTop = log.scrollHeight;
}

async function pollJobs() {
  const names = Object.values(JOB_SLOTS).filter(Boolean);
  let anyRunning = false;
  for (const name of new Set(names)) {
    let st;
    try { st = await api("/api/job/" + name); } catch (e) { continue; }
    if (st.running) anyRunning = true;
    for (const [slot, n] of Object.entries(JOB_SLOTS)) {
      if (n === name) renderJob(stripFor(slot), st);
    }
    if (name === currentFetchJob()) {
      $("startBtn").classList.toggle("hidden", !!st.running);
      $("stopBtn").classList.toggle("hidden", !st.running);
      if (!st.running && st.returncode === 0) {
        $("nextHint").textContent = "Finished. Move on to step 2 when you are ready.";
      }
    }
    if (name === "deps") $("depsBtn").disabled = !!st.running;
    if (name === "probe") $("probeBtn").disabled = !!st.running;
    if (name === "scan") $("scanBtn").disabled = !!st.running;
    if (name === "verify") $("verifyBtn").disabled = !!st.running;
    if (name === "bench") {
      $("benchBtn").disabled = !!st.running;
      if (!st.running && st.returncode === 0) renderBench(st.lines);
      if (!st.running && st.returncode) {
        $("benchHint").textContent = "The measurement failed \u2014 see the log below.";
      }
    }
    if (name === "train") {
      $("trainBtn").classList.toggle("hidden", !!st.running);
      $("trainStop").classList.toggle("hidden", !st.running);
      $("toLive").classList.toggle("hidden", !st.running);
      if (!st.running && st.returncode === 0) {
        $("trainHint").textContent = "Finished. Step 5 can install it into iw3.";
      }
    }
    if (name === "prep") {
      $("prepBtn").classList.toggle("hidden", !!st.running);
      $("prepStop").classList.toggle("hidden", !st.running);
      if (!st.running && st.returncode === 0) {
        $("prepHint").textContent = "Finished. Step 3 will use this dataset.";
      }
    }
  }
  clearTimeout(POLL);
  POLL = setTimeout(pollJobs, anyRunning ? 1200 : 4000);
}

/* ------------------------------------------------------------- page 1 */

function mediaType() {
  const r = document.querySelector('input[name="media"]:checked');
  return r ? r.value : "video";
}
function currentFetchJob() {
  return mediaType() === "image" ? "fetch_images" : "fetch_videos";
}

function applyMediaType() {
  const m = mediaType();
  $$(".vid-only").forEach(e => e.classList.toggle("hidden", m !== "video"));
  $$(".img-only").forEach(e => e.classList.toggle("hidden", m !== "image"));
  $("downloadOpts").classList.toggle("hidden", m === "own");
  $("ownOpts").classList.toggle("hidden", m !== "own");
  $("startBtn").classList.toggle("hidden", m === "own");
  $("startBtn").textContent = m === "image" ? "Start download" : "Start download";
  $("countHint").textContent = m === "image"
    ? "20000 or more is normal for images; they are small."
    : "About 2000–5000 videos gives the model enough variety.";
  if (m === "image" && +$("count").value === 400) $("count").value = 20000;
  if (m === "video" && +$("count").value === 20000) $("count").value = 400;
  $("workers").value = m === "image" ? 16 : 6;
  JOB_SLOTS["fetch"] = currentFetchJob();
  applyPrepMode();
  saveCollect();
}
$$('input[name="media"]').forEach(r => r.addEventListener("change", applyMediaType));

const COLLECT_FIELDS = ["count", "max_disk_gb", "min_width", "workers", "min_duration",
  "max_duration", "max_width", "mix", "src_pexels", "src_pixabay",
  "allow_ai_generated", "ignore_no_ai_training", "allow_low_quality"];

function readCollect() {
  const o = {};
  COLLECT_FIELDS.forEach(id => {
    const el = $(id);
    if (!el) return;
    o[id] = el.type === "checkbox" ? el.checked
      : el.type === "number" ? (el.value === "" ? "" : +el.value) : el.value;
  });
  o.media = mediaType();
  o.image_source = (document.querySelector('input[name="imgsrc"]:checked') || {}).value || "pd12m";
  o.download_dir = $("collectDir").value.trim();
  return o;
}
function writeCollect(o) {
  if (!o) return;
  COLLECT_FIELDS.forEach(id => {
    const el = $(id);
    if (!el || o[id] === undefined) return;
    if (el.type === "checkbox") el.checked = !!o[id]; else el.value = o[id];
  });
  if (o.media) {
    const r = document.querySelector(`input[name="media"][value="${o.media}"]`);
    if (r) r.checked = true;
  }
  if (o.image_source) {
    const r = document.querySelector(`input[name="imgsrc"][value="${o.image_source}"]`);
    if (r) r.checked = true;
  }
}

let saveTimer = null;
function saveCollect() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    const o = readCollect();
    const patch = { collect: o, media_type: o.media };
    if (o.download_dir) patch.download_dir = o.download_dir;
    api("/api/config", patch).catch(() => {});
  }, 400);
}
document.addEventListener("change", e => {
  if (!e.target.closest("#page1")) return;
  if (e.target.id === "collectDir" && $("prep_src")) {
    // keep step 2 pointed at step 1's output; that is the whole point of
    // asking for the folder once
    $("prep_src").value = $("collectDir").value;
    checkPrepSrc();
    savePrep();
  }
  saveCollect();
});

/* keys are written straight to config.json and never read back out */
function saveKeys() {
  const patch = {};
  if ($("pexels_key").value) patch.pexels_key = $("pexels_key").value;
  if ($("pixabay_key").value) patch.pixabay_key = $("pixabay_key").value;
  if (!Object.keys(patch).length) return Promise.resolve();
  return api("/api/config", patch).then(() => {
    $("pexels_key").placeholder = patch.pexels_key ? "Pexels key saved" : "Pexels key";
    $("pixabay_key").placeholder = patch.pixabay_key ? "Pixabay key saved" : "Pixabay key";
    $("pexels_key").value = ""; $("pixabay_key").value = "";
  });
}
["pexels_key", "pixabay_key"].forEach(id =>
  $(id).addEventListener("change", () => saveKeys().catch(e => toast(e.message, true))));

async function startJob(slot, opts) {
  try {
    await saveKeys();
    const st = await api(`/api/job/${slot}/start`, opts || {});
    JOB_SLOTS[slotKeyFor(slot)] = slot;
    renderJob(stripFor(slotKeyFor(slot)), st);
    pollJobs();
  } catch (e) { toast(e.message, true); }
}
function slotKeyFor(slot) {
  return (slot === "fetch_videos" || slot === "fetch_images") ? "fetch" : slot;
}

$("depsBtn").addEventListener("click", () => startJob("deps"));
$("compileSupportBtn").addEventListener("click", () => startJob("compile_support"));
$("probeBtn").addEventListener("click", () => startJob("probe"));
$("startBtn").addEventListener("click", () => {
  const o = readCollect();
  if (!o.download_dir) { toast("Pick a folder to save into first.", true); return; }
  if (o.media === "video" && !o.src_pexels && !o.src_pixabay) {
    toast("Tick at least one source.", true); return;
  }
  $("nextHint").textContent = "";
  startJob(currentFetchJob(), o);
});
$("stopBtn").addEventListener("click", async () => {
  try { await api(`/api/job/${currentFetchJob()}/stop`, {}); } catch (e) { toast(e.message, true); }
  pollJobs();
});

async function inspectDir() {
  const p = $("collectDir").value.trim();
  const el = $("dirStatus");
  $("cleanDir").classList.add("hidden");
  if (!p) { setStatus(el, "Type or pick a folder first.", "bad"); return; }
  setStatus(el, "checking...");
  try {
    const d = await api("/api/inspect", { path: p });
    if (!d.exists) { setStatus(el, "That folder does not exist yet. It will be created.", "bad"); return; }
    const parts = [];
    if (d.videos) parts.push(`${d.videos} video${d.videos === 1 ? "" : "s"}`);
    if (d.images) parts.push(`${d.images} image${d.images === 1 ? "" : "s"}`);
    let msg = parts.length
      ? `${parts.join(", ")}, ${human(d.bytes)} on disk.`
      : "Folder is empty of videos and images.";
    let kind = parts.length ? "ok" : "";
    // Subfolders are searched too, so say so rather than leaving you to
    // wonder whether the clips you dropped in one are counted.
    if (d.nested) {
      msg += `  ${d.nested} of those ${d.nested === 1 ? "is" : "are"} in `
           + `${d.folders} subfolder${d.folders === 1 ? "" : "s"}, which `
           + `${d.nested === 1 ? "is" : "are"} used as well.`;
    }
    if (d.empty) {
      msg += `  ${d.empty} file${d.empty === 1 ? " is" : "s are"} empty or `
           + `half-downloaded and cannot be used.`;
      kind = "bad";
      $("cleanDir").classList.remove("hidden");
    }
    setStatus(el, msg, kind);
  } catch (e) { setStatus(el, e.message, "bad"); }
}
$("inspectDir").addEventListener("click", inspectDir);

$("cleanDir").addEventListener("click", async () => {
  const p = $("collectDir").value.trim();
  try {
    const d = await api("/api/clean", { path: p });
    toast(d.removed
      ? `Removed ${d.removed} unusable file${d.removed === 1 ? "" : "s"}.`
      : "Nothing to remove.");
    if (d.failed) toast(`${d.failed} could not be deleted \u2014 still open somewhere?`, true);
    inspectDir();
  } catch (e) { toast(e.message, true); }
});


/* ------------------------------------------------------------- page 2 */

const PREP_FIELDS = ["prep_src", "prep_out", "prep_seq", "prep_stride", "prep_clips",
  "prep_skip", "prep_start_seconds", "prep_fps", "prep_size", "prep_eval_ratio",
  "prep_limit", "prep_div_lo", "prep_div_hi", "prep_conv_lo", "prep_conv_hi",
  "prep_min_width", "prep_frame_width", "prep_min_detail", "prep_min_mask",
  "prep_mapper", "prep_png_level", "prep_no_mirror", "prep_model_type",
  "prep_resolution", "prep_gpu", "prep_depth_batch", "prep_mask_batch",
  "prep_workers", "prep_depth_aa", "prep_no_refine", "prep_overwrite",
  "prep_num_samples", "prep_rotate_prob", "prep_min_parallax"];

function startMode() {
  const r = document.querySelector('input[name="startmode"]:checked');
  return r ? r.value : "0";
}

function readPrep() {
  const o = {};
  PREP_FIELDS.forEach(id => {
    const el = $(id);
    if (!el) return;
    o[id] = el.type === "checkbox" ? el.checked
      : el.type === "number" ? (el.value === "" ? "" : +el.value) : el.value;
  });
  o.prep_start_mode = startMode();
  o.media = mediaType() === "image" ? "image" : "video";
  return o;
}
function writePrep(o) {
  if (!o) return;
  PREP_FIELDS.forEach(id => {
    const el = $(id);
    if (!el || o[id] === undefined) return;
    if (el.type === "checkbox") el.checked = !!o[id]; else el.value = o[id];
  });
  if (o.prep_start_mode) {
    const r = document.querySelector(`input[name="startmode"][value="${o.prep_start_mode}"]`);
    if (r) r.checked = true;
  }
}

/* Only the v3 models and Depth Anything v2 have an edge-smoothing model behind
   them (iw3's AA_SUPPORTED_MODELS). Asking for it on a v1 model is harmless --
   iw3 quietly leaves it off -- but the checkbox should not claim otherwise. */
const DEPTH_AA_MODELS = ["Any_V3_Mono", "Any_V3_Mono_01"];

const MODEL_NOTES = {
  Any_V3_Mono: "Best edges and the most reliable sky handling. Slowest.",
  Any_V3_Mono_01: "Same model, alternative output scaling.",
  Any_L: "Depth Anything v1, large. Noticeably faster than v3.",
  Any_B: "Depth Anything v1, base. What nunif's own inpaint training uses by default \u2014 a good balance.",
  Any_S: "Depth Anything v1, small. Fastest by far; edges are softer, so holes are less precise.",
};

function applyPrepMode() {
  const image = mediaType() === "image";
  $("prepVideoPanel").classList.toggle("hidden", image);
  $("prepImagePanel").classList.toggle("hidden", !image);
  $("startSecondsRow").classList.toggle("hidden", startMode() !== "seconds");

  const model = $("prep_model_type").value;
  const aa = DEPTH_AA_MODELS.includes(model);
  const box = $("prep_depth_aa");
  box.disabled = !aa;
  let note = MODEL_NOTES[model] || "";
  if (!aa) note += "  Smooth depth edges is not available for this model.";
  $("modelHint").textContent = note;

  // Turning refinement off is the one prep setting that quietly degrades every
  // mask in the dataset and cannot be undone afterwards, so it says so.
  const warn = $("prepRefineWarn");
  if (warn) warn.hidden = !$("prep_no_refine").checked;
}

let prepSaveTimer = null;
function savePrep() {
  clearTimeout(prepSaveTimer);
  prepSaveTimer = setTimeout(() => {
    const o = readPrep();
    const patch = { prep: o };
    if (o.prep_out) patch.dataset_dir = o.prep_out;
    api("/api/config", patch).catch(() => {});
  }, 400);
}
document.addEventListener("change", e => {
  if (!e.target.closest("#page2")) return;
  if (e.target.id === "prep_out" && $("train_data")) {
    $("train_data").value = $("prep_out").value;
    checkTrainData();
    saveTrain();
  }
  applyPrepMode();
  savePrep();
});

async function checkPrepSrc() {
  const el = $("prepSrcStatus");
  const p = $("prep_src").value.trim();
  if (!p) { setStatus(el, ""); return; }
  try {
    const d = await api("/api/inspect", { path: p });
    if (!d.exists) { setStatus(el, "That folder does not exist.", "bad"); return; }
    const image = mediaType() === "image";
    const n = image ? d.images : d.videos;
    const word = image ? "image" : "video";
    if (!n) {
      setStatus(el, `No ${word}s found there. Go back to step 1, or point this at your own folder.`, "bad");
    } else {
      let msg = `${n} ${word}${n === 1 ? "" : "s"} ready, ${human(d.bytes)} on disk.`;
      if (d.empty) msg += `  ${d.empty} unusable file(s) will be ignored.`;
      setStatus(el, msg, "ok");
    }
  } catch (e) { setStatus(el, e.message, "bad"); }
}
$("prep_src").addEventListener("change", checkPrepSrc);

$("scanBtn").addEventListener("click", () => startJob("scan", readPrep()));
$("prepBtn").addEventListener("click", () => {
  const o = readPrep();
  if (!o.prep_src) { toast("Pick the folder your footage is in.", true); return; }
  if (!o.prep_out) { toast("Pick where the dataset should be saved.", true); return; }
  if (o.prep_src === o.prep_out) {
    toast("The dataset must go somewhere other than the footage folder.", true); return;
  }
  $("prepHint").textContent = "";
  startJob("prep", o);
});
$("prepStop").addEventListener("click", async () => {
  try { await api("/api/job/prep/stop", {}); } catch (e) { toast(e.message, true); }
  pollJobs();
});


/* ------------------------------------------------------------- page 3 */

const TRAIN_FIELDS = ["train_data", "train_name", "train_epochs", "train_runs",
  "train_lr_reset", "train_num_samples", "train_arch", "train_crop", "train_batch",
  "train_backward", "train_eval_step", "train_eval_samples", "train_previews",
  "train_preview_rows", "train_keep_step", "train_optimizer", "train_lr",
  "train_loss", "train_workers", "train_gpu", "train_resume", "train_disable_hard",
  "train_ema", "train_ema_decay", "train_channels_last", "train_compile",
  "train_discriminator", "train_disc_weight", "train_disc_warmup", "train_disc_start",
  "train_lpips_frames",
  "train_amp_float", "train_overflow"];

function keepMode() {
  const r = document.querySelector('input[name="keep"]:checked');
  return r ? r.value : "best";
}

function readTrain() {
  const o = {};
  TRAIN_FIELDS.forEach(id => {
    const el = $(id);
    if (!el) return;
    o[id] = el.type === "checkbox" ? el.checked
      : el.type === "number" ? (el.value === "" ? "" : +el.value) : el.value;
  });
  o.train_keep = keepMode();
  o.media = mediaType() === "image" ? "image" : "video";
  return o;
}
function writeTrain(o) {
  if (!o) return;
  TRAIN_FIELDS.forEach(id => {
    const el = $(id);
    if (!el || o[id] === undefined) return;
    if (el.type === "checkbox") el.checked = !!o[id]; else el.value = o[id];
  });
  if (o.train_keep) {
    const r = document.querySelector(`input[name="keep"][value="${o.train_keep}"]`);
    if (r) r.checked = true;
  }
}

function applyTrainMode() {
  $("keepStepRow").classList.toggle("hidden", keepMode() !== "every");
  const epochs = +$("train_epochs").value || 0;
  const runs = +$("train_runs").value || 1;
  const reset = +$("train_lr_reset").value || 0;
  const cycles = reset ? Math.max(1, Math.round(epochs / reset)) : 1;
  const crop = +$("train_crop").value || 0;
  let plan = `${runs} run${runs === 1 ? "" : "s"} of ${epochs} epochs = `
    + `${runs * epochs} epochs in total, with ${cycles} learning rate reset`
    + `${cycles === 1 ? "" : "s"} inside each run.`;
  if (crop % 128 !== 0) {
    // JS % keeps the sign of the left operand, so (-300) % 128 is -44 and
    // "round up to the next multiple" has to be written the other way round
    const up = Math.ceil(crop / 128) * 128;
    plan += `  Crop ${crop} is padded to ${up} internally, so it costs as much as ${up}`
          + ` but trains on less \u2014 use ${up - 128} or ${up}.`;
  }
  const critic = $("train_discriminator").value;
  if (critic) {
    // Clamped the same way the server does, and said out loud: asking for the
    // critic at a run this chain never reaches used to look like it was on.
    const want = +$("train_disc_start").value || 1;
    const at = Math.min(Math.max(1, want), runs);
    plan += `  The critic joins at run ${at}`
          + (at !== want ? ` (moved from ${want} — there are only ${runs})` : "")
          + `, so runs 1–${at - 1 || 0} train on reconstruction loss alone.`
              .replace("runs 1–0 train on reconstruction loss alone.",
                       "it is on for the whole chain.");
  }
  $("trainPlan").textContent = plan;

  const dw = $("trainDiscWarn");
  if (dw) dw.hidden = !critic;
}

let trainSaveTimer = null;
function saveTrain() {
  clearTimeout(trainSaveTimer);
  trainSaveTimer = setTimeout(() => {
    api("/api/config", { train: readTrain() }).catch(() => {});
  }, 400);
}
document.addEventListener("change", e => {
  if (e.target.closest("#page3")) { applyTrainMode(); saveTrain(); }
});
["train_epochs", "train_runs", "train_lr_reset", "train_crop"].forEach(id =>
  $(id).addEventListener("input", applyTrainMode));

async function checkTrainData() {
  const el = $("trainDataStatus");
  const p = $("train_data").value.trim();
  if (!p) { setStatus(el, ""); return; }
  try {
    const d = await api("/api/dataset", { path: p });
    if (!d.exists) { setStatus(el, "That folder does not exist yet.", "bad"); return; }
    if (!d.train) {
      setStatus(el, "No train folder in there \u2014 finish step 2 first.", "bad"); return;
    }
    setStatus(el, `${d.train} training ${d.unit}, ${d.eval} held out for checking.`,
      d.eval ? "ok" : "bad");
    if (!d.eval) {
      setStatus(el, `${d.train} training ${d.unit}, but nothing held out for checking `
        + `\u2014 training will not be able to measure itself.`, "bad");
    }
  } catch (e) { setStatus(el, e.message, "bad"); }
}
$("train_data").addEventListener("change", checkTrainData);

$("trainBtn").addEventListener("click", () => {
  const o = readTrain();
  if (!o.train_data) { toast("Pick the dataset folder from step 2.", true); return; }
  if (!o.train_name.trim()) { toast("Give the model a name.", true); return; }
  $("trainHint").textContent = "";
  startJob("train", o);
});
$("trainStop").addEventListener("click", async () => {
  try { await api("/api/job/train/stop", {}); } catch (e) { toast(e.message, true); }
  pollJobs();
});
$("toLive").addEventListener("click", () => showPage(4));

/* The same setting appears on both pages; keep them in step so neither can
   quietly contradict the other. */
function syncResume(from) {
  const v = $(from).checked;
  $("train_resume").checked = v;
  $("live_resume").checked = v;
  $("resumeNote").textContent = v
    ? "will carry on from the last finished epoch"
    : "will start from the beginning";
  saveTrain();
}
$("train_resume").addEventListener("change", () => syncResume("train_resume"));
$("live_resume").addEventListener("change", () => syncResume("live_resume"));

/* Overflow is read by the RUNNING chain, not just at start, so it is written
   to config.json the moment it is ticked rather than waiting for the next
   Start. That is what lets it be turned on or off mid-run from the live page. */
function syncOverflow(from) {
  const v = $(from).checked;
  $("train_overflow").checked = v;
  $("live_overflow").checked = v;
  saveTrain();
  api("/api/config", { train_overflow: v }).catch(() => {});
}
$("train_overflow").addEventListener("change", () => syncOverflow("train_overflow"));
$("live_overflow").addEventListener("change", () => syncOverflow("live_overflow"));

$("liveStart").addEventListener("click", async () => {
  const cfg = (STATE && STATE.config && STATE.config.train) || {};
  const o = Object.assign({}, cfg, readTrain());
  if (!o.train_data || !(o.train_name || "").trim()) {
    toast("Set the dataset folder and a model name on the Train tab first.", true);
    showPage(3);
    return;
  }
  try {
    await api("/api/job/train/start", o);
    $("liveMsg").textContent = "";
    pollJobs(); pollLive(true);
  } catch (e) { toast(e.message, true); }
});
$("liveStop").addEventListener("click", async () => {
  try { await api("/api/job/train/stop", {}); } catch (e) { toast(e.message, true); }
  pollJobs(); pollLive(true);
});


/* ---------------------------------------------------- benchmark results */

/* bench_model.py prints a fixed-width table; these two shapes are the only
   lines that carry numbers, so parsing is reliable without changing the tool. */
const BENCH_ROW = /^(\S+)\s+([\d.]+)M\s+([\d.]+)\s+([\d.]+)\s+(\d+)MB\s*$/;
const BENCH_OOM = /^(\S+)\s+out of memory/;
const BENCH_DEV = /^device\s*:\s*(.+)$/;

function shortArch(name) {
  if (name.includes("light_")) return "iw3 stock (light)";
  const m = name.match(/nt_(?:video_)?inpaint_v2_?([sbl])?$/);
  const size = { s: "Small", b: "Base", l: "Large" }[m && m[1]] || name;
  return name.includes("video") ? `${size} (video)` : size;
}

function fmtDuration(seconds) {
  if (!isFinite(seconds) || seconds <= 0) return "?";
  if (seconds < 90) return `${Math.round(seconds)}s`;
  const m = seconds / 60;
  if (m < 90) return `${m.toFixed(0)} min`;
  const h = m / 60;
  return h < 48 ? `${h.toFixed(1)} h` : `${(h / 24).toFixed(1)} days`;
}

function renderBench(lines) {
  const box = $("benchResult");
  let device = "", totalGB = 0;
  const rows = [];
  (lines || []).forEach(raw => {
    const l = raw.replace(/\s+$/, "");
    const d = l.match(BENCH_DEV);
    if (d) {
      device = d[1].trim();
      const g = device.match(/\(([\d.]+)\s*GB\)/);
      if (g) totalGB = +g[1];
      return;
    }
    const m = l.match(BENCH_ROW);
    if (m) { rows.push({ arch: m[1], params: +m[2], sec: +m[3], rate: +m[4], vram: +m[5] }); return; }
    const o = l.match(BENCH_OOM);
    if (o) rows.push({ arch: o[1], oom: true });
  });
  if (!rows.length) { box.classList.add("hidden"); box.innerHTML = ""; return; }

  const isPick = r => r.arch.replace("video_", "") === $("train_arch").value;
  const perEpoch = +$("train_num_samples").value || 0;
  const runs = +$("train_runs").value || 1;
  const epochs = +$("train_epochs").value || 0;

  const body = rows.map(r => {
    if (r.oom) {
      return `<tr class="oom"><td>${shortArch(r.arch)}</td><td colspan="4">`
           + `does not fit at this crop and batch</td></tr>`;
    }
    const epochSecs = perEpoch && r.rate ? perEpoch / r.rate : 0;
    const share = totalGB ? (r.vram / 1024) / totalGB : 0;
    let vram = `${(r.vram / 1024).toFixed(1)} GB`;
    if (totalGB) vram += `<span class="tag">${Math.round(share * 100)}%</span>`;
    return `<tr class="${isPick(r) ? "pick" : ""}">
      <td>${shortArch(r.arch)}${isPick(r) ? '<span class="tag">your choice</span>' : ""}</td>
      <td>${r.params.toFixed(1)}M</td>
      <td>${r.sec.toFixed(3)}s</td>
      <td>${vram}</td>
      <td>${epochSecs ? fmtDuration(epochSecs) : "&ndash;"}</td></tr>`;
  }).join("");

  const mine = rows.find(r => isPick(r) && !r.oom);
  let verdict = "";
  if (mine) {
    const epochSecs = perEpoch / mine.rate;
    const share = totalGB ? (mine.vram / 1024) / totalGB : 0;
    const head = share > 0.92 ? "Very close to the memory limit \u2014 a longer clip or a "
                              + "bigger crop will fail. Drop the crop or the batch."
               : share > 0.8 ? "Tight on memory but workable. Close other programs using "
                             + "the card."
               : "Comfortable on memory.";
    verdict = `<div class="verdict"><b>${head}</b> At ${perEpoch.toLocaleString()} samples `
      + `per epoch that is about <b>${fmtDuration(epochSecs)}</b> an epoch, so `
      + `<b>${fmtDuration(epochSecs * epochs)}</b> for one run of ${epochs} and `
      + `<b>${fmtDuration(epochSecs * epochs * runs)}</b> for all ${runs}.</div>`;
  } else if (rows.some(r => r.oom && isPick(r))) {
    verdict = `<div class="verdict"><b>Your chosen size does not fit.</b> Lower the crop `
      + `size, or pick a smaller model.</div>`;
  }

  box.classList.remove("hidden");
  box.innerHTML = `<div class="bench">
    ${device ? `<div class="dev">${device}</div>` : ""}
    <table><thead><tr><th>model</th><th>params</th><th>per step</th>
      <th>peak memory</th><th>per epoch</th></tr></thead>
      <tbody>${body}</tbody></table>
    ${verdict}</div>`;
}

$("benchBtn").addEventListener("click", () => {
  const o = readTrain();
  o.bench_all = $("bench_all").checked;
  $("benchResult").classList.add("hidden");
  $("benchHint").textContent = "";
  startJob("bench", o);
});


/* ------------------------------------------------------------- page 4 */

let LIVE = null;            // last /api/progress
let liveRun = null;
let liveEpoch = null;
let liveSlot = "all";
// Which samples the preview grid is currently built for, and the shape of a
// preview image. Both exist so changing epoch never rebuilds or resizes the
// grid -- see drawPreview.
let previewShape = "";
let previewRatio = "";
let previewCacheMax = 48;
let hoverX = null;          // pixel x of the cursor over the chart
let chartGeom = null;       // {x0,x1,pad,W,H,scale} from the last draw

function fmtLoss(v, d) {
  return (v === null || v === undefined) ? "–" : (+v).toFixed(d === undefined ? 4 : d);
}

/* Train and eval are the same quantity measured two ways, so they go on ONE
   scale. Two independent axes make the lines look interchangeable and hide the
   thing you actually want to see -- whether eval is tracking train or drifting
   away from it. */
function drawChart() {
  const c = $("chart"), dpr = devicePixelRatio || 1;
  const W = c.clientWidth, H = c.clientHeight;
  if (!W || !H) return;
  c.width = W * dpr; c.height = H * dpr;
  const g = c.getContext("2d");
  g.scale(dpr, dpr); g.clearRect(0, 0, W, H);
  const rows = (LIVE && LIVE.rows) || [];
  const tr = rows.filter(r => r.train !== null);
  const ev = rows.filter(r => r.eval !== null);
  if (!tr.length && !ev.length) { chartGeom = null; return; }

  const css = n => getComputedStyle(document.body).getPropertyValue(n).trim();
  const pad = { l: 64, r: 16, t: 14, b: 28 };
  const xs = rows.map(r => r.epoch);
  const x0 = Math.min(...xs), x1 = Math.max(...xs, x0 + 1);
  const X = e => pad.l + (e - x0) / (x1 - x0) * (W - pad.l - pad.r);

  const all = tr.map(r => r.train).concat(ev.map(r => r.eval));
  let lo = Math.min(...all), hi = Math.max(...all);
  if (hi === lo) { const d = Math.abs(hi) * 0.05 || 1e-6; lo -= d; hi += d; }
  const m = (hi - lo) * 0.08;
  lo -= m; hi += m;
  const Y = v => H - pad.b - (v - lo) / (hi - lo) * (H - pad.t - pad.b);

  g.font = "11px ui-sans-serif,system-ui,sans-serif";
  g.strokeStyle = css("--line"); g.lineWidth = 1;
  g.fillStyle = css("--faint"); g.textAlign = "right";
  for (let i = 0; i <= 4; i++) {
    const y = Math.round(H - pad.b - (H - pad.t - pad.b) * i / 4) + 0.5;
    g.beginPath(); g.moveTo(pad.l, y); g.lineTo(W - pad.r, y); g.stroke();
    g.fillText((lo + (hi - lo) * i / 4).toPrecision(3), pad.l - 8, y + 4);
  }
  g.textAlign = "left"; g.fillText("epoch " + x0, pad.l, H - 8);
  g.textAlign = "right"; g.fillText("epoch " + x1, W - pad.r, H - 8);
  g.textAlign = "left";

  // phase dividers
  let prev = rows.length ? rows[0].phase : 1;
  for (const r of rows) {
    if (r.phase > prev) {
      const x = Math.round(X(r.epoch)) + 0.5;
      g.save(); g.setLineDash([4, 4]); g.strokeStyle = css("--faint");
      g.globalAlpha = 0.5; g.beginPath(); g.moveTo(x, pad.t); g.lineTo(x, H - pad.b);
      g.stroke(); g.restore();
      g.fillStyle = css("--faint"); g.fillText("run " + r.phase, x + 4, pad.t + 11);
      prev = r.phase;
    }
  }

  // lines only -- no dots. At 600 epochs the markers merge into a smear and
  // hide the shape of the curve, which is the whole point of the graph.
  const line = (pts, color) => {
    if (!pts.length) return;
    g.strokeStyle = color; g.lineWidth = 1.8; g.lineJoin = "round";
    g.beginPath();
    pts.forEach((p, i) => i ? g.lineTo(X(p[0]), Y(p[1])) : g.moveTo(X(p[0]), Y(p[1])));
    g.stroke();
  };
  line(tr.map(r => [r.epoch, r.train]), css("--accent"));
  line(ev.map(r => [r.epoch, r.eval]), css("--accent2"));

  chartGeom = { x0, x1, pad, W, H, lo, hi, X, Y };
  if (hoverX !== null) drawHover(g, css);
}

function nearestRow(epoch) {
  const rows = (LIVE && LIVE.rows) || [];
  let best = null, bd = Infinity;
  for (const r of rows) {
    const d = Math.abs(r.epoch - epoch);
    if (d < bd) { bd = d; best = r; }
  }
  return best;
}
function lastEvalAtOrBefore(epoch) {
  const rows = (LIVE && LIVE.rows) || [];
  let out = null;
  for (const r of rows) {
    if (r.epoch > epoch) break;
    if (r.eval !== null) out = r;
  }
  return out;
}

function drawHover(g, css) {
  const G = chartGeom;
  if (!G) return;
  const frac = (hoverX - G.pad.l) / (G.W - G.pad.l - G.pad.r);
  if (frac < -0.02 || frac > 1.02) return;
  const epoch = Math.round(G.x0 + frac * (G.x1 - G.x0));
  const row = nearestRow(epoch);
  if (!row) return;
  const x = Math.round(G.X(row.epoch)) + 0.5;

  g.save();
  g.strokeStyle = css("--faint"); g.globalAlpha = 0.8; g.lineWidth = 1;
  g.beginPath(); g.moveTo(x, G.pad.t); g.lineTo(x, G.H - G.pad.b); g.stroke();
  g.globalAlpha = 1;
  // eval only exists on check epochs, so carry the most recent one forward
  // rather than showing a dash for three quarters of the graph
  const ev = row.eval !== null ? row : lastEvalAtOrBefore(row.epoch);
  if (row.train !== null) {
    g.fillStyle = css("--accent");
    g.beginPath(); g.arc(x, G.Y(row.train), 3.5, 0, 7); g.fill();
  }
  if (ev) {
    g.fillStyle = css("--accent2");
    g.beginPath(); g.arc(G.X(ev.epoch), G.Y(ev.eval), 3.5, 0, 7); g.fill();
  }

  const lines = [`epoch ${row.epoch}`];
  if (row.train !== null) lines.push(`train  ${fmtLoss(row.train)}`);
  if (ev) lines.push(`eval   ${fmtLoss(ev.eval)}`
    + (ev.epoch !== row.epoch ? `  (epoch ${ev.epoch})` : ""));
  if (row.lr) lines.push(`lr     ${(+row.lr).toExponential(2)}`);

  g.font = "11.5px ui-monospace,Consolas,monospace";
  const wid = Math.max(...lines.map(t => g.measureText(t).width)) + 18;
  const hei = lines.length * 16 + 12;
  let bx = x + 12, by = G.pad.t + 8;
  if (bx + wid > G.W - G.pad.r) bx = x - 12 - wid;
  g.fillStyle = css("--tip-bg"); g.strokeStyle = css("--faint");
  g.beginPath(); g.roundRect(bx, by, wid, hei, 7); g.fill(); g.stroke();
  g.fillStyle = css("--fg");
  lines.forEach((t, i) => g.fillText(t, bx + 9, by + 20 + i * 16));
  g.restore();
}

$("chart").addEventListener("mousemove", e => {
  const r = $("chart").getBoundingClientRect();
  hoverX = e.clientX - r.left;
  drawChart();
});
$("chart").addEventListener("mouseleave", () => { hoverX = null; drawChart(); });
addEventListener("resize", () => drawChart());

function drawLiveStats() {
  const rows = (LIVE && LIVE.rows) || [];
  const last = rows[rows.length - 1];
  const evs = rows.filter(r => r.eval !== null);
  const best = evs.length ? evs.reduce((a, b) => b.eval < a.eval ? b : a) : null;
  const secs = rows.filter(r => r.seconds).slice(-5).map(r => r.seconds);
  const avg = secs.length ? secs.reduce((a, b) => a + b, 0) / secs.length : null;
  $("liveStats").innerHTML = !last ? "" : `
    <div class="stat"><b>${last.epoch}</b><span>epoch${last.phase > 1 ? " (run " + last.phase + ")" : ""}</span></div>
    <div class="stat"><b>${fmtLoss(last.train)}</b><span>train loss</span></div>
    <div class="stat"><b>${fmtLoss(evs.length ? evs[evs.length - 1].eval : null)}</b><span>eval loss</span></div>
    <div class="stat"><b>${best ? fmtLoss(best.eval) : "–"}</b><span>best eval${best ? " (epoch " + best.epoch + ")" : ""}</span></div>
    <div class="stat"><b>${last.lr !== null ? (+last.lr).toExponential(2) : "–"}</b><span>learning rate</span></div>
    <div class="stat"><b>${avg ? fmtDuration(avg) : "–"}</b><span>per epoch</span></div>`;

  const job = (LIVE && LIVE.job) || {};
  drawPace(job);
  drawLiveButtons(job);
  const el = $("liveEta");
  if (job.running) {
    const parts = [];
    if (job.done && job.total) parts.push(`epoch <b>${job.done}</b> of <b>${job.total}</b>`);
    // The server labels each step with the phase the trainer is really in.
    // Counting the chain's own steps here said "run 1 of 3" after a restart
    // that was actually phase 2, and hid that three more phases were queued.
    if (job.steplabel && /^run /.test(job.steplabel)) parts.push(job.steplabel);
    const leftEpochs = (job.total || 0) - (job.done || 0);
    if (job.eta) parts.push(`about <b>${job.eta}</b> left`);
    else if (avg && leftEpochs > 0) {
      parts.push(`about <b>${fmtDuration(leftEpochs * avg)}</b> left`);
    } else if (leftEpochs <= 0 && job.total) {
      parts.push("finishing up");
    }
    el.innerHTML = "Training — " + parts.join(" · ");
  } else if (avg && rows.length) {
    el.innerHTML = `Not running. Last update ${last ? last.timestamp || "" : ""}.`;
  } else {
    el.innerHTML = "";
  }
}

/* What is happening inside the current epoch. An epoch is minutes long, so
   without this the page looks frozen between the once-per-epoch updates. */
function drawPace(job) {
  const box = $("livePace");
  const sub = job.running ? (job.sub || {}) : null;
  if (!sub || !sub.total) { box.classList.add("hidden"); return; }
  box.classList.remove("hidden");
  const pct = sub.percent || 0;
  const what = sub.stage === "eval" ? "Checking against the eval set"
             : sub.stage === "train" ? "Training"
             : "Working";
  $("paceWhat").textContent = `${what}${sub.epoch ? " \u2014 epoch " + sub.epoch : ""}`
    + `  ${pct.toFixed(0)}%`;
  const bits = [`${sub.n} of ${sub.total}`];
  if (sub.rate) bits.push(sub.rate);
  if (sub.left && sub.left !== "?") bits.push(`${sub.left} left`);
  $("paceNums").textContent = bits.join("  \u00b7  ");
  $("paceBar").style.width = pct + "%";
}

function drawLiveButtons(job) {
  const running = !!job.running;
  $("liveStart").classList.toggle("hidden", running);
  $("liveStop").classList.toggle("hidden", !running);
  const msg = $("liveMsg");
  if (running) {
    msg.textContent = "";
  } else if (job.any) {
    // the train slot is busy with a different model
    msg.textContent = `Another run is training: ${job.other}`;
  } else {
    msg.textContent = "";
  }
}

function drawPreview() {
  const eps = (LIVE && LIVE.preview_epochs) || [];
  const r = $("epochRange");
  r.min = 0; r.max = Math.max(0, eps.length - 1);
  if ($("follow").checked && eps.length) liveEpoch = eps[eps.length - 1];
  let idx = eps.indexOf(liveEpoch);
  if (idx < 0) { idx = eps.length - 1; liveEpoch = eps[idx]; }
  r.value = idx < 0 ? 0 : idx;
  $("epochLabel").textContent = (liveEpoch === null || liveEpoch === undefined) ? "-" : liveEpoch;

  const sel = $("slot"), slots = (LIVE && LIVE.slots) || [];
  if (sel.options.length !== slots.length + 1) {
    sel.innerHTML = `<option value="all">all</option>`
      + slots.map(s => `<option value="${s}">${s}</option>`).join("");
    sel.value = String(liveSlot);
  }
  const files = (LIVE && LIVE.previews[String(liveEpoch)]) || {};
  let keys = Object.keys(files).map(Number).sort((a, b) => a - b);
  if (liveSlot !== "all" && files[String(liveSlot)] !== undefined) keys = [+liveSlot];
  $("previewCount").textContent = keys.length
    ? `${keys.length} sample${keys.length > 1 ? "s" : ""}` : "";
  const run = encodeURIComponent(liveRun || "");
  const box = $("preview");
  if (!keys.length) {
    previewShape = "";
    box.innerHTML = `<div class="empty">No preview images yet. They appear after the
       first check of the eval set.</div>`;
    return;
  }
  const srcs = keys.map(k => `/api/preview/${run}/${encodeURIComponent(files[String(k)])}`);

  // Dragging the slider only changes which epoch the same samples show. Tearing
  // the markup down and writing it again gave every <img> an empty src for a
  // frame or two: the boxes collapsed to nothing, the page reflowed, and the
  // chart above jumped down into the gap and back. That was the flash.
  const shape = keys.join(",");
  if (shape !== previewShape) {
    previewShape = shape;
    // Two layers per sample. Only one is ever shown, and the new epoch is
    // decoded into the hidden one before they swap, so what you are looking at
    // is never replaced by an empty box -- that empty box, painted over the
    // black image background, was the flash between epochs.
    const ar = previewRatio ? ` style="aspect-ratio:${previewRatio}"` : "";
    // The observer only reports on the next frame, so seed the first screenful
    // as visible; it corrects anything wrong a frame later.
    box.innerHTML = keys.map(k =>
        `<div class="shot"><div class="cap">sample ${k}</div>`
      + `<img alt="epoch ${liveEpoch} sample ${k}" decoding="async"${ar}>`
      + `<img alt="" decoding="async" hidden${ar}></div>`).join("");
    box.querySelectorAll(".shot").forEach((sh, i) => {
      if (i < 12) sh.dataset.vis = "1";
    });
  }
  // Only the tiles you can actually see are swapped. With 256 previews an
  // epoch, doing all of them turned one nudge of the slider into 256 fetches
  // and 256 PNG decodes, which is the second or so of lag -- and the ones off
  // screen were decoded for nothing. The observer swaps the rest as they scroll
  // in, so the picture under the cursor changes immediately.
  const shots = box.querySelectorAll(".shot");
  const observer = previewObserver();
  shots.forEach((shot, i) => {
    shot.dataset.src = srcs[i];
    shot.dataset.want_alt = `epoch ${liveEpoch} sample ${keys[i]}`;
    if (!shot.dataset.watched) {
      shot.dataset.watched = "1";
      observer.observe(shot);
    }
    if (shot.dataset.vis === "1") applyShot(shot);
  });
  prefetchNeighbours(keys, files, run, box);
}

/* Which tiles are on screen. rootMargin keeps a screenful either side warm, so
   a small scroll never shows an empty box. */
let _previewObserver = null;
function previewObserver() {
  if (_previewObserver) return _previewObserver;
  _previewObserver = new IntersectionObserver(entries => {
    for (const e of entries) {
      e.target.dataset.vis = e.isIntersecting ? "1" : "0";
      if (e.isIntersecting) applyShot(e.target);
    }
  }, { rootMargin: "400px 0px" });
  return _previewObserver;
}

function applyShot(shot) {
  if (shot.dataset.src) swapPreview(shot, shot.dataset.src, shot.dataset.want_alt || "");
}

/* Decoded frames, kept alive so scrubbing back over an epoch you have already
   seen is instant. The browser cache alone is not enough: it can evict, and a
   cache hit still has to decode the PNG again. Held by src, oldest dropped. */
const previewCache = new Map();

function cacheImage(src) {
  let im = previewCache.get(src);
  if (im) {                                   // refresh its place in the queue
    previewCache.delete(src);
    previewCache.set(src, im);
    return im;
  }
  im = new Image();
  im.decoding = "async";
  im.src = src;
  previewCache.set(src, im);
  while (previewCache.size > previewCacheMax) {
    previewCache.delete(previewCache.keys().next().value);
  }
  return im;
}

/* Warm the epochs either side of this one, so stepping through the slider hits
   memory rather than the disk. */
function prefetchNeighbours(keys, files, run, box) {
  const eps = (LIVE && LIVE.preview_epochs) || [];
  const at = eps.indexOf(liveEpoch);
  if (at < 0) return;
  // Only the samples on screen, and only a few epochs either way. Warming all
  // 256 samples of four epochs is a thousand requests for pictures nobody is
  // looking at, and it evicts the ones they are.
  let want = keys;
  if (box) {
    const seen = [...box.querySelectorAll(".shot")]
      .filter(sh => sh.dataset.vis === "1")
      .map(sh => sh.dataset.want_alt)
      .map(a => +(String(a).split("sample ")[1] || NaN))
      .filter(n => !Number.isNaN(n));
    if (seen.length) want = seen;
  }
  want = want.slice(0, 12);
  previewCacheMax = Math.max(24, want.length * 6);
  for (const near of [eps[at + 1], eps[at - 1], eps[at + 2], eps[at - 2]]) {
    const f = (LIVE && LIVE.previews[String(near)]) || null;
    if (!f) continue;
    for (const k of want) {
      if (f[String(k)] !== undefined) {
        cacheImage(`/api/preview/${run}/${encodeURIComponent(f[String(k)])}`);
      }
    }
  }
}

function swapPreview(shot, src, alt) {
  if (shot.dataset.want === src) return;
  shot.dataset.want = src;
  const imgs = shot.querySelectorAll("img");
  const shown = imgs[0].hidden ? imgs[1] : imgs[0];
  const spare = shown === imgs[0] ? imgs[1] : imgs[0];
  const cached = cacheImage(src);             // keep the bitmap around
  if (spare.getAttribute("src") !== src) spare.src = src;
  const reveal = () => {
    if (shot.dataset.want !== src) return;    // a later drag already won
    if (spare.naturalWidth && spare.naturalHeight) {
      previewRatio = `${spare.naturalWidth} / ${spare.naturalHeight}`;
      spare.style.aspectRatio = shown.style.aspectRatio = previewRatio;
    }
    spare.alt = alt;
    spare.hidden = false;
    shown.hidden = true;
  };
  // Already decoded from an earlier visit or a prefetch: swap on this frame
  // rather than waiting a turn of the event loop for decode() to resolve.
  if ((spare.complete && spare.naturalWidth) || (cached.complete && cached.naturalWidth
      && spare.getAttribute("src") === src && spare.complete)) {
    reveal();
    return;
  }
  spare.decode().then(reveal).catch(() => {
    // decode() rejects if the src changed under us, which the guard handles,
    // or if the file really is missing -- say so rather than sitting blank.
    if (shot.dataset.want === src && !spare.complete) shown.alt = "preview missing";
  });
}

$("epochRange").addEventListener("input", e => {
  const eps = (LIVE && LIVE.preview_epochs) || [];
  $("follow").checked = false;
  liveEpoch = eps[+e.target.value];
  drawPreview(); drawChart();
});
$("slot").addEventListener("change", e => {
  liveSlot = e.target.value === "all" ? "all" : +e.target.value;
  drawPreview();
});
$("follow").addEventListener("change", () => { drawPreview(); });
$("liveRun").addEventListener("change", e => {
  liveRun = e.target.value; LIVE = null; liveEpoch = null;
  api("/api/config", { live_run: liveRun }).catch(() => {});
  pollLive(true);
});

async function refreshRuns() {
  try {
    const d = await api("/api/runs", {});
    const names = (d.runs || []).map(r => r.name);
    const sel = $("liveRun");
    if (sel.options.length !== names.length
        || names.some((n, i) => sel.options[i] && sel.options[i].value !== n)) {
      sel.innerHTML = names.map(n => `<option value="${n}">${n}</option>`).join("")
        || `<option value="">no runs yet</option>`;
    }
    if (!liveRun || !names.includes(liveRun)) liveRun = names[0] || null;
    if (liveRun) sel.value = liveRun;
  } catch (e) { /* server busy */ }
}

let livePoll = null;
async function pollLive(force) {
  clearTimeout(livePoll);
  const visible = !$("page4").classList.contains("hidden");
  if (visible || force) {
    await refreshRuns();
    if (liveRun) {
      try {
        const next = await api("/api/progress", { run: liveRun });
        const changed = !LIVE || JSON.stringify(next.rows) !== JSON.stringify(LIVE.rows)
          || JSON.stringify(next.preview_epochs) !== JSON.stringify(LIVE.preview_epochs)
          || JSON.stringify(next.job) !== JSON.stringify(LIVE.job);
        LIVE = next;
        if (changed || force) { drawLiveStats(); drawChart(); drawPreview(); }
      } catch (e) { /* mid-write */ }
    }
  }
  const running = LIVE && LIVE.job && LIVE.job.running;
  livePoll = setTimeout(pollLive, visible ? (running ? 4000 : 8000) : 20000);
}


/* ------------------------------------------------------------- page 5 */

let MODELS = null;

function pickMode() {
  const r = document.querySelector('input[name="pickmode"]:checked');
  return r ? r.value : "best";
}
function chosenModel() {
  if (pickMode() === "custom") return $("modelPath").value.trim();
  return $("modelPick").value || "";
}

function suggestName(fp, label) {
  const base = (label || fp.split(/[\\/]/).pop() || "").replace(/\.pth$/i, "");
  const clean = base.split("—")[0].trim().replace(/[^A-Za-z0-9_.\-]+/g, "_")
    .replace(/^_+|_+$/g, "");
  return clean || "my_model";
}

function renderModelInfo(info) {
  const box = $("modelInfo");
  if (!info || !info.exists) {
    box.classList.remove("hidden");
    box.innerHTML = `<div class="r warn"><span>File</span><b>not found</b></div>`;
    return false;
  }
  if (!info.nunif || !info.arch) {
    box.classList.remove("hidden");
    box.innerHTML = `<div class="r warn"><span>File</span><b>This is not a model
      saved by nunif — iw3 will not be able to load it.</b></div>`;
    return false;
  }
  box.classList.remove("hidden");
  box.innerHTML = `
    <div class="r"><span>Architecture</span><b>${info.arch}</b></div>
    <div class="r"><span>Works on</span><b>${info.kind === "video"
      ? "video (and is what iw3 uses for video files)"
      : "still images"}</b></div>
    <div class="r"><span>Size</span><b>${human(info.bytes)}</b></div>
    ${info.saved ? `<div class="r"><span>Trained until</span><b>${info.saved}</b></div>` : ""}
    <div class="r"><span>File</span><b>${info.path}</b></div>`;
  return true;
}

let lastInfo = null;
async function refreshModelInfo(setName) {
  const fp = chosenModel();
  lastInfo = null;
  if (!fp) { $("modelInfo").classList.add("hidden"); return; }
  const opt = $("modelPick").selectedOptions[0];
  const label = (pickMode() === "best" && opt) ? opt.dataset.label : "";
  try {
    const info = await api("/api/inspect_model", { path: fp });
    lastInfo = renderModelInfo(info) ? info : null;
    if (setName && lastInfo && !$("installName").value.trim()) {
      $("installName").value = suggestName(fp, label);
    }
    checkName();
  } catch (e) { toast(e.message, true); }
}

function checkName() {
  const n = $("installName").value.trim();
  const names = (MODELS && MODELS.yml && MODELS.yml.names) || [];
  if (!n) { $("nameHint").textContent = ""; return; }
  if (!/^[A-Za-z0-9_.\-]{1,64}$/.test(n)) {
    $("nameHint").textContent = "Letters, numbers, dot, dash and underscore only.";
    return;
  }
  $("nameHint").textContent = names.includes(n)
    ? `There is already an entry called “${n}” in iw3 — installing will replace it.`
    : `Will appear in iw3 as “${n}”.`;
}
$("installName").addEventListener("input", checkName);

function applyPickMode() {
  const custom = pickMode() === "custom";
  $("pickBestBox").classList.toggle("hidden", custom);
  $("pickCustomBox").classList.toggle("hidden", !custom);
}
$$('input[name="pickmode"]').forEach(r => r.addEventListener("change", () => {
  applyPickMode(); refreshModelInfo(true);
}));
$("modelPick").addEventListener("change", () => refreshModelInfo(true));
$("modelCheck").addEventListener("click", () => refreshModelInfo(true));

function renderInstallState(d) {
  const hook = d.hook || {}, yml = d.yml || {}, f = hook.features || {};
  const rows = [];
  const on = (b) => b ? "on" : "off";
  let state, good = false;
  if (!hook.installed) state = "not installed — iw3 cannot load these models yet";
  else if (!f.model) state = "installed without model support — press Install";
  else if (!hook.current) state = "older version — press Install to update it";
  else { state = "installed and up to date"; good = true; }
  rows.push(`<div class="r ${good ? "good" : "warn"}"><span>iw3 extras</span><b>${state}</b></div>`);
  if (hook.installed && Object.keys(f).length) {
    rows.push(`<div class="r"><span>Model support</span><b>${on(f.model)}`
      + ` (mask handling ${on(f.mask_patch)}, window reuse ${on(f.window)})</b></div>`);
    rows.push(`<div class="r"><span>Low-res inpainting</span><b>${on(f.lowres)}</b></div>`);
    rows.push(`<div class="r"><span>Screen-edge fix</span><b>${on(f.border)}</b></div>`);
  }
  if (hook.legacy)
    rows.push(`<div class="r warn"><span>Old hook</span><b>this GUI's earlier hook is still `
      + `installed and conflicts — press Install to replace it</b></div>`);
  if (hook.folder) rows.push(`<div class="r"><span>Installed to</span><b>${hook.folder}</b></div>`);
  if (hook.dir) rows.push(`<div class="r"><span>Hook location</span><b>${hook.dir}</b></div>`);
  rows.push(`<div class="r"><span>iw3 model list</span><b>${yml.path || "not found"}</b></div>`);
  rows.push(`<div class="r"><span>Entries in it</span><b>${
    (yml.names || []).length ? yml.names.join(", ") : "none yet"}</b></div>`);
  $("installState").innerHTML = rows.join("");
}

function renderChecks(checks) {
  const box = $("installChecks");
  if (!checks || !checks.length) { box.classList.add("hidden"); return; }
  box.innerHTML = checks.map(c => {
    const ok = c.startsWith("[ok]");
    return `<div class="r ${ok ? "good" : "warn"}"><span>${ok ? "ok" : "problem"}</span>`
      + `<b>${c.replace(/^\[(ok|!!)\]\s*/, "")}</b></div>`;
  }).join("");
  box.classList.remove("hidden");
}

function saveInstallOpts() {
  api("/api/config", { install_lowres: $("install_lowres").checked,
                       install_border: $("install_border").checked }).catch(() => {});
}
$("install_lowres").addEventListener("change", saveInstallOpts);
$("install_border").addEventListener("change", saveInstallOpts);

async function refreshModels() {
  try {
    MODELS = await api("/api/models", {});
  } catch (e) { return; }
  const sel = $("modelPick");
  const want = sel.value;
  sel.innerHTML = (MODELS.models || []).map(m =>
    `<option value="${m.path.replace(/"/g, "&quot;")}" data-label="${m.label}">`
    + `${m.label}  —  ${m.kind || "?"}, ${human(m.bytes)}</option>`).join("")
    || `<option value="">nothing trained yet</option>`;
  if (want && (MODELS.models || []).some(m => m.path === want)) sel.value = want;
  renderInstallState(MODELS);
  checkName();
}

$("installBtn").addEventListener("click", async () => {
  const fp = chosenModel();
  const name = $("installName").value.trim();
  if (!fp) { toast("Pick a model first.", true); return; }
  if (!lastInfo) { await refreshModelInfo(true); if (!lastInfo) {
    toast("That file cannot be used — see the note above.", true); return; } }
  if (!name) { toast("Give it a name for the iw3 list.", true); return; }
  setStatus($("installResult"), "installing, then checking it in a fresh iw3 process "
    + "(this loads the model, so give it a moment)...");
  renderChecks([]);
  $("installBtn").disabled = true;
  try {
    const d = await api("/api/install", {
      path: fp, name, kind: lastInfo.kind,
      lowres: $("install_lowres").checked, border: $("install_border").checked });
    setStatus($("installResult"), (d.ok ? "Done. " : "Installed, but the check found a problem. ")
      + (d.did || []).join("  ·  "), d.ok ? "ok" : "bad");
    renderChecks(d.checks);
    await refreshModels();
    $("installMsg").textContent = "Restart iw3 if it is open, then pick it in "
      + "the Inpainting Model box.";
  } catch (e) { setStatus($("installResult"), e.message, "bad"); }
  finally { $("installBtn").disabled = false; }
});
$("verifyBtn").addEventListener("click", () => startJob("verify", {
  name: $("installName").value.trim(), kind: lastInfo ? lastInfo.kind : "" }));

/* --------------------------------------------------------------- setup */

$("detectNunif").addEventListener("click", async () => {
  setStatus($("nunifStatus"), "looking...");
  try {
    const d = await api("/api/detect", {});
    if (d.nunif_dir) {
      $("nunifPath").value = d.nunif_dir;
      // Report what the check actually says, not a hard-coded "found it" --
      // auto-detect can land on a folder that is missing the python beside it.
      await checkNunifPath(true, true);
    } else {
      setStatus($("nunifStatus"),
        "Could not find it. Type the path, or use Browse.", "bad");
    }
  } catch (e) { setStatus($("nunifStatus"), e.message, "bad"); }
});

/* Check the path as it is typed, without saving it. The old page only found
   out when you pressed Save, so a stale red message sat there while you fixed
   the path, and a wrong path got written to config.json before being rejected. */
/* The four fields the setup panel owns. They are the only settings a job reads
   from the saved config instead of from the page, so leaving them unsaved is
   not a cosmetic problem: the models folder in particular has no override in
   the request, so an unsaved one sends a three-day run to the wrong place. */
function setupPatch() {
  return {
    nunif_dir: $("nunifPath").value.trim(),
    download_dir: $("downloadDir").value.trim(),
    dataset_dir: $("datasetDir").value.trim(),
    models_dir: $("modelsDir").value.trim(),
  };
}
function setupDirty(resolved) {
  const c = (STATE && STATE.config) || {}, p = (STATE && STATE.paths) || {};
  const o = setupPatch();
  return (resolved || o.nunif_dir) !== p.nunif_dir
      || o.download_dir !== (c.download_dir || "")
      || o.dataset_dir !== (c.dataset_dir || "")
      || o.models_dir !== (c.models_dir || "");
}
async function persistSetup() {
  const st = await api("/api/config", setupPatch());
  fillFromState(st);
  repointDerived(st);
  refreshRuns().catch(() => {});
  checkPrepSrc();
  checkTrainData();
  return st;
}

let nunifCheckTimer = null, nunifCheckSeq = 0;
async function checkNunifPath(immediate, keep) {
  clearTimeout(nunifCheckTimer);
  const run = async () => {
    const value = $("nunifPath").value;
    const seq = ++nunifCheckSeq;
    try {
      const d = await api("/api/check_nunif", { path: value });
      if (seq !== nunifCheckSeq) return;          // a later keystroke won
      setStatus($("nunifStatus"), d.message, d.kind);
      $("saveSetup").disabled = !d.ok && !!value.trim();
      // A path that checks out is worth keeping the moment it is settled on.
      // It used to live only in the text box until Save was pressed, so
      // auto-detect could fill it in, report "looks right", and then clicking
      // a tab threw it away -- leaving every job refusing to start with "set
      // the nunif folder first" while the box on screen showed a good path.
      if (d.ok && keep && setupDirty(d.nunif_dir)) await persistSetup();
    } catch (e) { /* server busy; leave the message alone */ }
  };
  if (immediate) return run();
  nunifCheckTimer = setTimeout(run, 350);
}
$("nunifPath").addEventListener("input", () => checkNunifPath(false));
$("nunifPath").addEventListener("change", () => checkNunifPath(true, true));

$("saveSetup").addEventListener("click", async () => {
  try {
    const st = await persistSetup();          // same four fields, one code path
    if (!st.paths.nunif_dir) {
      setStatus($("nunifStatus"),
        "That folder does not look like a nunif install \u2014 it needs nunif\\ and iw3\\ inside it.", "bad");
      return;
    }
    if (!st.paths.python_exe) {
      setStatus($("nunifStatus"),
        "Found nunif, but not its python folder next to it. Jobs will not be able to run.", "bad");
      return;
    }
    toast("Saved. Using " + st.paths.nunif_dir);
    showSetup(false);
  } catch (e) { setStatus($("nunifStatus"), e.message, "bad"); }
});


/* When the nunif folder changes, any page still pointing at the OLD install's
   default data folders is now pointing somewhere meaningless. Fields the user
   set by hand are left alone; only ones still holding a previous default move. */
function repointDerived(st) {
  const p = st.paths || {}, c = st.config || {};
  const move = (id, next, previous) => {
    const el = $(id);
    if (!el || !next) return;
    const cur = el.value.trim();
    if (!cur || cur === (previous || "")) el.value = next;
  };
  const oldp = (STATE_PATHS || {});
  move("collectDir", (c.collect && c.collect.download_dir) || p.download_dir, oldp.download_dir);
  move("prep_src", $("collectDir").value, oldp.download_dir);
  move("prep_out", p.dataset_dir, oldp.dataset_dir);
  move("train_data", $("prep_out").value, oldp.dataset_dir);
  STATE_PATHS = Object.assign({}, p);
}

/* ----------------------------------------------------------------- boot */

function fillFromState(st) {
  STATE = st;
  const c = st.config || {}, p = st.paths || {};
  $("nunifPath").value = c.nunif_dir || p.nunif_dir || "";
  $("downloadDir").value = c.download_dir || "";
  $("datasetDir").value = c.dataset_dir || "";
  $("modelsDir").value = c.models_dir || "";
  if (c.theme) applyTheme(c.theme, false);
  renderThemes();
  writeCollect(c.collect);
  $("collectDir").value = (c.collect && c.collect.download_dir) || p.download_dir || "";
  writePrep(c.prep);
  // Step 2 inherits step 1's folder unless it was deliberately pointed elsewhere.
  if (!$("prep_src").value) {
    $("prep_src").value = (c.collect && c.collect.download_dir) || p.download_dir || "";
  }
  if (!$("prep_out").value) $("prep_out").value = p.dataset_dir || "";
  applyPrepMode();
  writeTrain(c.train);
  $("live_resume").checked = $("train_resume").checked;
  $("live_overflow").checked = $("train_overflow").checked;
  $("install_lowres").checked = c.install_lowres !== false;
  $("install_border").checked = c.install_border !== false;
  $("resumeNote").textContent = $("train_resume").checked
    ? "will carry on from the last finished epoch"
    : "will start from the beginning";
  if (!$("train_data").value) $("train_data").value = $("prep_out").value || p.dataset_dir || "";
  applyTrainMode();
  if (st.has_pexels_key) $("pexels_key").placeholder = "Pexels key saved";
  if (st.has_pixabay_key) $("pixabay_key").placeholder = "Pixabay key saved";
  applyMediaType();
  const ok = p.ready;
  $("setupBtn").textContent = ok ? "Settings" : "Set up";
  return ok;
}

(async function boot() {
  let st;
  try { st = await api("/api/state"); }
  catch (e) { toast("Cannot reach the GUI server: " + e.message, true); return; }
  const ok = fillFromState(st);
  JOB_SLOTS["deps"] = "deps";
  JOB_SLOTS["compile_support"] = "compile_support";
  JOB_SLOTS["probe"] = "probe";
  JOB_SLOTS["fetch"] = currentFetchJob();
  JOB_SLOTS["scan"] = "scan";
  JOB_SLOTS["prep"] = "prep";
  JOB_SLOTS["train"] = "train";
  JOB_SLOTS["bench"] = "bench";
  JOB_SLOTS["verify"] = "verify";
  applyPickMode();
  liveRun = (st.config && st.config.live_run) || null;
  pollLive(false);
  checkPrepSrc();
  checkTrainData();
  if (!ok) {
    showSetup(true);
    if (!$("nunifPath").value) $("detectNunif").click();
  } else {
    showPage(st.config.page || 1);
  }
  pollJobs();
})();
