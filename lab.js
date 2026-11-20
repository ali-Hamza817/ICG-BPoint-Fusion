/* Lab: plug-and-play exploration and experiments, entirely in the browser (no server).
   Loaded on demand by index.html (window.LabApp). Data = decrypted lab/* assets. Python (Pyodide + scikit-learn) runs in a Web Worker. */
(function () {
"use strict";
var PY_BASE = "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/";
var CM_BASE = "https://cdnjs.cloudflare.com/ajax/libs/codemirror/5.65.16/";
var ST = window.LabState = window.LabState || {
  data: null, loading: null, rt: "idle", rtText: "", worker: null, running: false, runs: [], refs: null, pane: "explore", pat: 0, beat: 0, showAbel: true, showSheikh: true,
  cfg: { feat: "sheikh", model: "ada", val: "cv5", preset: "quick", trees: 20, depth: 6, mf: 0.3, lr: 1, alpha: 100, patient: "VP_003" },
  code: "", edited: false, cm: null, codeTab: "exp", sel: [], detail: null, fltDs: "All", est: "drost2022", log: [], prog: null, pendingRun: null, verify: null, idCounter: 0 };
var $q = function (s, r) { return (r || document).querySelector(s); };
var ABEL_LABEL = { "arbol2017-isoelectric-crossings": "Arbol17 IC", "arbol2017-second-derivative": "Arbol17 SD", "arbol2017-third-derivative": "Arbol17 TD", "debski1993-second-derivative": "Debski93",
  "drost2022": "Drost22", "forouzanfar2018": "Forouzanfar18", "lozano2007-linear-regression": "Lozano07 LR", "lozano2007-quadratic-regression": "Lozano07 QR", "miljkovic2022": "Miljkovic22",
  "pale2021": "Pale21", "sherwood1990": "Sherwood90", "stern1985": "Stern85", "rr_interval_ms": "RR interval" };
var MORPH = [[450, 451, "Last notch amplitude", "ICG value at the last notch in the search window"], [451, 452, "Last inflection amplitude", "ICG value at the last inflection point"],
  [452, 453, "Amplitude at max d³Z/dt³", "ICG value where the 3rd derivative peaks (rapid slope change)"], [453, 454, "C amplitude", "peak ICG value (maximum ejection velocity)"],
  [454, 455, "C − last notch", "amplitude difference"], [455, 456, "C − last inflection", "amplitude difference"], [456, 457, "C − max d³ point", "amplitude difference"],
  [457, 458, "t(notch) − t(inflection)", "time difference, ms"], [458, 459, "RC − t(notch)", "time difference, ms"], [459, 460, "RC − t(inflection)", "time difference, ms"],
  [460, 461, "RC − t(max d³)", "time difference, ms"], [461, 462, "RC interval", "R to C peak, ms"], [463, 463, "t(last notch)", "time index, ms (0 = none found)"],
  [462, 464, "t(last inflection)", "time index, ms (0 = none found)"], [464, 465, "t(max d³)", "time index, ms"]];
var COL = { sheikh: "#0072B2", abel: "#D55E00", ok: "#009E73", warn: "#E69F00", red: "#d1243e" };

/* ---------------- data ---------------- */
function gunzip(buf) { return new Response(new Blob([buf]).stream().pipeThrough(new DecompressionStream("gzip"))).arrayBuffer(); }
async function loadData(setText) {
  if (ST.data) return ST.data;
  if (ST.loading) return ST.loading;
  ST.loading = (async function () {
    setText("Decrypting metadata…"); var meta = await jsonAsset("lab/meta.json");
    setText("Decrypting waveforms (17 MB)…"); var icgB = await gunzip(await asset("lab/icg404.f32.gz"));
    setText("Decrypting features…"); var morphB = await gunzip(await asset("lab/morph15.f32.gz")), abelB = await gunzip(await asset("lab/abel13.f32.gz"));
    var n = meta.n, ICG = new Float32Array(icgB), MORPHA = new Float32Array(morphB), ABEL = new Float32Array(abelB), SK = new Float32Array(n * 465), d1 = new Float64Array(404), d2 = new Float64Array(404);
    setText("Building the 465-feature matrix…");
    for (var i = 0; i < n; i++) {
      var b = i * 404, o = i * 465;
      for (var j = 1; j < 403; j++) d1[j] = (ICG[b + j + 1] - ICG[b + j - 1]) / 2;
      for (j = 2; j < 402; j++) d2[j] = (d1[j + 1] - d1[j - 1]) / 2;
      for (j = 0; j < 151; j++) SK[o + j] = ICG[b + 2 + j];
      for (j = 0; j < 150; j++) SK[o + 151 + j] = d1[2 + j];
      for (j = 0; j < 149; j++) SK[o + 301 + j] = d2[2 + j];
      for (j = 0; j < 15; j++) SK[o + 450 + j] = MORPHA[i * 15 + j];
    }
    ST.data = { meta: meta, ICG: ICG, MORPH: MORPHA, ABEL: ABEL, SK: SK, buffers: { meta: JSON.stringify(meta), icg: icgB, morph: morphB, abel: abelB } };
    return ST.data;
  })();
  return ST.loading;
}

/* ---------------- Python runtime (Web Worker) ---------------- */
var WORKER_SRC = "let py=null;const post=(type,o)=>self.postMessage(Object.assign({type},o||{}));" +
  "self.onmessage=async(e)=>{const m=e.data;try{" +
  "if(m.type==='init'){importScripts(m.base+'pyodide.js');post('status',{text:'Starting Python…'});py=await loadPyodide({indexURL:m.base});" +
  "py.setStdout({batched:s=>post('log',{text:s})});py.setStderr({batched:s=>post('log',{text:s})});" +
  "post('status',{text:'Loading numpy, scipy and scikit-learn (first time only, about 50 MB)…'});await py.loadPackage(['numpy','scipy','scikit-learn']);post('rt_ready');}" +
  "else if(m.type==='data'){post('status',{text:'Sending data to Python…'});py.globals.set('_meta_json',m.meta);py.globals.set('_icg',new Uint8Array(m.icg));py.globals.set('_morph',new Uint8Array(m.morph));py.globals.set('_abel',new Uint8Array(m.abel));" +
  "const sk=await (await fetch(m.sheikh)).text();const pre=await (await fetch(m.prelude)).text();py.runPython(sk);py.runPython(pre);}" +
  "else if(m.type==='run'){await py.runPythonAsync(m.code);post('done');}" +
  "else if(m.type==='verify'){py.globals.set('_vi',m.index);const r=py.runPython(\"import numpy as _np\\nv,a=features_live(ICG[_vi])\\nd=float(_np.max(_np.abs(v.astype(_np.float32)-lab.X465[_vi])))\\njson.dumps(dict(maxdiff=d,n=int(len(v)),infl=bool(a['has_infl']),notch=bool(a['has_notch'])))\");post('verify',{res:r});}" +
  "}catch(err){post('error',{text:String((err&&err.message)||err)});}};";
function startRuntime() {
  if (ST.worker || ST.rt === "loading") return;
  ST.rt = "loading"; ST.rtText = "Starting…"; refreshStatus();
  var url = URL.createObjectURL(new Blob([WORKER_SRC], { type: "text/javascript" })); var w = new Worker(url); ST.worker = w; ST.dataSent = false;
  w.onmessage = onWorker; w.onerror = function (e) { ST.rt = "error"; ST.rtText = "Worker error: " + (e.message || "unknown"); ST.running = false; refreshAll(); };
  w.postMessage({ type: "init", base: PY_BASE });
}
function sendData() {
  if (ST.dataSent || !ST.data) return; ST.dataSent = true; var b = ST.data.buffers;
  var u = function (f) { return new URL(f, location.href).href; };
  ST.worker.postMessage({ type: "data", meta: b.meta, icg: b.icg, morph: b.morph, abel: b.abel, sheikh: u("lab_sheikh.py"), prelude: u("lab_prelude.py") });
}
function onWorker(e) {
  var m = e.data;
  if (m.type === "status") { ST.rtText = m.text; refreshStatus(); }
  else if (m.type === "log") { ST.log.push(m.text); renderLog(); }
  else if (m.type === "rt_ready") { ST.rt = "data"; ST.rtText = "Runtime ready, loading data…"; loadData(function () {}).then(function () { sendData(); }); refreshStatus(); }
  else if (m.type === "data_ready") { ST.rt = "ready"; ST.rtText = "Python ready"; refreshAll(); if (ST.pendingRun) { var c = ST.pendingRun; ST.pendingRun = null; runCode(c); } }
  else if (m.type === "progress") { ST.prog = m; renderProgress(); }
  else if (m.type === "result") { addRun(m.result); }
  else if (m.type === "done") { ST.running = false; ST.prog = ST.prog ? Object.assign({}, ST.prog, { done: ST.prog.total }) : null; refreshAll(); }
  else if (m.type === "verify") { ST.verify = JSON.parse(m.res); ST.verify.index = ST.verifyIdx; renderVerify(); }
  else if (m.type === "error") { ST.running = false; ST.log.push("ERROR: " + m.text); ST.rtErr = m.text; renderLog(); refreshAll(); }
}
function runCode(code) {
  if (ST.rt !== "ready") { ST.pendingRun = code; if (!ST.worker) startRuntime(); refreshAll(); return; }
  ST.running = true; ST.prog = { done: 0, total: 1, text: "Starting…" }; ST.log = ["# run started " + new Date().toLocaleTimeString()]; ST.rtErr = null; ST.t0 = Date.now(); refreshAll();
  ST.worker.postMessage({ type: "run", code: code });
}
function stopRun() {
  if (!ST.worker) return; ST.worker.terminate(); ST.worker = null; ST.running = false; ST.rt = "idle"; ST.rtText = "Stopped. The runtime restarts on the next run (cached, faster)."; ST.log.push("# stopped by user"); ST.dataSent = false; ST.pendingRun = null; refreshAll();
}

/* ---------------- references (verified results) ---------------- */
async function loadRefs() {
  if (ST.refs) return;
  var S0 = await jsonAsset("data/summary.json"), MOD = { ada: "AdaBoost", rf: "Random Forest", lgbm: "LightGBM", svr: "SVR" }, F = { sheikh: "Sheikh 465", fus: "Sheikh 465 + Abel 13" };
  ST.refs = [{ id: "pub", name: "Abel 2026 (published)", feat: "Abel 12 rules + RR", model: "Random Forest", val: "published, pooled", MAE: S0.published.pooled, SD: S0.published.pooled_sd, ref: "pub" }]
    .concat(S0.pooled.map(function (r) { var k = r.model.split("_"); return { id: "ref_" + r.model, name: F[k[0]] + " · " + MOD[k[1]], feat: F[k[0]], model: MOD[k[1]], val: "5-fold, full settings", MAE: r.MAE, SD: r.SD, MedAE: r.MedAE, ref: "ours" }; }));
}

/* ---------------- charts ---------------- */
function cvs(c, h) { return scv(c, h); }
function drawAxes(g, x0, x1, y0, y1, xt, yt, fx, fy, xl, yl, mut, line) {
  g.strokeStyle = line; g.lineWidth = 1; g.fillStyle = mut; g.font = "11px " + css("--font-body");
  xt.forEach(function (t) { var x = fx(t); g.beginPath(); g.moveTo(x, y0); g.lineTo(x, y1); g.stroke(); g.textAlign = "center"; g.fillText(t, x, y0 + 14); });
  (yt || []).forEach(function (t) { var y = fy(t); g.beginPath(); g.moveTo(x0, y); g.lineTo(x1, y); g.stroke(); g.textAlign = "right"; g.fillText(t, x0 - 6, y + 4); });
  g.textAlign = "center"; if (xl) g.fillText(xl, (x0 + x1) / 2, y0 + 30); if (yl) { g.save(); g.translate(12, (y0 + y1) / 2); g.rotate(-Math.PI / 2); g.fillText(yl, 0, 0); g.restore(); }
}
function beatIndex() { var p = ST.data.meta.patients[ST.pat]; return p.start + Math.min(ST.beat, p.end - p.start - 1); }
function icgAt(i, t) { var f = t + 2, a = Math.floor(f), r = f - a, b = i * 404; if (a < 0 || a > 402) return NaN; return ST.data.ICG[b + a] * (1 - r) + ST.data.ICG[b + a + 1] * r; }
function searchWindow(i) {
  var d = ST.data, RC = d.meta.C[i], mag = d.meta.Camp[i], u = RC >= 140 ? 140 : RC, thr = 0.66 * mag, ulm = u;
  for (var m = 1; m <= u; m++) { if (ICG_ms(i, m - 1) > thr) { ulm = m - 1; break; } }
  ulm = Math.max(ulm, 37); return [34, ulm - 1];
}
function ICG_ms(i, ms) { return ST.data.ICG[i * 404 + 2 + ms]; }
function landmarks(i) {
  var sk = ST.data.SK, o = i * 465, L = [];
  [[463, "last notch", COL.sheikh], [462, "last inflection", "#7b61c9"], [464, "max d³", "#0aa5b8"]].forEach(function (a) { var v = sk[o + a[0]]; L.push({ name: a[1], t: v > 0 ? v - 1 : NaN, col: a[2] }); });
  return L;
}
function drawBeat() {
  var c = $q("#lb_ex"); if (!c || !ST.data) return; var d = ST.data, i = beatIndex(), o = cvs(c, 300), g = o.g, w = o.w, h = o.h, mut = css("--muted"), line = css("--line"), fg = css("--fg");
  var x0 = 44, x1 = w - 12, y0 = h - 34, y1 = 10, T0 = -2, T1 = 401, lo = 1e9, hi = -1e9;
  for (var t = -2; t <= 401; t++) { var v = icgAt(i, t); if (v < lo) lo = v; if (v > hi) hi = v; }
  var pd = (hi - lo) * 0.08; lo -= pd; hi += pd; var fx = function (t) { return x0 + (t - T0) / (T1 - T0) * (x1 - x0); }, fy = function (v) { return y0 - (v - lo) / (hi - lo) * (y0 - y1); };
  var yt = [], st = Math.max(0.5, Math.round((hi - lo) / 5 * 2) / 2); for (var y = Math.ceil(lo / st) * st; y <= hi; y += st) yt.push(Math.round(y * 10) / 10);
  drawAxes(g, x0, x1, y0, y1, [0, 50, 100, 150, 200, 250, 300, 350, 400], yt, fx, fy, "time after R-peak (ms)", "ICG dZ/dt", mut, line);
  if (ST.showSheikh) { var sw = searchWindow(i); g.fillStyle = "rgba(0,114,178,.09)"; g.fillRect(fx(sw[0]), y1, fx(sw[1]) - fx(sw[0]), y0 - y1); g.fillStyle = COL.sheikh; g.font = "11px " + css("--font-body"); g.textAlign = "left"; g.fillText("Sheikh search window", fx(sw[0]) + 4, y1 + 12); }
  g.strokeStyle = fg; g.lineWidth = 2; g.beginPath(); for (t = -2; t <= 401; t++) { var xx = fx(t), yy = fy(icgAt(i, t)); t === -2 ? g.moveTo(xx, yy) : g.lineTo(xx, yy); } g.stroke();
  var B = d.meta.B[i], C = d.meta.C[i];
  g.setLineDash([5, 4]); g.lineWidth = 1.6; g.strokeStyle = COL.abel; g.beginPath(); g.moveTo(fx(B), y1); g.lineTo(fx(B), y0); g.stroke(); g.setLineDash([]);
  g.fillStyle = COL.abel; g.beginPath(); g.arc(fx(B), fy(icgAt(i, B)), 5, 0, 7); g.fill(); g.font = "600 12px " + css("--font-body"); g.textAlign = "left"; g.fillText("B (rater 1) " + B.toFixed(0) + " ms", fx(B) + 8, fy(icgAt(i, B)) + 16);
  g.fillStyle = COL.ok; g.beginPath(); g.moveTo(fx(C), fy(icgAt(i, C)) - 8); g.lineTo(fx(C) + 7, fy(icgAt(i, C))); g.lineTo(fx(C), fy(icgAt(i, C)) + 8); g.lineTo(fx(C) - 7, fy(icgAt(i, C))); g.fill(); g.fillText("C " + C + " ms", fx(C) + 10, fy(icgAt(i, C)) - 4);
  if (ST.showSheikh) landmarks(i).forEach(function (m) { if (isNaN(m.t)) return; g.fillStyle = m.col; g.beginPath(); g.arc(fx(m.t), fy(icgAt(i, m.t)), 4.5, 0, 7); g.fill(); });
  if (ST.showAbel) { var A = d.ABEL, k; for (k = 0; k < 12; k++) { var e = A[i * 13 + k]; if (isNaN(e) || e < -2 || e > 400) continue; g.strokeStyle = COL.abel; g.lineWidth = 1.4; g.beginPath(); g.arc(fx(e), fy(icgAt(i, e)), 4, 0, 7); g.stroke(); } }
}
function drawEstimators() {
  var c = $q("#lb_est"); if (!c || !ST.data) return; var d = ST.data, i = beatIndex(), rows = [], B = d.meta.B[i];
  for (var k = 0; k < 12; k++) rows.push({ name: ABEL_LABEL[d.meta.cols.abel[k]], t: d.ABEL[i * 13 + k], grp: "abel" });
  landmarks(i).forEach(function (m) { rows.push({ name: "Sheikh · " + m.name, t: m.t, grp: "sheikh" }); });
  var rh = 20, o = cvs(c, rows.length * rh + 40), g = o.g, w = o.w, mut = css("--muted"), line = css("--line"), fg = css("--fg"), lx = 130, x0 = lx, x1 = w - 90, T0 = -50, T1 = 300, fx = function (t) { return x0 + (Math.max(T0, Math.min(T1, t)) - T0) / (T1 - T0) * (x1 - x0); };
  g.font = "12px " + css("--font-body"); g.strokeStyle = line; g.fillStyle = mut; g.textAlign = "center";
  for (var t = T0; t <= T1; t += 50) { g.beginPath(); g.moveTo(fx(t), 6); g.lineTo(fx(t), rows.length * rh + 8); g.stroke(); g.fillText(t, fx(t), rows.length * rh + 24); }
  g.fillText("estimated B time after R (ms)", (x0 + x1) / 2, rows.length * rh + 38);
  g.strokeStyle = COL.abel; g.setLineDash([5, 4]); g.lineWidth = 1.6; g.beginPath(); g.moveTo(fx(B), 4); g.lineTo(fx(B), rows.length * rh + 8); g.stroke(); g.setLineDash([]);
  rows.forEach(function (r, k) { var y = 8 + k * rh + rh / 2, col = r.grp === "abel" ? COL.abel : COL.sheikh; g.fillStyle = fg; g.textAlign = "right"; g.fillText(r.name, lx - 8, y + 4);
    if (isNaN(r.t) || r.t <= 0 && r.grp === "sheikh") { g.fillStyle = mut; g.textAlign = "left"; g.fillText("none", x0 + 6, y + 4); return; }
    g.fillStyle = col; g.beginPath(); g.arc(fx(r.t), y, 5, 0, 7); g.fill(); g.fillStyle = mut; g.textAlign = "left"; g.fillText((r.t - B >= 0 ? "+" : "−") + Math.abs(r.t - B).toFixed(0) + " ms", x1 + 14, y + 4); });
}
function miniLine(c, off, len, title) {
  var i = beatIndex(), o = cvs(c, 110), g = o.g, w = o.w, h = o.h, mut = css("--muted"), v = [], lo = 1e9, hi = -1e9, k;
  for (k = 0; k < len; k++) { var x = ST.data.SK[i * 465 + off + k]; v.push(x); if (x < lo) lo = x; if (x > hi) hi = x; }
  g.fillStyle = mut; g.font = "11px " + css("--font-body"); g.textAlign = "left"; g.fillText(title, 4, 12);
  g.strokeStyle = COL.sheikh; g.lineWidth = 1.6; g.beginPath(); v.forEach(function (y, j) { var px = 4 + j / (len - 1) * (w - 8), py = h - 6 - (y - lo) / ((hi - lo) || 1) * (h - 30); j ? g.lineTo(px, py) : g.moveTo(px, py); }); g.stroke();
}
function drawMini() { var a = $q("#lb_m1"); if (!a || !ST.data) return; miniLine(a, 0, 151, "features 1–151 · ICG, R to R+150 ms"); miniLine($q("#lb_m2"), 151, 150, "152–301 · 1st derivative"); miniLine($q("#lb_m3"), 301, 149, "302–450 · 2nd derivative"); }

/* ---------------- explore pane ---------------- */
function fx2(v, d) { return v == null || isNaN(v) ? "–" : Number(v).toFixed(d == null ? 2 : d); }
function explorePane() {
  var d = ST.data, M = d.meta, opts = ["EmpkinS", "Guardian"].map(function (ds) { return '<optgroup label="' + ds + '">' + M.patients.map(function (p, k) { return p.dataset === ds ? '<option value="' + k + '"' + (k === ST.pat ? " selected" : "") + ">" + esc(p.id) + "</option>" : ""; }).join("") + "</optgroup>"; }).join("");
  var p = M.patients[ST.pat], nb = p.end - p.start;
  return '<div class="lb-row"><label>Patient <select id="ex_p">' + opts + '</select></label><label class="grow">Beat <input type="range" id="ex_b" min="0" max="' + (nb - 1) + '" value="' + Math.min(ST.beat, nb - 1) + '"></label><span id="ex_bn" class="mono"></span>' +
    '<button class="chip" id="ex_prev" aria-label="Previous beat">◀</button><button class="chip" id="ex_next" aria-label="Next beat">▶</button></div>' +
    '<div class="lb-row"><label class="chk"><input type="checkbox" id="ex_sa"' + (ST.showAbel ? " checked" : "") + '> Abel\'s 12 estimates</label><label class="chk"><input type="checkbox" id="ex_ss"' + (ST.showSheikh ? " checked" : "") + '> Sheikh\'s landmarks and search window</label><span id="ex_info" class="muted"></span></div>' +
    '<div class="panel"><h3>ICG of this beat</h3><canvas id="lb_ex"></canvas></div>' +
    '<div class="lb-grid"><div class="panel"><h3>Sheikh · the 465 features of this beat</h3><canvas id="lb_m1"></canvas><canvas id="lb_m2"></canvas><canvas id="lb_m3"></canvas><div id="ex_morph"></div></div>' +
    '<div class="panel"><h3>Abel · 12 rules + RR vs the true B point</h3><div id="ex_abel"></div><canvas id="lb_est"></canvas></div></div>' +
    '<details class="panel" id="ex_all"><summary style="cursor:pointer;font-weight:600">All 465 values of this beat</summary><div id="ex_allt" style="margin-top:8px"></div></details>' +
    '<div class="panel"><h3>Verify in front of you</h3><p class="muted" style="margin:2px 0 8px">Recompute this beat\'s 465 features with Python running in your browser and compare with the stored vector.</p><button class="chip f" id="ex_ver">Recompute with Python</button> <span id="ex_vres"></span></div>';
}
function updateExplore() {
  if (!$q("#ex_p")) return; var d = ST.data, M = d.meta, i = beatIndex(), p = M.patients[ST.pat], nb = p.end - p.start;
  $q("#ex_b").max = nb - 1; $q("#ex_b").value = Math.min(ST.beat, nb - 1); $q("#ex_bn").textContent = (Math.min(ST.beat, nb - 1) + 1) + " / " + nb;
  $q("#ex_info").innerHTML = esc(p.dataset + " · " + p.id + " · " + M.phases[M.phase[i]] + " · beat " + M.beat_id[i] + " · PEP " + fx2(M.PEP[i], 0) + " ms") + ' · <a href="#' + hashOf(p.dataset + "_" + p.id) + '">ECG, PEP and every beat in Task 1 →</a>';
  fillAll();
  var sk = d.SK, o = i * 465, rows = MORPH.map(function (m) { var v = sk[o + m[0]]; return "<tr><td class='n'>" + m[1] + "</td><td>" + m[2] + "</td><td class='n'>" + fx2(v, 3) + "</td><td class='muted'>" + m[3] + "</td></tr>"; }).join("");
  $q("#ex_morph").innerHTML = '<div class="tablewrap" style="max-height:none;margin-top:8px"><table class="lbt"><thead><tr><th class="n">#</th><th>morphology feature</th><th class="n">value</th><th>meaning</th></tr></thead><tbody>' + rows + "</tbody></table></div>";
  var B = M.B[i], ar = ""; for (var k = 0; k < 13; k++) { var e = d.ABEL[i * 13 + k]; ar += "<tr><td>" + ABEL_LABEL[M.cols.abel[k]] + "</td><td class='n'>" + fx2(e, 0) + "</td><td class='n'>" + (k < 12 ? (isNaN(e) ? "–" : (e - B >= 0 ? "+" : "−") + Math.abs(e - B).toFixed(0)) : "") + "</td></tr>"; }
  $q("#ex_abel").innerHTML = '<div class="tablewrap" style="max-height:none"><table class="lbt"><thead><tr><th>rule</th><th class="n">estimate (ms)</th><th class="n">error vs B</th></tr></thead><tbody>' + ar + "</tbody></table></div>";
  drawBeat(); drawMini(); drawEstimators(); renderVerify();
}
function fillAll() {
  var det = $q("#ex_all"); if (!det) return; if (!det.open) return; var d = ST.data, o = beatIndex() * 465, nm = d.meta.cols.sheikh, rows = [];
  for (var j = 0; j < 465; j++) rows.push("<tr><td class='n'>" + (j + 1) + "</td><td>" + (j >= 450 ? "morphology: " + nm[j] : nm[j]) + "</td><td class='n'>" + fx2(d.SK[o + j], 4) + "</td></tr>");
  $q("#ex_allt").innerHTML = '<div class="tablewrap" style="max-height:320px"><table class="lbt"><thead><tr><th class="n">#</th><th>feature</th><th class="n">value</th></tr></thead><tbody>' + rows.join("") + "</tbody></table></div><p class=\"muted\" style=\"margin:6px 0 0\">1–151 ICG, 152–301 first derivative, 302–450 second derivative (R to R+150 ms); 451–465 morphology. Order follows Sheikh's MATLAB file.</p>";
}
function renderVerify() {
  var el = $q("#ex_vres"); if (!el) return; var v = ST.verify; if (!v || v.index !== beatIndex()) { el.textContent = ST.rt === "ready" ? "" : (ST.rt === "idle" ? "The Python runtime starts on first use (about 50 MB, cached afterwards)." : ST.rtText); return; }
  el.innerHTML = v.maxdiff === 0 ? '<b style="color:var(--ok)">Identical</b> · ' + v.n + " features recomputed live, largest difference 0.0" : "Largest difference <b>" + v.maxdiff + "</b> over " + v.n + " features";
}
function verifyBeat() {
  var i = beatIndex(); ST.verify = null; renderVerify();
  var go = function () { ST.verifyIdx = i; ST.worker.postMessage({ type: "verify", index: i }); };
  if (ST.rt === "ready") go(); else { ST.afterReady = go; if (!ST.worker) startRuntime(); }
}

/* ---------------- feature comparison pane ---------------- */
function subsetIdx() { var m = ST.data.meta.ds, f = ST.fltDs, out = []; for (var i = 0; i < m.length; i++) if (f === "All" || (f === "EmpkinS" ? m[i] === 0 : m[i] === 1)) out.push(i); return out; }
function estimators() {
  var d = ST.data, out = []; d.meta.cols.abel.slice(0, 12).forEach(function (c, k) { out.push({ id: c, name: "Abel · " + ABEL_LABEL[c], get: function (i) { return d.ABEL[i * 13 + k]; } }); });
  [[463, "t(last notch)"], [462, "t(last inflection)"], [464, "t(max d³)"]].forEach(function (a) { out.push({ id: "sk" + a[0], name: "Sheikh · " + a[1], get: function (i) { var v = d.SK[i * 465 + a[0]]; return v > 0 ? v - 1 : NaN; } }); });
  return out;
}
function comparePane() {
  var chips = ["All", "EmpkinS", "Guardian"].map(function (x) { return '<button class="chip f" aria-pressed="' + (ST.fltDs === x) + '" data-ds="' + x + '">' + x + "</button>"; }).join("");
  var est = estimators().map(function (e) { return '<option value="' + e.id + '"' + (e.id === ST.est ? " selected" : "") + ">" + e.name + "</option>"; }).join("");
  return '<div class="lb-row"><span class="muted">Beats</span><span class="chips" id="cp_f" style="margin:0">' + chips + '</span><span id="cp_n" class="muted"></span></div>' +
    '<div class="lb-grid"><div class="panel"><h3>Each estimator vs the true B point · mean absolute error</h3><canvas id="cp_c1"></canvas></div><div class="panel"><h3>Sheikh features most related to B · top of each block (|correlation|)</h3><canvas id="cp_c2"></canvas></div></div>' +
    '<div class="panel"><div class="lb-row" style="margin:0 0 6px"><h3 style="margin:0">Estimate vs true B</h3><select id="cp_sel">' + est + '</select><span id="cp_st" class="muted"></span></div><canvas id="cp_c3"></canvas></div>';
}
function updateCompare() {
  if (!$q("#cp_c1")) return; var d = ST.data, idx = subsetIdx(), B = d.meta.B, mut = css("--muted"), line = css("--line"), fg = css("--fg"); $q("#cp_n").textContent = idx.length.toLocaleString() + " beats";
  var E = estimators(), stat = E.map(function (e) { var s = 0, n = 0; idx.forEach(function (i) { var v = e.get(i); if (!isNaN(v)) { s += Math.abs(v - B[i]); n++; } }); return { name: e.name, mae: n ? s / n : NaN, cov: n / idx.length, abel: e.name.indexOf("Abel") === 0 }; }).sort(function (a, b) { return a.mae - b.mae; });
  var o = cvs($q("#cp_c1"), stat.length * 24 + 34), g = o.g, w = o.w, lx = 150, x0 = lx, x1 = w - 170, mx = Math.max(60, Math.ceil(Math.max.apply(null, stat.map(function (s) { return isNaN(s.mae) ? 0 : s.mae; })) / 10) * 10), fx = function (v) { return x0 + v / mx * (x1 - x0); };
  g.font = "12px " + css("--font-body"); for (var t = 0; t <= mx; t += 10) { g.strokeStyle = line; g.beginPath(); g.moveTo(fx(t), 4); g.lineTo(fx(t), stat.length * 24 + 6); g.stroke(); g.fillStyle = mut; g.textAlign = "center"; g.fillText(t, fx(t), stat.length * 24 + 20); }
  stat.forEach(function (s, k) { var y = 6 + k * 24; g.fillStyle = fg; g.textAlign = "right"; g.fillText(s.name, lx - 8, y + 14); if (isNaN(s.mae)) return; g.fillStyle = s.abel ? COL.abel : COL.sheikh; g.fillRect(x0, y + 3, fx(s.mae) - x0, 15); g.fillStyle = mut; g.textAlign = "left"; g.fillText(s.mae.toFixed(1) + " ms · " + Math.round(100 * s.cov) + "% beats", fx(s.mae) + 6, y + 15); });
  /* correlation of every Sheikh feature with B */
  var SK = d.SK, nI = idx.length, mB = 0, k2, j, i; idx.forEach(function (i) { mB += B[i]; }); mB /= nI; var vB = 0; idx.forEach(function (i) { vB += (B[i] - mB) * (B[i] - mB); });
  var cor = []; for (j = 0; j < 465; j++) { var m = 0; for (k2 = 0; k2 < nI; k2++) m += SK[idx[k2] * 465 + j]; m /= nI; var sxy = 0, sxx = 0; for (k2 = 0; k2 < nI; k2++) { i = idx[k2]; var dx = SK[i * 465 + j] - m; sxy += dx * (B[i] - mB); sxx += dx * dx; } cor.push({ j: j, r: sxx > 0 ? Math.abs(sxy / Math.sqrt(sxx * vB)) : 0 }); }
  var blocks = [[0, 151, "ICG", 4, COL.sheikh], [151, 301, "1st derivative", 4, "#56B4E9"], [301, 450, "2nd derivative", 4, "#7b61c9"], [450, 465, "", 7, COL.ok]], top = [];
  blocks.forEach(function (b) { cor.filter(function (c) { return c.j >= b[0] && c.j < b[1]; }).sort(function (x, y) { return y.r - x.r; }).slice(0, b[3]).forEach(function (c) { top.push({ j: c.j, r: c.r, b: b }); }); });
  var o2 = cvs($q("#cp_c2"), top.length * 24 + 34), g2 = o2.g, w2 = o2.w, x02 = 170, x12 = w2 - 50, fx2_ = function (v) { return x02 + v * (x12 - x02); };
  g2.font = "12px " + css("--font-body"); for (t = 0; t <= 1; t += 0.25) { g2.strokeStyle = line; g2.beginPath(); g2.moveTo(fx2_(t), 4); g2.lineTo(fx2_(t), top.length * 24 + 6); g2.stroke(); g2.fillStyle = mut; g2.textAlign = "center"; g2.fillText(t.toFixed(2), fx2_(t), top.length * 24 + 20); }
  top.forEach(function (c, k) { var y = 6 + k * 24, jj = c.j, lab;
    if (jj < 151) lab = "ICG +" + jj + " ms"; else if (jj < 301) lab = "1st deriv +" + (jj - 151) + " ms"; else if (jj < 450) lab = "2nd deriv +" + (jj - 301) + " ms"; else { var mm = MORPH.filter(function (m) { return m[0] === jj; })[0]; lab = "#" + mm[1] + " " + mm[2]; }
    g2.fillStyle = fg; g2.textAlign = "right"; g2.fillText(lab, x02 - 8, y + 14); g2.fillStyle = c.b[4]; g2.fillRect(x02, y + 3, fx2_(c.r) - x02, 15); g2.fillStyle = mut; g2.textAlign = "left"; g2.fillText(c.r.toFixed(2), fx2_(c.r) + 6, y + 15); });
  drawScatter(idx);
}
function drawScatter(idx) {
  var c = $q("#cp_c3"); if (!c) return; var d = ST.data, e = estimators().filter(function (x) { return x.id === ST.est; })[0], B = d.meta.B, o = cvs(c, 340), g = o.g, w = o.w, h = o.h, mut = css("--muted"), line = css("--line");
  var x0 = 44, x1 = w - 12, y0 = h - 34, y1 = 10, mn = -50, mx = 300, fx = function (v) { return x0 + (v - mn) / (mx - mn) * (x1 - x0); }, fy = function (v) { return y0 - (v - mn) / (mx - mn) * (y0 - y1); }, tk = [0, 50, 100, 150, 200, 250, 300];
  drawAxes(g, x0, x1, y0, y1, tk, tk, fx, fy, "estimate (ms after R)", "true B (ms)", mut, line);
  g.strokeStyle = mut; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(fx(mn), fy(mn)); g.lineTo(fx(mx), fy(mx)); g.stroke(); g.setLineDash([]);
  var step = Math.max(1, Math.floor(idx.length / 2500)), s = 0, n = 0, col = e.id.indexOf("sk") === 0 ? COL.sheikh : COL.abel; g.fillStyle = col; g.globalAlpha = 0.35;
  idx.forEach(function (i, k) { var v = e.get(i); if (isNaN(v)) return; s += Math.abs(v - B[i]); n++; if (k % step) return; g.beginPath(); g.arc(fx(Math.max(mn, Math.min(mx, v))), fy(Math.max(mn, Math.min(mx, B[i]))), 2, 0, 7); g.fill(); }); g.globalAlpha = 1;
  $q("#cp_st").textContent = "MAE " + (n ? (s / n).toFixed(1) : "–") + " ms · " + Math.round(100 * n / idx.length) + "% of beats have an estimate · dashed = perfect";
}

/* ---------------- experiments pane ---------------- */
var FEATS = [["sheikh", "Sheikh 465"], ["waveform", "Sheikh waveform only (1–450)"], ["morphology", "Sheikh morphology only (451–465)"], ["abel", "Abel 13"], ["fusion", "Sheikh 465 + Abel 13"]];
var MODELS = [["ada", "AdaBoost (Sheikh's model)"], ["rf", "Random Forest"], ["hgb", "Gradient boosting (histogram)"], ["ridge", "Ridge regression"]];
var VALS = [["cv5", "5-fold, patients held out"], ["e2g", "Train EmpkinS → test Guardian"], ["g2e", "Train Guardian → test EmpkinS"], ["one", "Hold out one patient"]];
var PRESETS = { quick: { ada: [20, 6, 0.3, 1], rf: [30, 12, 0.3, 1], hgb: [100, 6, 0.3, 0.1], ridge: [0, 0, 0, 100] }, standard: { ada: [40, 8, 0.4, 1], rf: [100, 16, 0.3, 1], hgb: [300, 8, 0.3, 0.1], ridge: [0, 0, 0, 100] } };
function applyPreset() { var p = PRESETS[ST.cfg.preset]; if (!p) return; var v = p[ST.cfg.model]; ST.cfg.trees = v[0]; ST.cfg.depth = v[1]; ST.cfg.mf = v[2]; ST.cfg.lr = v[3]; ST.cfg.alpha = v[3]; }
function genCode(batch) {
  var c = ST.cfg, fname = FEATS.filter(function (f) { return f[0] === c.feat; })[0][1], mname = MODELS.filter(function (m) { return m[0] === c.model; })[0][1], vname = VALS.filter(function (v) { return v[0] === c.val; })[0][1];
  var model = { ada: "make_pipeline(StandardScaler(), AdaBoostRegressor(\n        estimator=DecisionTreeRegressor(max_depth=" + c.depth + ", max_features=" + c.mf + ", random_state=10),\n        n_estimators=" + c.trees + ", learning_rate=" + c.lr + ", loss=\"exponential\", random_state=1))",
    rf: "RandomForestRegressor(n_estimators=" + c.trees + ", max_depth=" + c.depth + ", max_features=" + c.mf + ", n_jobs=1, random_state=0)",
    hgb: "HistGradientBoostingRegressor(max_iter=" + c.trees + ", max_depth=" + c.depth + ", learning_rate=" + c.lr + ", random_state=0)",
    ridge: "make_pipeline(StandardScaler(), Ridge(alpha=" + c.alpha + "))" }[c.model];
  var splits = { cv5: "lab.subject_folds(5, seed=0)", e2g: "[lab.train_test(\"EmpkinS\", \"Guardian\")]", g2e: "[lab.train_test(\"Guardian\", \"EmpkinS\")]", one: "[lab.holdout(\"" + c.patient + "\")]" }[c.val];
  var imports = "import time\nimport numpy as np\nfrom sklearn.pipeline import make_pipeline\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor, HistGradientBoostingRegressor\nfrom sklearn.linear_model import Ridge\nfrom sklearn.tree import DecisionTreeRegressor\n\n";
  var mn = mname.split(" (")[0];
  if (batch) return "# Sheikh vs Abel: the same model (" + mn + ") and validation (" + vname + ") on three feature sets\n# Everything below is plain Python; edit it and press Run.\n" + imports +
    "y = lab.y                          # target: B-point time after R (ms), rater 1\n\ndef make_model():\n    return " + model + "\n\nsplits = " + splits + "      # " + vname + "\n" +
    "for fkey, fname in [(\"sheikh\", \"Sheikh 465\"), (\"abel\", \"Abel 13\"), (\"fusion\", \"Sheikh 465 + Abel 13\")]:\n    X = lab.features(fkey)\n    pred = np.full(len(y), np.nan)\n    t0 = time.time()\n    for k, (train, test) in enumerate(splits):\n        model = make_model()\n        model.fit(X[train], y[train])\n        pred[test] = model.predict(X[test])\n        ok = ~np.isnan(pred)\n        lab.progress(k + 1, len(splits), f\"{fname}: split {k + 1}/{len(splits)} · running MAE {np.mean(np.abs(pred[ok] - y[ok])):.2f} ms\")\n" +
    "    lab.report(f\"{fname} · " + mn + " · " + vname + "\", y, pred, features=fname, model=\"" + mn + "\", validation=\"" + vname + "\", seconds=round(time.time() - t0, 1))\n";
  var name = fname + " · " + mn + " · " + vname;
  return "# " + name + "\n# Everything below is plain Python; edit it and press Run.\n" + imports +
    "X = lab.features(\"" + c.feat + "\")      # one row per beat (" + fname + ")\ny = lab.y                          # target: B-point time after R (ms), rater 1\npred = np.full(len(y), np.nan)\n\ndef make_model():\n    return " + model + "\n\nsplits = " + splits + "      # " + vname + "\nt0 = time.time()\nfor k, (train, test) in enumerate(splits):\n    model = make_model()\n    model.fit(X[train], y[train])\n    pred[test] = model.predict(X[test])\n    ok = ~np.isnan(pred)\n    lab.progress(k + 1, len(splits), f\"split {k + 1}/{len(splits)} done · running MAE {np.mean(np.abs(pred[ok] - y[ok])):.2f} ms\")\n\n" +
    "lab.report(\"" + name.replace(/"/g, "'") + "\", y, pred, features=\"" + fname + "\", model=\"" + mn + "\", validation=\"" + vname + "\", seconds=round(time.time() - t0, 1))\n";
}
function setCode(v) { ST.code = v; if (ST.cm && ST.codeTab === "exp") { var cur = ST.cm.getValue(); if (cur !== v) ST.cm.setValue(v); } }
function regen() { if (ST.edited) return; setCode(genCode()); }
function expPane() {
  var c = ST.cfg, chip = function (grp, arr, cur) { return '<div class="chips" data-g="' + grp + '">' + arr.map(function (a) { return '<button class="chip f" data-v="' + a[0] + '" aria-pressed="' + (a[0] === cur) + '">' + a[1] + "</button>"; }).join("") + "</div>"; };
  var pat = ST.data.meta.patients.map(function (p) { return '<option value="' + p.id + '"' + (p.id === c.patient ? " selected" : "") + ">" + esc(p.dataset + " · " + p.id) + "</option>"; }).join("");
  return '<div class="lb-grid"><div class="panel"><h3>1 · Features</h3>' + chip("feat", FEATS, c.feat) + '<h3 style="margin-top:12px">2 · Model</h3>' + chip("model", MODELS, c.model) + '<h3 style="margin-top:12px">3 · Validation</h3>' + chip("val", VALS, c.val) +
    '<div id="ex_pat" style="margin-top:6px"' + (c.val === "one" ? "" : " hidden") + '><select id="ex_pid">' + pat + "</select></div>" +
    '<h3 style="margin-top:12px">4 · Settings</h3><div class="chips" data-g="preset"><button class="chip f" data-v="quick" aria-pressed="' + (c.preset === "quick") + '">Quick (about 1–2 min)</button><button class="chip f" data-v="standard" aria-pressed="' + (c.preset === "standard") + '">Standard (about 5–10 min)</button><button class="chip f" data-v="custom" aria-pressed="' + (c.preset === "custom") + '">Custom</button></div>' +
    '<div class="lb-row" id="ex_params" style="margin-top:6px"></div><p class="muted" style="margin:4px 0 0">The study used 500 trees with depth 12–20 and max features 0.7; that takes hours in a browser, so its verified results are the “full settings” rows. Expect larger errors from quick runs.</p>' +
    '<div class="lb-row" style="margin-top:12px"><button class="chip f" id="ex_run" style="padding:7px 18px;font-weight:600">▶ Run experiment</button><button class="chip" id="ex_stop" hidden>■ Stop</button><button class="chip" id="ex_cmp3" title="Runs Sheikh 465, Abel 13 and the fusion with the model and validation chosen above">Compare Sheikh vs Abel (3 runs)</button><span id="ex_state" class="muted"></span></div>' +
    '<div id="ex_prog" class="prog" hidden><div></div></div><div id="ex_ptxt" class="muted" style="margin-top:4px"></div></div>' +
    '<div class="panel"><div class="lb-row" style="margin:0 0 6px"><h3 style="margin:0">Code</h3><span class="chips" id="ex_ctabs" style="margin:0"></span><button class="chip" id="ex_reset" hidden>Reset from options</button></div><textarea id="ex_code"></textarea><pre id="ex_log" class="log"></pre></div></div>' +
    '<div class="panel" id="ex_res"></div>';
}
function paramsHtml() {
  var c = ST.cfg, f = function (id, lab, val, step) { return '<label class="sm">' + lab + ' <input type="number" id="' + id + '" value="' + val + '" step="' + step + '" style="width:76px"></label>'; };
  if (c.model === "ridge") return f("pm_alpha", "alpha", c.alpha, 1);
  return f("pm_trees", c.model === "hgb" ? "iterations" : "trees", c.trees, 10) + f("pm_depth", "max depth", c.depth, 1) + (c.model === "hgb" ? "" : f("pm_mf", "max features", c.mf, 0.1)) + (c.model === "rf" ? "" : f("pm_lr", "learning rate", c.lr, 0.1));
}
function bindExp() {
  var root = $q("#lb_pane"); if (!root) return;
  root.querySelectorAll(".chips[data-g]").forEach(function (g) { g.onclick = function (e) { var b = e.target.closest(".chip"); if (!b) return; var grp = g.dataset.g, v = b.dataset.v; ST.cfg[grp] = v; if (grp === "model" || grp === "preset") { if (ST.cfg.preset !== "custom") applyPreset(); } if (grp === "model" && ST.cfg.preset === "custom") { /* keep numbers */ } ST.edited = false; mountPane(); }; });
  var pid = $q("#ex_pid"); if (pid) pid.onchange = function () { ST.cfg.patient = this.value; ST.edited = false; regen(); };
  ["trees", "depth", "mf", "lr", "alpha"].forEach(function (k) { var el = $q("#pm_" + k); if (el) el.oninput = function () { ST.cfg[k] = parseFloat(this.value); ST.cfg.preset = "custom"; root.querySelectorAll('.chips[data-g="preset"] .chip').forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.v === "custom")); }); ST.edited = false; regen(); }; });
  $q("#ex_run").onclick = function () { runCode(ST.cm ? ST.cm.getValue() : ST.code); }; $q("#ex_stop").onclick = stopRun; $q("#ex_cmp3").onclick = function () { ST.edited = true; ST.code = genCode(true); ST.codeTab = "exp"; if (ST.cm) { ST.cm.setOption("readOnly", false); ST.cm.setValue(ST.code); } var rb = $q("#ex_reset"); if (rb) rb.hidden = false; renderCodeTabs(); runCode(ST.code); };
  $q("#ex_reset").onclick = function () { ST.edited = false; regen(); refreshAll(); };
  renderCodeTabs();
}
var LIBS = {};
function renderCodeTabs() {
  var t = $q("#ex_ctabs"); if (!t) return; var tabs = [["exp", "Experiment"], ["api", "Lab API"], ["feat", "Feature extractor"]];
  t.innerHTML = tabs.map(function (a) { return '<button class="chip f" data-t="' + a[0] + '" aria-pressed="' + (ST.codeTab === a[0]) + '">' + a[1] + "</button>"; }).join("");
  t.onclick = function (e) { var b = e.target.closest(".chip"); if (!b) return; ST.codeTab = b.dataset.t; showCodeTab(); renderCodeTabs(); };
}
function showCodeTab() {
  if (!ST.cm) return; var t = ST.codeTab; $q("#ex_run").disabled = false;
  if (t === "exp") { ST.cm.setOption("readOnly", false); ST.cm.setValue(ST.code); }
  else { ST.cm.setOption("readOnly", true); var f = t === "api" ? "lab_prelude.py" : "lab_sheikh.py"; if (LIBS[f]) ST.cm.setValue(LIBS[f]); else { ST.cm.setValue("loading…"); fetch(f).then(function (r) { return r.text(); }).then(function (s) { LIBS[f] = s; if (ST.codeTab !== "exp") ST.cm.setValue(s); }); } }
}
function loadScript(src) { return new Promise(function (res, rej) { var s = document.createElement("script"); s.src = src; s.onload = res; s.onerror = rej; document.head.appendChild(s); }); }
function loadCss(href) { var l = document.createElement("link"); l.rel = "stylesheet"; l.href = href; document.head.appendChild(l); }
async function initEditor() {
  if (!window.CodeMirror) { loadCss(CM_BASE + "codemirror.min.css"); await loadScript(CM_BASE + "codemirror.min.js"); await loadScript(CM_BASE + "mode/python/python.min.js"); }
  var ta = $q("#ex_code"); if (!ta) return; if (ST.cm) { try { ST.cm.toTextArea(); } catch (e) { } }
  ST.cm = window.CodeMirror.fromTextArea(ta, { mode: "python", lineNumbers: true, indentUnit: 4, viewportMargin: Infinity, lineWrapping: false }); ST.cm.setSize("100%", 360);
  ST.cm.on("change", function (cm, ch) { if (ch.origin === "setValue" || ST.codeTab !== "exp") return; ST.code = cm.getValue(); if (!ST.edited) { ST.edited = true; var r = $q("#ex_reset"); if (r) r.hidden = false; } });
  regen(); showCodeTab();
}
function renderLog() { var el = $q("#ex_log"); if (el) { el.textContent = ST.log.slice(-60).join("\n"); el.hidden = !ST.log.length; el.scrollTop = el.scrollHeight; } }
function renderProgress() {
  var p = ST.prog, box = $q("#ex_prog"), t = $q("#ex_ptxt"); if (!box) return; box.hidden = !(ST.running || (p && p.done < p.total)); if (p) { box.firstElementChild.style.width = (100 * p.done / Math.max(1, p.total)) + "%"; t.textContent = p.text + (ST.running ? " · " + Math.round((Date.now() - ST.t0) / 1000) + " s" : ""); }
}
function refreshStatus() {
  var d = $q("#lb_data"), r = $q("#lb_rt"), b = $q("#lb_load"); if (!d) return;
  d.textContent = ST.data ? "Data: " + ST.data.meta.n.toLocaleString() + " beats loaded" : "Data: loading…";
  var rt = { idle: "Python: not started", loading: "Python: " + ST.rtText, data: "Python: " + ST.rtText, ready: "Python: ready", error: "Python: " + ST.rtText }[ST.rt] || ""; r.textContent = rt;
  r.className = ST.rt === "ready" ? "ok" : ""; b.hidden = ST.rt !== "idle"; var st = $q("#ex_state"); if (st) st.textContent = ST.running ? "running…" : (ST.pendingRun ? ST.rtText : (ST.rt === "ready" ? "" : "The runtime starts on the first run (about 50 MB, cached afterwards)."));
  var run = $q("#ex_run"), stop = $q("#ex_stop"); if (run) { run.disabled = ST.running || (ST.rt === "loading" || ST.rt === "data"); var c3 = $q("#ex_cmp3"); if (c3) c3.disabled = run.disabled; } if (stop) stop.hidden = !ST.running;
}
function refreshAll() { refreshStatus(); renderProgress(); renderLog(); renderResults(); if (ST.afterReady && ST.rt === "ready") { var f = ST.afterReady; ST.afterReady = null; f(); } }

/* ---------------- results: leaderboard, charts, detail, compare ---------------- */
function addRun(r) {
  r.id = "run" + (++ST.idCounter); r.live = true; ST.runs.push(r); ST.detail = r.id; ST.sel = [r.id]; renderResults();
}
var SF = { "Sheikh 465": "465", "Sheikh waveform only (1–450)": "waveform", "Sheikh morphology only (451–465)": "morphology 15", "Abel 13": "Abel 13", "Sheikh 465 + Abel 13": "465+13" }, SV = { "5-fold, patients held out": "5-fold", "Train EmpkinS → test Guardian": "E→G", "Train Guardian → test EmpkinS": "G→E", "Hold out one patient": "1 patient" };
function shortName(r) {
  if (r.ref === "pub") return "Abel 2026 published";
  if (r.ref === "ours") return (SF[r.feat] || r.feat) + " · " + r.model + " (full)";
  if (!r.feat || r.feat === "none") return r.name.length > 28 ? r.name.slice(0, 27) + "…" : r.name;
  return "live: " + (SF[r.feat] || r.feat) + " · " + (r.model || "") + " · " + (SV[r.val] || r.val || "");
}
function allRows() { return (ST.refs || []).map(function (x) { return Object.assign({ live: false }, x); }).concat(ST.runs.map(function (r) { return { id: r.id, name: r.name, feat: r.info.features, model: r.info.model, val: r.info.validation, MAE: r.MAE, SD: r.SD, MedAE: r.MedAE, n: r.n, sec: r.info.seconds, live: true, run: r, short: null }; })).map(function (r) { r.short = shortName(r); return r; }); }
function renderResults() {
  var el = $q("#ex_res"); if (!el || !ST.data) return; var rows = allRows().sort(function (a, b) { return a.MAE - b.MAE; }), bestLive = null; rows.forEach(function (r) { if (r.live && (!bestLive || r.MAE < bestLive.MAE)) bestLive = r; });
  var body = rows.map(function (r) { var cls = (r.live ? "live" : "ref") + (bestLive && r.id === bestLive.id ? " best" : ""), chk = r.live ? '<input type="checkbox" data-id="' + r.id + '"' + (ST.sel.indexOf(r.id) >= 0 ? " checked" : "") + ' aria-label="select ' + esc(r.name) + '">' : "";
    return '<tr class="' + cls + '" data-id="' + r.id + '"><td>' + chk + "</td><td>" + esc(r.name) + (r.live ? "" : ' <span class="tag">' + (r.ref === "pub" ? "published" : "full settings") + "</span>") + "</td><td>" + esc(r.val || "") + '</td><td class="n">' + r.MAE.toFixed(2) + '</td><td class="n">' + (r.SD != null ? r.SD.toFixed(2) : "") + '</td><td class="n">' + (r.MedAE != null ? r.MedAE.toFixed(2) : "") + '</td><td class="n">' + (r.sec != null ? r.sec + " s" : "") + "</td></tr>"; }).join("");
  var nl = ST.runs.length;
  el.innerHTML = '<div class="lb-row" style="margin:0 0 8px"><h3 style="margin:0">Results</h3><span class="muted">' + (nl ? nl + " run" + (nl > 1 ? "s" : "") + " in this session" : "Run an experiment to add it here, next to the verified results") + '</span><span class="grow"></span><button class="chip" id="rs_cmp" ' + (ST.sel.length === 2 ? "" : "disabled") + '>Compare the 2 selected</button><button class="chip" id="rs_clear" ' + (nl ? "" : "disabled") + ">Clear runs</button></div>" +
    '<div class="lb-grid"><div><div class="tablewrap" style="max-height:340px"><table class="lbt"><thead><tr><th></th><th>Run</th><th>Validation</th><th class="n">MAE ms</th><th class="n">SD</th><th class="n">Median</th><th class="n">Time</th></tr></thead><tbody>' + body + '</tbody></table></div><p class="muted" style="font-size:12px;margin:6px 0 0">Live runs use the settings you chose (often reduced, one repeat). "Full settings" rows are the verified results of the full study; the published row is Abel et al. (2026), different cohort and folds.</p></div><div><canvas id="rs_bar"></canvas></div></div>' +
    '<div id="rs_detail"></div>';
  el.querySelectorAll("tbody tr").forEach(function (tr) { tr.onclick = function (e) { if (e.target.tagName === "INPUT") return; if (tr.classList.contains("live")) { ST.detail = tr.dataset.id; renderDetail(); } }; });
  el.querySelectorAll('input[type=checkbox]').forEach(function (cb) { cb.onchange = function () { var id = cb.dataset.id, k = ST.sel.indexOf(id); if (cb.checked && k < 0) ST.sel.push(id); if (!cb.checked && k >= 0) ST.sel.splice(k, 1); if (ST.sel.length > 2) ST.sel.shift(); renderResults(); }; });
  $q("#rs_clear").onclick = function () { ST.runs = []; ST.sel = []; ST.detail = null; renderResults(); }; $q("#rs_cmp").onclick = function () { ST.detail = "cmp"; renderDetail(); };
  drawBar(rows); renderDetail();
}
function drawBar(rows) {
  var c = $q("#rs_bar"); if (!c) return; var o = cvs(c, rows.length * 24 + 40), g = o.g, w = o.w, mut = css("--muted"), line = css("--line"), fg = css("--fg"), acc = css("--accent"), lx = Math.min(190, w * .42), x0 = lx, x1 = w - 52, mx = Math.max(12, Math.ceil(Math.max.apply(null, rows.map(function (r) { return r.MAE; })) / 2) * 2), fx = function (v) { return x0 + v / mx * (x1 - x0); };
  g.font = "12px " + css("--font-body"); for (var t = 0; t <= mx; t += 2) { g.strokeStyle = line; g.beginPath(); g.moveTo(fx(t), 4); g.lineTo(fx(t), rows.length * 24 + 6); g.stroke(); g.fillStyle = mut; g.textAlign = "center"; g.fillText(t, fx(t), rows.length * 24 + 20); } g.fillText("MAE (ms)", (x0 + x1) / 2, rows.length * 24 + 36);
  rows.forEach(function (r, k) { var y = 6 + k * 24; g.fillStyle = fg; g.textAlign = "right"; g.fillText(r.short, lx - 8, y + 14); g.fillStyle = r.live ? acc : (r.ref === "pub" ? COL.red : "#8fa7ba"); g.globalAlpha = r.live ? 1 : .75; g.fillRect(x0, y + 3, fx(r.MAE) - x0, 15); g.globalAlpha = 1; g.fillStyle = mut; g.textAlign = "left"; g.fillText(r.MAE.toFixed(2), fx(r.MAE) + 6, y + 15); });
}
function renderDetail() {
  var el = $q("#rs_detail"); if (!el) return; if (ST.detail === "cmp" && ST.sel.length === 2) { return renderCompare(el); }
  var r = ST.runs.filter(function (x) { return x.id === ST.detail; })[0]; if (!r) { el.innerHTML = ""; return; }
  el.innerHTML = '<h3 style="margin:16px 0 6px">Run detail · ' + esc(r.name) + '</h3><div class="lb-grid3"><div><canvas id="dt_sc"></canvas></div><div><canvas id="dt_hi"></canvas></div><div><canvas id="dt_pp"></canvas></div></div><p class="muted" style="font-size:12px;margin:4px 0 0">MAE ' + r.MAE.toFixed(2) + " ms · SD " + r.SD.toFixed(2) + " · median " + r.MedAE.toFixed(2) + " · mean error " + r.ME.toFixed(2) + " ms · " + r.n.toLocaleString() + " beats tested</p>";
  var mut = css("--muted"), line = css("--line"), acc = css("--accent"), fg = css("--fg"), o = cvs($q("#dt_sc"), 260), g = o.g, w = o.w, h = o.h, x0 = 40, x1 = w - 8, y0 = h - 32, y1 = 8, lo = -40, hi = 260, fx = function (v) { return x0 + (v - lo) / (hi - lo) * (x1 - x0); }, fy = function (v) { return y0 - (v - lo) / (hi - lo) * (y0 - y1); }, tk = [0, 100, 200];
  drawAxes(g, x0, x1, y0, y1, tk, tk, fx, fy, "true B (ms)", "predicted", mut, line); g.strokeStyle = mut; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(fx(lo), fy(lo)); g.lineTo(fx(hi), fy(hi)); g.stroke(); g.setLineDash([]); g.fillStyle = acc; g.globalAlpha = .35; r.scatter.forEach(function (p) { g.beginPath(); g.arc(fx(Math.max(lo, Math.min(hi, p[0]))), fy(Math.max(lo, Math.min(hi, p[1]))), 2, 0, 7); g.fill(); }); g.globalAlpha = 1;
  o = cvs($q("#dt_hi"), 260); g = o.g; w = o.w; h = o.h; var hx0 = 30, hx1 = w - 8, mxh = Math.max.apply(null, r.hist), bw = (hx1 - hx0) / r.hist.length; g.font = "11px " + css("--font-body"); r.hist.forEach(function (v, k) { g.fillStyle = k === 14 || k === 15 ? COL.ok : acc; g.fillRect(hx0 + k * bw + 1, h - 32 - v / mxh * (h - 46), bw - 2, v / mxh * (h - 46)); });
  g.fillStyle = mut; g.textAlign = "center"; [-60, -30, 0, 30, 60].forEach(function (t) { g.fillText(t, hx0 + (t + 60) / 4 * bw, h - 16); }); g.fillText("error (ms), predicted − true", (hx0 + hx1) / 2, h - 3);
  o = cvs($q("#dt_pp"), 260); g = o.g; w = o.w; h = o.h; var P = r.per.slice().sort(function (a, b) { return a[1] - b[1]; }), mxp = Math.max.apply(null, P.map(function (p) { return p[1]; })), bw2 = (w - 40) / P.length;
  P.forEach(function (p, k) { var ph = p[1] / mxp * (h - 50); g.fillStyle = ST.data.meta.patients[p[0]].dataset === "EmpkinS" ? COL.sheikh : COL.abel; g.fillRect(30 + k * bw2 + 1, h - 30 - ph, Math.max(1, bw2 - 2), ph); }); g.fillStyle = mut; g.font = "11px " + css("--font-body"); g.textAlign = "left"; g.fillText("MAE per patient (ms) · blue EmpkinS · orange Guardian · max " + mxp.toFixed(1), 30, 12); g.fillText("patients sorted by error", 30, h - 10);
}
function renderCompare(el) {
  var A = ST.runs.filter(function (x) { return x.id === ST.sel[0]; })[0], B = ST.runs.filter(function (x) { return x.id === ST.sel[1]; })[0]; if (!A || !B) { el.innerHTML = ""; return; }
  var pa = {}, pb = {}; A.per.forEach(function (p) { pa[p[0]] = p[1]; }); B.per.forEach(function (p) { pb[p[0]] = p[1]; }); var ids = Object.keys(pa).filter(function (k) { return pb[k] != null; }).map(Number); ids.sort(function (x, y) { return pa[x] - pa[y]; });
  var better = ids.filter(function (k) { return pa[k] < pb[k]; }).length, diff = ids.reduce(function (s, k) { return s + pa[k] - pb[k]; }, 0) / (ids.length || 1);
  el.innerHTML = '<h3 style="margin:16px 0 6px">Comparison · per patient</h3><div class="lb-row"><span class="swatch" style="background:' + COL.sheikh + '"></span>A · ' + esc(A.name) + '<span class="swatch" style="background:' + COL.abel + '"></span>B · ' + esc(B.name) + '</div><canvas id="cm_c"></canvas>' +
    "<p style=\"margin:6px 0 0\"><b>" + (diff < 0 ? "A" : "B") + " is better:</b> A has the lower error for " + better + " of " + ids.length + " patients; mean per-patient MAE A " + A.subjMAE.toFixed(2) + " vs B " + B.subjMAE.toFixed(2) + " ms (difference " + diff.toFixed(2) + " ms). Overall MAE A " + A.MAE.toFixed(2) + " vs B " + B.MAE.toFixed(2) + " ms.</p>";
  var c = $q("#cm_c"), o = cvs(c, ids.length * 15 + 40), g = o.g, w = o.w, mut = css("--muted"), line = css("--line"), fg = css("--fg"), lx = 70, x0 = lx, x1 = w - 20, mx = Math.max(10, Math.ceil(Math.max.apply(null, ids.map(function (k) { return Math.max(pa[k], pb[k]); })) / 5) * 5), fx = function (v) { return x0 + v / mx * (x1 - x0); };
  g.font = "11px " + css("--font-body"); for (var t = 0; t <= mx; t += 5) { g.strokeStyle = line; g.beginPath(); g.moveTo(fx(t), 4); g.lineTo(fx(t), ids.length * 15 + 8); g.stroke(); g.fillStyle = mut; g.textAlign = "center"; g.fillText(t, fx(t), ids.length * 15 + 22); } g.fillText("MAE per patient (ms)", (x0 + x1) / 2, ids.length * 15 + 36);
  ids.forEach(function (k, j) { var y = 10 + j * 15; g.fillStyle = fg; g.textAlign = "right"; g.fillText(ST.data.meta.patients[k].id, lx - 8, y + 4); g.strokeStyle = mut; g.beginPath(); g.moveTo(fx(pa[k]), y); g.lineTo(fx(pb[k]), y); g.stroke(); g.fillStyle = COL.sheikh; g.beginPath(); g.arc(fx(pa[k]), y, 4, 0, 7); g.fill(); g.fillStyle = COL.abel; g.beginPath(); g.arc(fx(pb[k]), y, 4, 0, 7); g.fill(); });
}

/* ---------------- shell ---------------- */
var CSS = ".lab .lb-row{display:flex;flex-wrap:wrap;gap:8px 14px;align-items:center;margin-bottom:10px}.lab .grow{flex:1;min-width:140px;display:flex;gap:8px;align-items:center}.lab .grow input[type=range]{flex:1}.lab .muted{color:var(--muted);font-size:12px}.lab .mono{font-variant-numeric:tabular-nums}" +
  ".lab .seg{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden;background:var(--surface)}.lab .seg button{background:none;border:0;padding:7px 14px;color:var(--muted);cursor:pointer;font-weight:500}.lab .seg button[aria-selected=true]{background:var(--accent-soft);color:var(--fg);font-weight:600}" +
  ".lab .lb-top{display:flex;flex-wrap:wrap;gap:10px 18px;align-items:center;justify-content:space-between;margin-bottom:14px}.lab .lb-top .st{display:flex;flex-wrap:wrap;gap:6px 16px;align-items:center;font-size:12px;color:var(--muted)}.lab .ok{color:var(--ok)}" +
  ".lab .lb-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin-bottom:12px}.lab .lb-grid3{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}@media(max-width:900px){.lab .lb-grid,.lab .lb-grid3{grid-template-columns:minmax(0,1fr)}}" +
  ".lab .panel{margin-bottom:12px}.lab .panel h3{margin:0 0 6px}.lab select,.lab input[type=number]{font:inherit;color:var(--fg);background:var(--surface);border:1px solid var(--line);border-radius:6px;padding:5px 8px}.lab label.sm{font-size:12px;color:var(--muted);display:flex;gap:6px;align-items:center}.lab label.chk{display:flex;gap:6px;align-items:center;font-size:13px}" +
  ".lab .lbt{width:100%;border-collapse:collapse;font-size:12.5px;font-variant-numeric:tabular-nums}.lab .lbt th,.lab .lbt td{padding:4px 8px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}.lab .lbt th{position:sticky;top:0;background:var(--surface);color:var(--muted);font-weight:500;font-size:11px;text-transform:uppercase;letter-spacing:.04em}.lab .lbt .n{text-align:right}.lab .lbt td.muted{white-space:normal;min-width:160px}.lab .lbt tr.best td{background:var(--accent-soft)}.lab .lbt tr.ref td{color:var(--muted)}.lab .lbt tr.live{cursor:pointer}" +
  ".lab .tag{font-size:10px;border:1px solid var(--line);border-radius:999px;padding:0 6px;color:var(--muted)}.lab .prog{height:8px;border-radius:4px;background:var(--line);overflow:hidden;margin-top:10px}.lab .prog>div{height:100%;background:var(--accent);width:0;transition:width .3s}" +
  ".lab .log{background:#071014;color:#cfe3ee;border-radius:6px;padding:8px 10px;font:12px/1.5 ui-monospace,Menlo,Consolas,monospace;max-height:150px;overflow:auto;margin:8px 0 0;white-space:pre-wrap}.lab .swatch{width:10px;height:10px;border-radius:2px;display:inline-block;margin-left:10px;margin-right:4px}" +
  ".lab .CodeMirror{border:1px solid var(--line);border-radius:6px;font-size:12.5px;height:360px;background:var(--surface);color:var(--fg)}.lab .CodeMirror-gutters{background:var(--bg);border-right:1px solid var(--line)}.lab .CodeMirror-linenumber{color:var(--muted)}.lab .CodeMirror-cursor{border-left-color:var(--fg)}" +
  ".lab .cm-keyword{color:#c0392b}.lab .cm-def{color:#0b6e99}.lab .cm-string{color:#18794e}.lab .cm-number{color:#b05d00}.lab .cm-comment{color:var(--muted)}.lab .cm-builtin{color:#7b4cc0}.lab .cm-variable{color:var(--fg)}.lab .CodeMirror-selected{background:var(--accent-soft)!important}";
function mountPane() {
  var pane = $q("#lb_pane"); if (!pane || !ST.data) return; var p = ST.pane;
  document.querySelectorAll(".lab .seg button").forEach(function (b) { b.setAttribute("aria-selected", String(b.dataset.p === p)); });
  if (p === "explore") { pane.innerHTML = explorePane(); bindExplore(); updateExplore(); }
  else if (p === "compare") { pane.innerHTML = comparePane(); bindCompare(); updateCompare(); }
  else { applyPresetIfNeeded(); pane.innerHTML = expPane(); $q("#ex_params").innerHTML = paramsHtml(); bindExp(); initEditor().then(function () { renderLog(); }); loadRefs().then(renderResults); refreshAll(); }
}
function applyPresetIfNeeded() { if (ST.cfg.preset !== "custom" && !ST.presetApplied) { applyPreset(); ST.presetApplied = true; } }
function bindExplore() {
  $q("#ex_p").onchange = function () { ST.pat = parseInt(this.value); ST.beat = 0; ST.verify = null; mountPane(); };
  $q("#ex_b").oninput = function () { ST.beat = parseInt(this.value); updateExplore(); };
  var step = function (d) { var p = ST.data.meta.patients[ST.pat], nb = p.end - p.start; ST.beat = (Math.min(ST.beat, nb - 1) + d + nb) % nb; updateExplore(); };
  $q("#ex_prev").onclick = function () { step(-1); }; $q("#ex_next").onclick = function () { step(1); };
  $q("#ex_sa").onchange = function () { ST.showAbel = this.checked; drawBeat(); }; $q("#ex_ss").onchange = function () { ST.showSheikh = this.checked; drawBeat(); };
  $q("#ex_ver").onclick = verifyBeat; $q("#ex_all").ontoggle = fillAll;
}
function bindCompare() {
  $q("#cp_f").onclick = function (e) { var b = e.target.closest(".chip"); if (!b) return; ST.fltDs = b.dataset.ds; mountPane(); };
  $q("#cp_sel").onchange = function () { ST.est = this.value; drawScatter(subsetIdx()); };
}
var wrapEl = null;
function mount(el) {
  if (!document.getElementById("lab-css")) { var s = document.createElement("style"); s.id = "lab-css"; s.textContent = CSS; document.head.appendChild(s); }
  el.innerHTML = '<div class="lab"><div class="lb-top"><div><h2 style="margin:0 0 4px">Lab</h2><div class="st"><span id="lb_data">Data: loading…</span><span id="lb_rt">Python: not started</span><button class="chip f" id="lb_load">Start Python now</button></div></div>' +
    '<div class="seg" role="tablist"><button data-p="explore" role="tab">Beat explorer</button><button data-p="compare" role="tab">Feature comparison</button><button data-p="exp" role="tab">Experiments</button></div></div><div id="lb_pane"><p class="note" id="lb_wait">Preparing the data (about 18 MB, decrypted in your browser)…</p></div></div>';
  el.querySelectorAll(".lab .seg button").forEach(function (b) { b.onclick = function () { ST.pane = b.dataset.p; mountPane(); }; });
  $q("#lb_load").onclick = function () { startRuntime(); };
  ST.data ? (refreshStatus(), mountPane()) : loadData(function (t) { var w = $q("#lb_wait"); if (w) w.textContent = t; }).then(function () { refreshStatus(); mountPane(); }).catch(function (e) { var w = $q("#lb_wait"); if (w) w.textContent = "Could not load the Lab data (" + e + ")."; });
  refreshStatus();
}
function redraw() { if (!ST.data) return; if (ST.pane === "explore") { drawBeat(); drawMini(); drawEstimators(); } else if (ST.pane === "compare") updateCompare(); else renderResults(); }
window.LabApp = { mount: mount, redraw: redraw, state: ST };
})();
