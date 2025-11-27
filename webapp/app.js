const API_BASE = "";

if ("scrollRestoration" in history) {history.scrollRestoration = "manual";}
window.addEventListener("load", () => {window.scrollTo({ top: 0, left: 0, behavior: "auto" });});
const GRAPH_PLACEHOLDERS = Object.create(null);

const $ = s => document.querySelector(s);
const $$ = s => Array.from(document.querySelectorAll(s));
const slug = s => (s||"").toLowerCase().replace(/[^a-z0-9]+/g,"-").replace(/(^-|-$)/g,"") || `id-${Math.random().toString(36).slice(2)}`;
const qColor = q => q==="OK" ? "#19c37d" : q==="CHECK" ? "#ffb266" : "#e05a61";

function normaliseQuality(r) {
  let q = (r.quality_pred || "").toUpperCase();
  if (r.qc && r.qc.capped) {
    q = "LOW";
  }
  if (q !== "OK" && q !== "CHECK" && q !== "LOW") {
    q = "CHECK";
  }
  return q;
}

const safe = (x, dp = 2) =>
  (x === null || x === undefined || Number.isNaN(x))
    ? "—"
    : Number(x).toFixed(dp);

const state = { files: [], quick: [], results: [], room_summary: null, aiSummary: "", editing: false, history: []};
let detailsOpenAll = true;
function setFiles(list) { state.files = Array.from(list || []); renderActiveRecordings(); }
function getFiles()     { return state.files; }
window.activeFiles = state.files;

let playback = { audio: null, id: null };

function clearPlaybackState() {
  if (playback.audio) {
    try { playback.audio.pause(); } catch (_) {}
  }
  // reset all buttons
  document.querySelectorAll(".wave-play.is-playing").forEach(btn => {
    btn.classList.remove("is-playing");
    btn.textContent = "▶";
  });
  playback = { audio: null, id: null };
}

function togglePlay(fileId) {
  // If same file and currently playing → stop
  if (playback.audio && playback.id === fileId && !playback.audio.paused) {
    clearPlaybackState();
    return;
  }

  // Stop anything else
  clearPlaybackState();

  const item = (state.files || []).find(f => f.id === fileId);
  if (!item || !item.url) {
    console.warn("No URL to play for", fileId);
    return;
  }

  const audio = new Audio(item.url);
  playback = { audio, id: fileId };

  const btn = document.querySelector(`.wave-play[data-id="${fileId}"]`);
  if (btn) {
    btn.classList.add("is-playing");
    btn.textContent = "⏸";
  }

  audio.addEventListener("ended", () => {
    clearPlaybackState();
  });

  audio.play().catch(err => {
    console.error("Playback failed", err);
    clearPlaybackState();
  });
}

let currentModel = "phone_mic";
const MODEL_SEQUENCE = ["phone_mic", "external_mic"];

const el = {
  apiDot: $("#api-dot"), apiMsg: $("#api-msg"), apiMsgLabel: $("#api-msg-label"), apiModel: $("#api-model"),
  drop: $("#drop"), pick: $("#pick"), fileInput: $("#file-input"),
  recGrid: $("#recGrid"), editToggle: $("#editToggle"), doneToggle: $("#doneToggle"),
  recordBtn: $("#recordBtn"), analyzeBtn: $("#analyzeBtn"),
  results: $("#results"), resultsEmpty: $("#resultsEmpty"),
  toggleDetailsAll: $("#toggleDetailsAll"),
  ctxUse: $("#ctxUse"), ctxGoal: $("#ctxGoal"), ctxLabel: $("#ctxLabel"),
  preset: $("#preset"), savePlots: $("#savePlots"),
  fbToggle: $("#fbToggle"), fbBody: $("#fbBody"), fbSummary: $("#fbSummary"), fbBullets: $("#fbBullets"),
  graphsToggle: $("#graphsToggle"), graphsBody: $("#graphsBody"), graphsHost: $("#graphsHost"), graphsEmpty: $("#graphsEmpty"),
  saveSessionBtn: $("#saveSessionBtn"),
  historyBody: $("#historyBody"),
};

// Click API message to toggle between model profiles
if (el.apiModel) {
  el.apiModel.addEventListener("click", async () => {
    const idx  = MODEL_SEQUENCE.indexOf(currentModel);
    const next = MODEL_SEQUENCE[(idx + 1) % MODEL_SEQUENCE.length];

    try {
      const res = await fetch(`${API_BASE}/model/${next}`, { method: "POST" });
      const j   = await res.json();

      if (j.status === "ok") {
        currentModel = j.model || next;

        await checkAPI();

        if (typeof toast === "function") {
          toast(`Switched model to ${currentModel}`, "ok");
        }
      } else {
        console.warn("Model switch error", j);
        if (typeof toast === "function") {
          toast("Model switch error", "err");
        }
      }
    } catch (err) {
      console.error("Model switch failed", err);
      if (typeof toast === "function") {
        toast("Model switch failed", "err");
      }
    }
  });
}

document.addEventListener('DOMContentLoaded', () => {
  el.fbSummary = el.fbSummary || document.getElementById('fbSummary');
  el.fbBullets = el.fbBullets || document.getElementById('fbBullets');
});

const pickBtn   = document.getElementById('pick');
const fileInput = document.getElementById('file-input');
pickBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', () => setFiles(fileInput.files));

function ensureFbSummary(){
  const body = document.getElementById('fbBody');
  const bullets = document.getElementById('fbBullets');
  if (!body || !bullets) return;

  if (!document.getElementById('fbSummary')){
    const div = document.createElement('div');
    div.id = 'fbSummary';
    div.style.margin = '0 0 6px 0';
    body.insertBefore(div, bullets);
  }
}

// ---- API status ----
async function checkAPI() {
  try {
    const r = await fetch(`${API_BASE}/health`);
    const j = await r.json();

    const ok = (j.status === "ok");
    el.apiDot.style.background = ok ? "#19c37d" : "#f5c542";

    const msg = j.message || (ok ? "API ready" : "API error");

    const m = msg.match(/^(.*?)(\s*\(models\\.*\))$/);
    if (m) {
      el.apiMsgLabel.textContent = m[1].trim();
      el.apiModel.textContent = m[2].trim();
    } else {
      el.apiMsgLabel.textContent = msg;
      el.apiModel.textContent = "";
    }

    // Track active model
    if (j.current_model) {
      currentModel = j.current_model;
    } else if (m) {
      const mm = /\(models\\([^)\s]+)\)/.exec(m[2]);
      if (mm) currentModel = mm[1];
    }
  } catch (err) {
    console.error("[/health] failed", err);
    el.apiDot.style.background = "#f5c542";
    el.apiMsgLabel.textContent = "API not reachable";
    el.apiModel.textContent = "";
  }
}

// ---- Modals ----
$$(".pill").forEach(p=>p.addEventListener("click",()=>document.getElementById(p.dataset.modal).classList.add("open")));
$$(".modal .close").forEach(b=>b.addEventListener("click",()=>document.getElementById(b.dataset.close).classList.remove("open")));
$$(".modal").forEach(m=>m.addEventListener("click",(e)=>{if(e.target===m)m.classList.remove("open");}));

// ---- Drag & drop / file picking ----
["dragenter","dragover"].forEach(evt=>el.drop.addEventListener(evt,(e)=>{e.preventDefault();e.currentTarget.style.borderColor="#4F0769";}));
["dragleave","drop"].forEach(evt=>el.drop.addEventListener(evt,(e)=>{e.preventDefault();e.currentTarget.style.borderColor="var(--line)";}));
el.drop.addEventListener("drop",(e)=>addFiles([...e.dataTransfer.files].filter(f=>f.type.startsWith("audio/"))));
el.pick.addEventListener("click",()=>el.fileInput.click());
el.fileInput.addEventListener("change",()=>addFiles([...el.fileInput.files]));

// ---- Active recordings ----
function addFiles(files){
  const next = files.map(f=>({id:crypto.randomUUID(), file:f, name:f.name, url:URL.createObjectURL(f), quality:null}));
  state.files.push(...next);
  renderRecs();
  quickQuality(next).catch(()=>{});
}

function renderRecs(){
  el.recGrid.innerHTML = "";
  el.recGrid.classList.toggle("editing", state.editing);

  state.files.forEach(item=>{
    const card = document.createElement("div");
    card.className = "rec-card";
    const id = item.id;

    card.innerHTML = `
      <div class="rec-title" title="${item.name}">${item.name}</div>
      <div class="rec-x-bubble" data-id="${id}">×</div>
      <div class="rec-row">
        <div class="wave">
          <canvas class="wavecan"></canvas>
          <button type="button" class="wave-play" data-id="${id}">▶</button>
        </div>
        <div class="qchip ${chipClass(item.quality)}" title="${qualityTitle(item)}">
          ${(item.quality||"").toUpperCase()||"—"}
        </div>
      </div>`;

    el.recGrid.appendChild(card);

    // draw waveform into canvas
    drawWaveformToCanvas(item, card.querySelector("canvas.wavecan"));

    // delete bubble
    card.querySelector(".rec-x-bubble").addEventListener("click", ()=>{
      state.files = state.files.filter(f=>f.id!==id);
      renderRecs();
    });

    // play/pause overlay button
    const playBtn = card.querySelector(".wave-play");
    if (playBtn) {
      playBtn.addEventListener("click", (ev) => {
        ev.stopPropagation();
        togglePlay(id);
      });
    }
  });
}

const chipClass = q => !q ? "" : (q.toUpperCase()==="OK"?"qc-ok":q.toUpperCase()==="CHECK"?"qc-check":"qc-low");
function qualityTitle(item){
  const res = state.quick.find(r=>r.name===item.name) || state.results.find(r=>r.name===item.name);
  const r2 = res?.qc?.r2;
  const q = (item.quality||"").toUpperCase() || "—";
  return r2!=null ? `${q} (r² ${Number(r2).toFixed(2)})` : q;
}

async function drawWaveformToCanvas(item, canvas){
  const ctx=canvas.getContext("2d");
  try{
    const acx=new (window.AudioContext||window.webkitAudioContext)();
    const buf=await (await fetch(item.url)).arrayBuffer();
    const audio=await acx.decodeAudioData(buf);
    const data=audio.getChannelData(0);
    const W=canvas.parentElement.clientWidth, H=canvas.parentElement.clientHeight;
    canvas.width=W; canvas.height=H;
    ctx.fillStyle="#0a1a26"; ctx.fillRect(0,0,W,H);
    ctx.strokeStyle="#7fb5ff"; ctx.lineWidth=1; ctx.beginPath();
    const step=Math.ceil(data.length/W), amp=H/2.2;
    for(let x=0;x<W;x++){
      let min=1,max=-1, start=x*step;
      for(let i=0;i<step;i++){const v=data[start+i]||0; if(v<min)min=v; if(v>max)max=v;}
      ctx.moveTo(x,(1+min)*amp); ctx.lineTo(x,(1+max)*amp);
    }
    ctx.stroke();
  }catch{}
}

// Edit toggles
el.editToggle.addEventListener("click",()=>{state.editing=true;el.editToggle.style.display="none";el.doneToggle.style.display="inline-flex";renderRecs();});
el.doneToggle.addEventListener("click",()=>{state.editing=false;el.doneToggle.style.display="none";el.editToggle.style.display="inline-flex";renderRecs();});

// ---- Quick quality preview ----
async function quickQuality(items){
  if(!items.length) return;
  const fd = new FormData();
  items.forEach(f => fd.append("files", f.file, f.name));
  const presetVal = el.preset?.value || "";
  const perBandFlag = presetVal ? "1" : "0";
  const bandLimitFlag = (presetVal === "speech") ? "1" : "0";

  fd.append("context_use",  (el.ctxUse?.value  || "").trim());
  fd.append("context_goal", (el.ctxGoal?.value || "").trim());
  fd.append("context_label", (el.ctxLabel?.value || "").trim());
  fd.append("preset",     presetVal);
  fd.append("per_band",   perBandFlag);
  fd.append("band_limit", bandLimitFlag);
  fd.append("save_plots", "1");
  try{
    const r=await fetch(`${API_BASE}/analyze`,{method:"POST",body:fd});
    const j = await r.json();
    const results = j.cards || j.results || [];
    results.forEach(res => {
      const idx = state.quick.findIndex(x => x.name === res.name);
      if (idx >= 0) state.quick[idx] = res; else state.quick.push(res);
      const f = state.files.find(x => x.name === res.name);
      if (f) f.quality = normaliseQuality(res);
    });
    renderRecs();
    renderFeedback();
  }catch{}
}

// ---- Recording ----
let mediaRec=null, chunks=[];
$("#recordBtn").addEventListener("click",async ()=>{
  if(!mediaRec){
    try{
      const stream=await navigator.mediaDevices.getUserMedia({audio:true});
      mediaRec=new MediaRecorder(stream); chunks=[];
      mediaRec.ondataavailable=e=>chunks.push(e.data);
      mediaRec.onstop=()=>{
        const blob=new Blob(chunks,{type:"audio/webm"});
        addFiles([new File([blob],`recording_${Date.now()}.webm`,{type:blob.type})]);
        mediaRec=null; $("#recordBtn").innerHTML=`<span class="rec-dot"></span> Record`;
      };
      mediaRec.start(); $("#recordBtn").textContent="■ ‎ ‎ Stop";
    }catch{ alert("Microphone not available."); }
  }else mediaRec.stop();
});

// ---- Analyse ----
let ANALYZE_IN_FLIGHT = false;

function setAnalyzeLoading(on){
  const btn = document.getElementById('analyzeBtn');
  btn.innerHTML = on ? `<span class="spinner"></span>` : "► Analyse";
  btn.disabled = !!on;
}

async function doAnalyze(){
  const files = state.files;                       // [{ file, name, ... }]
  if (!files || !files.length) { alert("Add recordings first."); return; }
  setAnalyzeLoading(true);
  const fd = new FormData();
  for (const it of files) fd.append('files', it.file, it.name);   // <-- File objects

  fd.append('context_use',   (el.ctxUse?.value  || '').trim());
  fd.append('context_goal',  (el.ctxGoal?.value || '').trim());
  fd.append('context_label', (el.ctxLabel?.value || '').trim());

  const presetVal    = el.preset?.value || "";
  const perBandFlag  = presetVal ? "1" : "0";
  const bandLimitFlag = (presetVal === "speech") ? "1" : "0";

  fd.append('preset',     presetVal);
  fd.append('per_band',   perBandFlag);
  fd.append('band_limit', bandLimitFlag);
  fd.append('save_plots', el.savePlots?.checked ? '1' : '0');
  let raw = "";
  try {
    const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: fd });
    raw = await res.text();
    const data = raw ? JSON.parse(raw) : {};

    state.results      = Array.isArray(data.results) ? data.results : (data.cards || []);
    state.room_summary = data.room_summary || null;
    state.aiSummary    = (data.ai || data.ai_summary || "").trim();
    detailsOpenAll = true;
    const toggle = document.getElementById("toggleDetailsAll");
    if (toggle) toggle.textContent = "Hide details";

    renderResults();
    renderFeedback();
    graphsOpen = true;
    el.graphsBody.classList.add("open");
    el.graphsToggle.textContent = "▾";
    renderGraphs();
  } catch (err) {
    console.error('[/analyze] failed', err, raw.slice(0,200));
    //toast?.('Analyse failed', 'err');
  } finally {
    setAnalyzeLoading(false);
  }
}

document.getElementById('analyzeBtn').addEventListener('click', () => doAnalyze().catch(e => alert(e.message)));

function saveSession() {
  if (!state.results || !state.results.length) {
    alert("Run an analysis before saving a session.");
    return;
  }

  const labelRaw = (el.ctxLabel?.value || "").trim();
  const label = labelRaw || "Untitled session";
  const when = new Date();

  const snapshot = {
    id: `sess-${when.getTime()}`,
    savedAt: when.toISOString(),
    title: label,
    timestampLabel: formatSessionTimestamp(when),
    context: {
      use: (el.ctxUse?.value || "").trim(),
      goal: (el.ctxGoal?.value || "").trim(),
      label: labelRaw,
    },
    preset: el.preset?.value || "",
    save_plots: !!(el.savePlots?.checked),
    results: JSON.parse(JSON.stringify(state.results || [])),
    room_summary: JSON.parse(JSON.stringify(state.room_summary || null)),
    aiSummary: state.aiSummary || "",
    resultsHtml: document.getElementById("results")?.innerHTML || "",
    feedbackHtml: document.getElementById("fbBody")?.innerHTML || "",
  };

  state.history.unshift(snapshot);
  renderHistory();

  state.files = [];
  state.quick = [];
  state.results = [];
  state.room_summary = null;
  state.aiSummary = "";
  renderRecs();
  renderResults();
  renderGraphs();
  renderFeedback();
  if (el.ctxUse) el.ctxUse.value = "";
  if (el.ctxGoal) el.ctxGoal.value = "";
  if (el.ctxLabel) el.ctxLabel.value = "";
  if (el.preset) el.preset.value = "";
}

if (el.saveSessionBtn) {
  el.saveSessionBtn.addEventListener("click", saveSession);
}

// ---- Results ----
$("#toggleDetailsAll").addEventListener("click", ()=>{
  detailsOpenAll = !detailsOpenAll;
  $("#toggleDetailsAll").textContent = detailsOpenAll ? "Hide details" : "Details";
  $$("#results .details").forEach(d=>d.classList.toggle("open", detailsOpenAll));
});

function renderResults(){
  const wrap = $("#results");
  wrap.innerHTML = "";

  const isEmpty = !state.results || state.results.length === 0;
  const panel = wrap.closest(".panel");
  if (panel) panel.classList.toggle("is-empty", isEmpty);

  const saveBtn = el.saveSessionBtn || document.getElementById("saveSessionBtn");
  if (saveBtn) {
    if (isEmpty) {
      saveBtn.style.display = "none";
      saveBtn.disabled = true;
    } else {
      saveBtn.style.display = "inline-flex";
      saveBtn.disabled = false;
    }
  }

  if (isEmpty) {
    wrap.innerHTML = `<div class="empty-hint">Add recordings and press <b>Analyse</b> to see results.</div>`;
    return;
  }

  el.results.innerHTML = "";
  if (!state.results.length) {
    return;
  }

  const SCALE_MAX = 3.0; // seconds cap

  state.results.forEach(r=>{
    const q = normaliseQuality(r);
    const card=document.createElement("div"); card.className="card"; const id=slug(r.name); card.id=`res-${id}`;

    const b=document.createElement("div"); b.className="badge "+(q==="OK"?"b-ok":q==="CHECK"?"b-check":"b-low"); b.textContent=q||"—"; b.title = qualityTitle({name:r.name, quality:q}); card.appendChild(b);

    const header=document.createElement("h4"); header.textContent = r.display || r.name; card.appendChild(header);

      // If overall quality is LOW, hide detailed metrics and just show a retest message
    if (q === "LOW") {
      const msg = document.createElement("p");
      msg.className = "low-note";
      msg.textContent = "Results are voided for this take because it did not meet quality thresholds. Please attempt recording again in adherence to the recording guide.";
      card.appendChild(msg);
      el.results.appendChild(card);
      return;
    }
    const staples = document.createElement("div"); staples.className="staples";
    const tile = (short, full, val) => {
      const d = document.createElement("div"); d.className="s"; d.title = `${full}: ${safe(val)}`;
      d.innerHTML = `<div class="k">${short}</div><div class="v">${safe(val)}</div>`;
      return d;
    };
    const rt60Measured = r.rt60_measured;
    const rt60Fused = r.rt60_fused ?? r.rt60_measured;
    const isCheck = q === "CHECK";

    const rtLabelShort = isCheck ? "Fused RT60 (s)" : "RT60 (s)";
    const rtLabelLong = isCheck
      ? "Fused reverberation time 60 (s)"
      : "Reverberation time 60 (s)";

    staples.appendChild(
      tile(rtLabelShort, rtLabelLong, rt60Fused)
    );
    staples.appendChild(tile("EDT (s)","Early Decay Time (s)", r.edt_s));
    staples.appendChild(tile("C50 (dB)","Clarity 50 (dB)", r.c50_db));
    staples.appendChild(tile("r²","Regression confidence (r²)", r.qc?.r2));
    card.appendChild(staples);

    // LMH visual (more headroom + labels on top and numeric values below)
    const L = num(r.rt60_low_med), M = num(r.rt60_mid_med), H = num(r.rt60_high_med);
    const color = qColor(q);
    const svg = document.createElementNS("http://www.w3.org/2000/svg","svg");
    svg.setAttribute("viewBox","0 0 100 100"); svg.classList.add("lmh-svg");

    const pos = [20, 50, 80]; const barWidth = 24;
    const addBar = (cx,val,topLabel,bottomVal)=>{
      const cap = val!=null && val > SCALE_MAX;
      const vv = val==null ? 0.02 : Math.min(val,SCALE_MAX);
      const height = Math.max(2, (vv/SCALE_MAX)*68);
      const y = 80 - height;

      if(cap){
        const o = document.createElementNS("http://www.w3.org/2000/svg","rect");
        o.setAttribute("x",cx-barWidth/2); o.setAttribute("y",y);
        o.setAttribute("width",barWidth); o.setAttribute("height",height);
        o.setAttribute("rx","2"); o.setAttribute("fill","none");
        o.setAttribute("stroke","#e05a61"); o.setAttribute("stroke-width","3.5");
        svg.appendChild(o);
      }
      const rect = document.createElementNS("http://www.w3.org/2000/svg","rect");
      rect.setAttribute("x",cx-barWidth/2); rect.setAttribute("y",y);
      rect.setAttribute("width",barWidth); rect.setAttribute("height",height);
      rect.setAttribute("rx","2"); rect.setAttribute("stroke","var(--line)");
      rect.setAttribute("fill", cap ? "#213b50" : "#163247");
      svg.appendChild(rect);

      const label = document.createElementNS("http://www.w3.org/2000/svg","text");
      label.setAttribute("x", cx);
      label.setAttribute("y", 7);
      label.setAttribute("text-anchor", "middle");
      label.setAttribute("fill", "var(--muted)");
      label.setAttribute("font-size", "9");
      label.setAttribute("dominant-baseline", "alphabetic");
      label.style.pointerEvents = "none";
      label.textContent = topLabel;
      svg.appendChild(label);

      const valtxt = document.createElementNS("http://www.w3.org/2000/svg","text");
      valtxt.setAttribute("x",cx); valtxt.setAttribute("y",95);
      valtxt.setAttribute("text-anchor","middle");
      valtxt.setAttribute("fill","var(--ink-2)");
      valtxt.setAttribute("font-size","10");
      valtxt.textContent = (bottomVal==null) ? "—" : bottomVal.toFixed(2);
      svg.appendChild(valtxt);
      return {x:cx, y};
    };
    const pL = addBar(pos[0],L,"Low",L);
    const pM = addBar(pos[1],M,"Mid",M);
    const pH = addBar(pos[2],H,"High",H);

    const path = document.createElementNS("http://www.w3.org/2000/svg","path");
    path.setAttribute("d",`M ${pL.x} ${pL.y} L ${pM.x} ${pM.y} L ${pH.x} ${pH.y}`);
    path.setAttribute("stroke", color); path.setAttribute("stroke-width","2.5"); path.setAttribute("fill","none");
    svg.appendChild(path);
    [pL,pM,pH].forEach(pt=>{
      const c=document.createElementNS("http://www.w3.org/2000/svg","circle");
      c.setAttribute("cx",pt.x); c.setAttribute("cy",pt.y); c.setAttribute("r","2.8"); c.setAttribute("fill","#fff"); c.setAttribute("stroke","var(--line)");
      svg.appendChild(c);
    });

    const lmhBox = document.createElement("div"); lmhBox.className="lmh";
    lmhBox.appendChild(svg);
    const units = document.createElement("div"); units.className="lmh-caption";
    units.textContent = "Low / Mid / High RT60 band medians (s)";
    lmhBox.appendChild(units);
    card.appendChild(lmhBox);

    const details = document.createElement("div");
    details.className = "details";
    if (detailsOpenAll) details.classList.add("open");

    const detailGrid = document.createElement("div");
    detailGrid.className = "detail-grid";
    detailGrid.innerHTML = `
      <div class="s" title="Clarity 80 (dB): ${safe(r.c80_db)}"><div class="k">C80 (dB)</div><div class="v">${safe(r.c80_db)}</div></div>
      <div class="s" title="Noise floor L90 (dB): ${safe(r.spl_l90_db)}"><div class="k">L90 (dB)</div><div class="v">${safe(r.spl_l90_db)}</div></div>
      <div class="s" title="Signal-to-noise ratio (dB): ${safe(r.qc?.snr_db)}"><div class="k">SNR (dB)</div><div class="v">${safe(r.qc?.snr_db)}</div></div>
      <div class="s" title="Was measurement capped in decay fit?"><div class="k">Capped</div><div class="v">${r.qc?.capped ? "true" : "false"}</div></div>
    `;
    details.appendChild(detailGrid);

    // NEW: per-band RT breakdown (only when per-band preset is on and data exists)
    if (r.per_band && Array.isArray(r.rt60_bands) && r.rt60_bands.length) {
      const bandBox = document.createElement("div");
      bandBox.className = "band-table";

      const title = document.createElement("div");
      title.className = "band-table-title";
      title.textContent = "Per-band RT60 (s)";
      bandBox.appendChild(title);

      const header = document.createElement("div");
      header.className = "band-header";
      header.innerHTML = `<div>Band</div><div>RT60 (s)</div><div>EDT (s)</div><div>C50 (dB)</div><div>r²</div>`;
      bandBox.appendChild(header);

      const maxRows = 12;
      r.rt60_bands.slice(0, maxRows).forEach((b) => {
        const row = document.createElement("div");
        row.className = "band-row";

        const band = document.createElement("div");
        band.textContent = b.tag || `${safe(b.lo_hz, 0)}-${safe(b.hi_hz, 0)} Hz`;

        const rt = document.createElement("div");
        rt.className = "rt";
        rt.textContent = safe(b.rt60_s);

        const edt = document.createElement("div");
        edt.className = "rt";
        edt.textContent = safe(b.edt_s);

        const c50 = document.createElement("div");
        c50.className = "rt";
        c50.textContent = safe(b.c50_db);

        const r2 = document.createElement("div");
        r2.className = "rt";
        r2.textContent = safe(b.r2);

        row.appendChild(band);
        row.appendChild(rt);
        row.appendChild(edt);
        row.appendChild(c50);
        row.appendChild(r2);

        bandBox.appendChild(row);
      });

      details.appendChild(bandBox);
    }

    // Build the list of feedback lines
    const tipItems = buildTips(r, q);
    const ul = document.createElement("ul");
    ul.className = "tips";
    for (const t of tipItems) {
      const li = document.createElement("li");
      li.textContent = t;
      ul.appendChild(li);
    }

    details.appendChild(ul);
    card.appendChild(details);

    // --- View graphs button ---
    const vg = document.createElement("button");
    vg.className = "pill";
    vg.style.cssText = "padding:8px 16px;font-size:11px;margin-top:8px";
    vg.textContent = "View graphs";

    vg.addEventListener("click", () => {
      // open graphs panel
      graphsOpen = true;
      el.graphsBody.classList.add("open");
      el.graphsToggle.textContent = "▾";
      renderGraphs();

      // try to jump to this file's block
      const slugId = slug(r.name || r.display || "");
      const target = document.getElementById(`graphs-${slugId}`);
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
        target.classList.add("pulse");
        setTimeout(() => target.classList.remove("pulse"), 800);
      } else {
        const panel = document.querySelector(".graphs");
        if (panel) panel.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    });

    const tgl = document.getElementById('toggleDetailsAll');
    if (tgl) tgl.textContent = detailsOpenAll ? "Hide details" : "Details";

    card.appendChild(vg);

    el.results.appendChild(card);

    const slugId = slug(r.name || r.display || "");
    card.id = `result-${slugId}`;
  });
}

function num(v){ const n=Number(v); return Number.isFinite(n)?n:null; }

function formatSessionTimestamp(date) {
  try {
    return date.toLocaleString(undefined, {
      year: "numeric",
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
    });
  } catch {
    return date.toISOString();
  }
}

function buildTips(r, quality) {
  if (!r) return [];

  const df = r.direct_feedback || {};
  const tips = [];
  const q = (quality || r.quality_label || r.quality_pred || "").toUpperCase();
  const rtOrig  = typeof r.rt60_measured === "number" ? r.rt60_measured : null;
  const rtFused = typeof r.rt60_fused === "number" ? r.rt60_fused : null;

  if (df.headline) {
    tips.push(df.headline);
  }

  if (Array.isArray(df.bullets)) {
    tips.push(...df.bullets);
  }

  // if (Array.isArray(df.actions) && df.actions.length) {
  //   tips.push("Actions: " + df.actions.join("; "));
  // }

  if (
    q === "CHECK" &&
    rtOrig !== null &&
    rtFused !== null &&
    Math.abs(rtFused - rtOrig) > 1e-3 &&
    tips.length
  ) {
    let idx = tips.findIndex(
      t =>
        typeof t === "string" &&
        t.toLowerCase().includes("lower confidence reverberation")
    );
    if (idx === -1) {
      idx = 0;
    }

    const prefix =
      `Original RT60 estimate was ${rtOrig.toFixed(2)} s; ` +
      `a fused RT60 of ${rtFused.toFixed(2)} s was used to stabilise the result. `;

    tips[idx] = prefix + tips[idx];
  }

  return tips.filter(Boolean);
}

// ---- Feedback ----
let fbOpen = true;
el.fbToggle.addEventListener("click", () => {
  fbOpen = !fbOpen;
  el.fbBody.style.display = fbOpen ? "block" : "none";
  el.fbToggle.textContent = fbOpen ? "▾" : "▴";
});

function renderFeedback(){
  try { ensureFbSummary(); } catch(_) {}
  if (!el.fbSummary) el.fbSummary = document.getElementById('fbSummary');
  if (!el.fbBullets) el.fbBullets = document.getElementById('fbBullets');
  if (!el.fbSummary || !el.fbBullets) return;

  // show nothing until Analyse has run at least once
  if (!state.results.length) {
    if (el.fbSummary) el.fbSummary.textContent = "";
    if (el.fbBullets) el.fbBullets.innerHTML = "";
    return;
  }

  // counts line (hardcoded format)
  const counts = {OK:0, CHECK:0, LOW:0};
  state.results.forEach(r=>{
    const q = normaliseQuality(r);
    if(counts[q]!==undefined) counts[q]++;
  });
  if (el.fbSummary)
    el.fbSummary.textContent = `This room: ${counts.OK} OK, ${counts.CHECK} CHECK, ${counts.LOW} LOW.`;

  // AI bullets
  const rs = state.room_summary || null;

  // Helper: turn any "Actions: ..." text into clean sentences
  function splitActionsText(text) {
    if (!text || typeof text !== "string") return [];
    let t = text.replace(/^actions:\s*/i, "").trim();
    if (!t) return [];

    // Normalise the weird "feel., Given..." style joins into ". "
    t = t.replace(/\.?\s*,\s+(?=[A-Z])/g, ". ");

    // Split on sentence boundaries (simple but works fine here)
    const parts = t.split(". ").map(s => s.trim()).filter(Boolean);

    return parts.map((s, idx) => {
      // Make sure each part ends with a full stop
      if (!s.endsWith(".")) {
        return s + ".";
      }
      return s;
    });
  }

  if (el.fbBullets) {
    if (rs && Array.isArray(rs.bullets) && rs.bullets.length) {

      const baseBullets = [];
      let actionsList = [];

      // 1) Scan bullets - pull out any that start with "Actions:"
      rs.bullets.forEach(b => {
        if (typeof b === "string" && /^actions:/i.test(b.trim())) {
          actionsList = actionsList.concat(splitActionsText(b));
        } else {
          baseBullets.push(b);
        }
      });

      // 2) Also fold in anything from rs.actions (if the model used it)
      if (Array.isArray(rs.actions)) {
        rs.actions.forEach(a => {
          actionsList = actionsList.concat(splitActionsText(a));
        });
      }

      // 3) Render normal bullets
      let html = baseBullets
        .map(b => `<li>${b}</li>`)
        .join("");

      // 4) Render Actions block if we found any
      if (actionsList.length) {
        const actionsHtml = actionsList
          .map(a => `<li class="fb-action-item">${a}</li>`)
          .join("");

        html += `
          <li class="fb-actions-heading">Actions</li>
          ${actionsHtml}
        `;
      }

      el.fbBullets.innerHTML = html;
    } else if (state.room_summary && state.room_summary.error) {
      const msg = state.room_summary.error_msg || "AI summary unavailable";
      el.fbBullets.innerHTML = `<li style="color:var(--ink-3)">${msg}</li>`;
    } else {
      el.fbBullets.innerHTML = "";
    }
  }
}

// ---- Graphs ----
let graphsOpen = true;

el.graphsToggle.addEventListener("click", () => {
  graphsOpen = !graphsOpen;
  el.graphsBody.classList.toggle("open", graphsOpen);
  el.graphsToggle.textContent = graphsOpen ? "▾" : "▴";
});

function renderGraphs() {
  el.graphsHost.innerHTML = "";

  const results = state.results.length ? state.results : state.quick;
  if (!results.length) {
    return;
  }

  results.forEach(r => {
    const graphs = r.graphs || GRAPH_PLACEHOLDERS[r.name] || {};
    if (!graphs.regression && !graphs.spectrogram) {
      return; // nothing to show for this file
    }

    const slugId = slug(r.name || r.display || "");
    const fileLabel = r.display || r.name || "Untitled take";

    const block = document.createElement("div");
    block.className = "gblock";
    block.id = `graphs-${slugId}`;

    block.innerHTML = `
      <div class="gfile">
        <button class="gfile-link" data-target="result-${slugId}">
          ${fileLabel}
        </button>
        <button class="g-info">i</button>
      </div>
      <div class="gpair">
        ${graphs.regression ? `
          <div class="gimg-wrap">
            <div class="gimg-title">Regression fit</div>
              <div class="gimg-inner">
              <img alt="Regression fit graph for ${fileLabel}" src="${graphs.regression}"/>
            </div>
          </div>` : ""}
        ${graphs.spectrogram ? `
          <div class="gimg-wrap">
            <div class="gimg-title">Spectrogram</div>
              <div class="gimg-inner">
              <img alt="Spectrogram for ${fileLabel}" src="${graphs.spectrogram}"/>
            </div>
          </div>` : ""}
      </div>
    `;

    el.graphsHost.appendChild(block);
  });

  // Link from graph header → result card
  $$(".gfile-link").forEach(btn => {
    btn.onclick = () => {
      const id = btn.dataset.target;
      const card = document.getElementById(id);
      if (card) {
        card.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    };
  });

  // Info icon → open Info modal
  $$(".g-info").forEach(btn => {
    btn.onclick = () => {
      const infoPill = document.querySelector('.pill[data-modal="infoModal"]');
      if (infoPill) infoPill.click();
    };
  });
}

function renderHistory() {
  const host = el.historyBody || document.getElementById("historyBody");
  if (!host) return;

  host.innerHTML = "";
  const items = state.history || [];
  if (!items.length) {
    const p = document.createElement("p");
    p.className = "empty";
    p.textContent = "No saved sessions yet. Run an analysis and click “Save session”.";
    host.appendChild(p);
    return;
  }

  items.forEach(sess => {
    const box = document.createElement("div");
    box.className = "panel section";
    box.style.marginBottom = "12px";
    box.style.padding = "12px 14px";

    // Header row: title + timestamp
    const titleRow = document.createElement("div");
    titleRow.style.display = "flex";
    titleRow.style.justifyContent = "space-between";
    titleRow.style.alignItems = "baseline";
    titleRow.style.marginBottom = "4px";

    const nameSpan = document.createElement("div");
    nameSpan.style.fontWeight = "700";
    nameSpan.textContent = sess.title;

    const timeSpan = document.createElement("div");
    timeSpan.style.fontSize = "11px";
    timeSpan.style.color = "var(--muted)";
    timeSpan.textContent = formatSessionTimestamp(new Date(sess.savedAt));

    titleRow.appendChild(nameSpan);
    titleRow.appendChild(timeSpan);
    box.appendChild(titleRow);

    // Context line (use • goal)
    if (sess.context && (sess.context.use || sess.context.goal)) {
      const ctx = document.createElement("div");
      ctx.style.fontSize = "12px";
      ctx.style.color = "var(--ink-2)";
      const bits = [];
      if (sess.context.use) bits.push(sess.context.use);
      if (sess.context.goal) bits.push(sess.context.goal);
      ctx.textContent = bits.join(" • ");
      ctx.style.marginBottom = "8px";
      box.appendChild(ctx);
    }

    // Results snapshot
    if (sess.resultsHtml) {
      const resWrap = document.createElement("div");
      resWrap.className = "cards";
      resWrap.innerHTML = sess.resultsHtml;
      box.appendChild(resWrap);
    }

    // Feedback snapshot (AI summary + bullets)
    if (sess.feedbackHtml) {
      const fbWrap = document.createElement("div");
      fbWrap.style.marginTop = "10px";
      fbWrap.innerHTML = sess.feedbackHtml;
      box.appendChild(fbWrap);
    }

    host.appendChild(box);
  });
}

// ---- Boot ----
checkAPI(); renderRecs(); renderResults(); renderFeedback(); renderGraphs(); renderHistory();
