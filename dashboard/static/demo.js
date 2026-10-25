"use strict";
(() => {
  const $ = id => document.getElementById(id);
  const text = (id, value) => { $(id).textContent = value ?? "—"; };
  const el = (tag, cls, value) => { const node = document.createElement(tag); if (cls) node.className = cls; if (value !== undefined) node.textContent = value; return node; };
  const num = value => Number.isFinite(value) ? value.toLocaleString(undefined, {maximumFractionDigits: 2}) : "—";
  const percent = value => Number.isFinite(value) ? `${num(value * 100)}%` : "—";
  const age = timestamp => { if (!Number.isFinite(timestamp)) return "Time not recorded"; const seconds = Math.max(0, Date.now() / 1000 - timestamp); return seconds < 60 ? `${Math.floor(seconds)}s ago` : seconds < 3600 ? `${Math.floor(seconds / 60)}m ago` : `${Math.floor(seconds / 3600)}h ago`; };
  const NS = "http://www.w3.org/2000/svg";
  let latest = null;
  let slides = [], slideIndex = 0, slideRun = null, slidePlaying = true;
  function image(id, src) {
    const target = $(id);
    if (!src) { target.hidden = true; target.removeAttribute("src"); return; }
    if (target.getAttribute("src") !== src) { target.hidden = true; target.src = src; }
  }
  for (const id of ["lab-scene", "lab-preview", "lab-wrist", "lab-comparison"]) {
    $(id).onload = () => { $(id).hidden = false; };
    $(id).onerror = () => { $(id).hidden = true; };
  }
  function showSlide() {
    const slide = slides[slideIndex];
    image("lab-comparison", slide?.comparison);
    $("slide-turn").max = Math.max(0, slides.length - 1);
    $("slide-turn").value = slideIndex;
    text("slide-label", slide ? `Turn ${slide.turn} / ${slides.at(-1).turn}` : "No frames");
    text("slide-play", slidePlaying ? "Pause" : "Play");
    $("slide-play").setAttribute("aria-pressed", String(slidePlaying));
    for (const id of ["slide-turn", "slide-play", "slide-latest"]) $(id).disabled = !slides.length;
    $("step-comparison").dataset.turn = slide?.turn ?? "";
    text("replay-caption", `Real scene (left) | twin (right) · ${slidePlaying ? "1 s per turn · loops" : "paused"}`);
  }
  $("slide-play").onclick = () => { slidePlaying = !slidePlaying; showSlide(); };
  $("slide-latest").onclick = () => { slidePlaying = false; slideIndex = Math.max(0, slides.length - 1); showSlide(); };
  $("slide-turn").oninput = event => { slidePlaying = false; slideIndex = Number(event.target.value); showSlide(); };
  setInterval(() => {
    if (latest?.lab.kind === "argon_step" && slidePlaying && slides.length && !document.hidden) {
      slideIndex = (slideIndex + 1) % slides.length; showSlide();
    }
  }, 1000);
  function renderStepLab(lab) {
    const oldTurn = slides[slideIndex]?.turn;
    const changedRun = slideRun !== lab.source;
    slideRun = lab.source; slides = lab.slides || [];
    if (changedRun) { slideIndex = Math.max(0, slides.length - 1); slidePlaying = true; }
    else { const found = slides.findIndex(s => s.turn === oldTurn); slideIndex = found < 0 ? Math.max(0, slides.length - 1) : found; }
    const video = $("replay");
    if (video.getAttribute("src")) { video.pause(); video.removeAttribute("src"); video.load(); }
    video.hidden = true; $("comparison-stills").hidden = true;
    text("reference-label", "Real scene camera");
    image("lab-wrist", lab.frames?.wrist);
    text("lab-wrist-label", `Latest wrist · turn ${slides.at(-1)?.turn ?? "—"}`);
    showSlide();
    text("metric-title", "Argon's sim"); text("metric-count", `${lab.sim.source} · recorded estimates`);
    text("sim-objects", num(lab.sim.objects));
    text("sim-z", Number.isFinite(lab.sim.z_mm) ? `${num(lab.sim.z_mm)} mm` : "—");
    text("sim-xy", lab.sim.xy_mm ? `[${lab.sim.xy_mm.map(num).join(", ")}] mm` : "—");
    text("tools-title", "Recent actions"); text("tool-count", "Latest first · ledger replies");
    $("tools").replaceChildren();
    for (const action of [...lab.actions].reverse()) {
      const item = el("li"); item.append(el("span", "", `T${action.turn}  `), document.createTextNode(action.label)); $("tools").append(item);
    }
    if (!lab.actions.length) $("tools").append(el("li", "muted", "Awaiting recorded actions"));
    text("edits-title", "Arm notes"); text("edit-count", "sim.json");
    text("sim-notes", lab.sim.notes); $("sim-notes").title = lab.sim.notes;
  }
  function spark(points, key, label) {
    const svg = document.createElementNS(NS, "svg"); svg.setAttribute("viewBox", "0 0 160 25"); svg.setAttribute("role", "img"); svg.setAttribute("aria-label", label);
    const values = points.map(p => p[key]).filter(Number.isFinite);
    if (!values.length) return svg;
    const lo = Math.min(...values), hi = Math.max(...values), x = i => 4 + i / Math.max(1, points.length - 1) * 152;
    const y = v => 21 - (v - lo) / (hi - lo || 1) * 17;
    let path = "", drawing = false;
    points.forEach((p, i) => { if (!Number.isFinite(p[key])) { drawing = false; return; } path += `${drawing ? "L" : "M"}${x(i)},${y(p[key])} `; drawing = true; });
    const line = document.createElementNS(NS, "path"); line.setAttribute("d", path); line.setAttribute("fill", "none"); line.setAttribute("stroke", "#a8c7ff"); line.setAttribute("stroke-width", "2"); svg.append(line);
    points.forEach((p, i) => { if (!Number.isFinite(p[key])) return; const dot = document.createElementNS(NS, "circle"); dot.setAttribute("cx", x(i)); dot.setAttribute("cy", y(p[key])); dot.setAttribute("r", "2"); dot.setAttribute("fill", "#a8c7ff"); svg.append(dot); });
    return svg;
  }
  function renderLab(lab) {
    const isStep = lab.kind === "argon_step";
    document.querySelector(".lab").classList.toggle("step-lab", isStep);
    for (const id of ["step-comparison", "slideshow-controls", "sim-summary", "sim-notes"]) $(id).hidden = !isStep;
    for (const id of ["residuals", "edits"]) $(id).hidden = isStep;
    text("lab-label", lab.label || "No lab run"); $("lab-label").className = `tag ${lab.synthetic ? "sim" : ""}`;
    text("lab-run", lab.available ? `${lab.run} · ${lab.model || "Model not recorded"}` : "Waiting for recorded lab artifacts");
    text("lab-status", lab.available ? `Turn ${lab.turn ?? "—"} · ${lab.status}` : "");
    text("thought", lab.thought?.replace(/^#\s*Thought:\s*/, "") || lab.why || "Start a lab experiment to see Argon's observations and next step.");
    $("thought").title = lab.thought || lab.why || "";
    if (isStep) { renderStepLab(lab); return; }
    text("metric-title", "Measured residuals"); text("tools-title", "Recent tool calls"); text("edits-title", "Sim edits");
    text("reference-label", lab.synthetic ? "Simulated reference" : lab.label === "Real bench" ? "Real observation" : "Reference · unverified source");
    image("lab-scene", lab.frames?.scene); image("lab-preview", lab.preview); image("lab-wrist", lab.frames?.wrist);
    text("lab-wrist-label", lab.synthetic ? "Wrist · simulated reference" : "Wrist · recorded observation");
    const video = $("replay");
    if (lab.video) {
      if (video.getAttribute("src") !== lab.video) { video.src = lab.video; video.load(); video.play().catch(() => {}); }
      video.hidden = false; $("comparison-stills").hidden = true;
      text("replay-caption", lab.video_kind === "comparison" ? `${lab.synthetic ? "Simulated reference" : "Recorded reference"} | sim replay · loops` : "Simulated replay · loops");
    } else {
      if (video.getAttribute("src")) { video.pause(); video.removeAttribute("src"); video.load(); }
      video.hidden = true; $("comparison-stills").hidden = false;
      text("replay-caption", lab.video_message || "Latest camera + sim render · awaiting a comparison replay");
    }
    const history = lab.history || [];
    text("metric-count", history.length ? `${history.length} measurements · by turn` : "Awaiting replay measurements");
    $("residuals").replaceChildren();
    for (const [key, label, unit] of [["joint_deg", "Joint RMS ↓", "°"], ["jaw_mm", "Jaw RMS ↓", "mm"], ["object_mm", "Object error ↓", "mm"], ["image_similarity", "Image match ↑", ""]]) {
      const item = el("div", "residual"); const value = history.at(-1)?.[key];
      const strong = el("strong", "", num(value)); if (unit) strong.append(el("small", "", ` ${unit}`));
      item.append(el("span", "", label), strong, spark(history, key, `${label} over recorded turns; gaps mean unavailable`)); $("residuals").append(item);
    }
    $("tools").replaceChildren(); text("tool-count", lab.calls?.length ? "Latest first" : "");
    for (const call of [...(lab.calls || [])].reverse().slice(0, 4)) {
      const li = el("li"); li.title = call.detail;
      li.append(el("span", "", `T${call.turn ?? "?"}  `), el("span", call.ok ? "ok" : "error", call.ok ? "✓ " : "! "), document.createTextNode(call.tool), el("span", "", call.ok ? "" : ` · ${call.detail}`)); $("tools").append(li);
    }
    if (!lab.calls?.length) $("tools").append(el("li", "muted", "No completed tool calls yet"));
    $("edits").replaceChildren(); text("edit-count", `${lab.edits?.length || 0} files`);
    for (const edit of (lab.edits || []).slice(-4)) {
      const li = el("li"); li.append(el("strong", "", edit.path), el("span", "", edit.diff_known ? `+${edit.added} −${edit.removed}` : `${edit.edits} edits · diff unavailable`)); li.title = `${edit.path}: ${edit.edits} accepted edits`; $("edits").append(li);
    }
    if (!lab.edits?.length) $("edits").append(el("li", "muted", "No accepted sim edits recorded"));
  }
  $("replay").onerror = () => { $("replay").hidden = true; $("comparison-stills").hidden = false; text("replay-caption", "Replay unavailable in this browser · showing latest stills"); };
  function render(data) {
    latest = data; document.body.dataset.ready = "true"; document.body.dataset.mock = "false";
    $("chain").replaceChildren();
    for (const stage of data.chain.stages) {
      const li = el("li", stage.state); const head = el("div", "chain-head"); head.append(el("strong", "", stage.name), el("span", "", stage.state));
      li.append(head, el("span", "chain-metric", stage.metric), el("span", "chain-provenance", stage.provenance)); li.title = stage.detail || stage.metric; $("chain").append(li);
    }
    text("chain-source", `${data.chain.synthetic ? "Synthetic rehearsal · " : ""}${data.chain.source}${data.chain.updated ? ` · ${age(data.chain.updated)}` : ""}`);
    $("chain-source").classList.toggle("synthetic", !!data.chain.synthetic);
    renderLab(data.lab);
    const game = data.game; $("game-link").href = /^https:\/\//.test(game.url) ? game.url : "#";
    $("trace-counts").replaceChildren();
    for (const [key, label] of [["human", "Human rounds"], ["headless", "Headless sim"], ["bot", "Bot rounds"]]) { const item = el("div"); item.append(el("strong", "", num(game.groups[key])), el("span", "", label)); $("trace-counts").append(item); }
    text("receiver", `${game.receiver_online ? `Receiver online · ${num(game.receiver_traces)} received` : "Receiver offline · showing saved traces"}${game.groups.expert ? ` · ${game.groups.expert} experts` : ""}${game.groups.unknown ? ` · ${game.groups.unknown} unclassified` : ""}${game.invalid ? ` · ${game.invalid} unreadable` : ""}`);
    const loop = game.after_step?.loop;
    if (loop && Object.keys(loop).length) {
      $("receiver").append(document.createTextNode(` · Loop ${loop.status || "pending"} (${loop.stage || "unknown stage"})${loop.error ? `: ${loop.error}` : ""}`));
    }
    const policy = data.policy; text("policy-label", policy.evidence || "No evaluation");
    text("clean", percent(policy.clean)); text("half", percent(policy.half)); text("noise", percent(policy.real));
    for (const key of ["clean", "half", "real"]) text(`${key}-n`, policy.counts?.[key] ? `n=${policy.counts[key]} scenes` : "n not recorded");
    text("policy-note", `${policy.name || "No export selected"} · ${policy.message || "Waiting for evaluation"}`); $("policy-note").title = $("policy-note").textContent;
    const deploy = data.deploy;
    text("deploy-label", deploy.synthetic ? "Simulated board" : deploy.available ? "Recorded board" : "No board yet"); $("deploy-label").className = `tag ${deploy.synthetic ? "warning" : ""}`;
    $("board").replaceChildren();
    deploy.board.forEach((mark, i) => { const move = deploy.move === i || deploy.move === `r${Math.floor(i / 3)}c${i % 3}`; const cell = el("div", `cell ${mark === "O" ? "o" : ""}${move ? " move" : ""}`, mark); cell.setAttribute("aria-label", `Row ${Math.floor(i / 3) + 1}, column ${i % 3 + 1}: ${mark || "empty"}`); $("board").append(cell); });
    text("move", deploy.move !== null && deploy.move !== undefined ? `${deploy.model?.includes("argon") ? "Argon's" : "Recorded"} move: ${deploy.move}` : "Waiting for a move");
    text("reason", deploy.why || "No explanation recorded yet."); $("reason").title = deploy.why || "";
    text("decision-meta", `${deploy.model || "No model response"} · ${Number.isFinite(deploy.latency_s) ? `${num(deploy.latency_s)}s decision` : "Latency not recorded"} · ${age(deploy.updated)}`);
    text("board-caption", deploy.board_valid ? "Robot's perspective" : "No valid board reading");
    text("footer-note", data.lab.synthetic ? "Lab reference is simulated · camera feeds show the physical bench separately" : "Read-only evidence · real, simulated and headless sources stay labeled");
    text("connection", "Evidence connected"); $("connection").classList.remove("stale"); tick();
  }
  function tick() { if (latest) text("lab-age", `Lab updated ${age(latest.lab.updated)}`); }
  async function poll() { try { const response = await fetch("/api/demo", {cache: "no-store", signal: AbortSignal.timeout(12000)}); if (!response.ok) throw Error(`HTTP ${response.status}`); render(await response.json()); } catch (error) { text("connection", `Evidence unavailable · last values may be stale (${error.message})`); $("connection").classList.add("stale"); } finally { setTimeout(poll, 2000); } }
  function camera(name) {
    const img = $(name); let busy = false;
    const label = name === "phone" ? "Phone" : "Wrist";
    img.onload = () => { busy = false; img.hidden = false; text(`${name}-label`, `${label} · live camera`); };
    img.onerror = () => { busy = false; img.hidden = true; text(`${name}-label`, `${label} · offline`); };
    function refresh() { if (!busy && !document.hidden) { busy = true; img.src = `/demo/camera/${name}.jpg?t=${Date.now()}`; } }
    refresh(); setInterval(refresh, 1500);
  }
  function focus(panel) { if (!panel || document.body.dataset.focus === panel) delete document.body.dataset.focus; else document.body.dataset.focus = panel; for (const button of document.querySelectorAll("[data-focus]")) button.setAttribute("aria-pressed", String(document.body.dataset.focus === button.dataset.focus)); }
  for (const button of document.querySelectorAll("button[data-focus]")) button.onclick = () => focus(button.dataset.focus);
  async function fullscreen() { try { if (document.fullscreenElement) await document.exitFullscreen(); else await document.documentElement.requestFullscreen(); } catch { text("connection", "Use the browser's full-screen control"); } }
  $("fullscreen").onclick = fullscreen;
  document.addEventListener("keydown", event => { if (event.altKey || event.ctrlKey || event.metaKey || /INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return; if (/^[1-4]$/.test(event.key)) focus(event.key); else if (event.key === "Escape" || event.key === "0") focus(null); else if (event.key.toLowerCase() === "f") fullscreen(); });
  setInterval(tick, 1000); camera("phone"); camera("wrist"); poll();
})();
