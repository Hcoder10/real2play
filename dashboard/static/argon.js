"use strict";
(() => {
  const strip = document.createElement("section");
  strip.id = "argon-strip";
  strip.setAttribute("aria-label", "Argon orchestrator");
  strip.setAttribute("aria-live", "polite");
  const title = document.createElement("strong");
  title.textContent = "ARGON";
  const status = document.createElement("span");
  status.className = "badge";
  const detail = document.createElement("div");
  const stage = document.createElement("div");
  const why = document.createElement("p");
  const metric = document.createElement("small");
  const managed = document.createElement("small");
  detail.append(stage, why, metric, managed);
  strip.append(title, status, detail);
  document.querySelector("header").after(strip);
  async function poll() {
    try {
      const response = await fetch("/api/argon", {cache: "no-store"});
      if (!response.ok) throw Error("Source unavailable");
      const data = await response.json();
      status.textContent = data.available ? `${data.synthetic ? "SIM · " : ""}${data.status}` : "WAITING";
      status.classList.toggle("warning", !!data.synthetic || data.status === "BLOCKED");
      stage.textContent = data.available ? `${data.stage} · ${data.last_tool || "planning"}` : "No orchestrator run yet";
      why.textContent = data.error || data.why || "Start an Argon run to see decisions and measured results.";
      const latest = [...data.ledger].reverse().find(row => row.kind === "result");
      metric.textContent = Object.entries(latest?.result?.metrics || {}).map(([key, value]) =>
        `${key}: ${Number.isInteger(value) ? value : Number(value).toFixed(3)}`).join(" · ");
      managed.textContent = (data.managed || []).map(row =>
        `${row.synthetic ? "Fake managed" : "Managed"} ${row.tool}: ${row.agent_id} · ${Number(row.duration_s).toFixed(1)}s · ${row.tests_passed} checks passed`).join(" | ");
      strip.title = `Next: ${JSON.stringify(data.next ?? "none")}`;
    } catch (error) {
      status.textContent = "STALE";
      why.textContent = "Orchestrator source unavailable; retrying.";
    } finally {
      setTimeout(poll, 2000);
    }
  }
  poll();
})();
