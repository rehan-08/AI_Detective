"use strict";
/* Single-page front end. Case documents are untrusted, so every piece of data reaches
   the page through textContent or setAttribute, never through innerHTML. */

const app = document.getElementById("app");
const TABS = [
  ["summary", "Summary"],
  ["evidence", "Evidence"],
  ["timeline", "Timeline"],
  ["suspects", "Suspects"],
  ["contradictions", "Contradictions"],
  ["reasoning", "Reasoning"],
  ["ask", "Ask"],
  ["report", "Report"],
  ["key", "Answer key"],
];
const DOC_TYPES = [
  ["witness_interview", "Witness interview"],
  ["incident_report", "Incident report"],
  ["email", "Email"],
  ["chat", "Chat"],
  ["document", "Other document"],
  ["system_log", "System log"],
  ["cctv", "CCTV summary"],
  ["financial_record", "Financial record"],
  ["phone_record", "Phone record"],
  ["anonymous_tip", "Anonymous tip"],
];
const LABELS = {
  opportunity_direct: "At the scene",
  opportunity_site: "On site",
  excluded: "Excluded",
  alibi_broken: "Alibi broken",
  motive: "Motive",
  means: "Means",
  suspicious: "Suspicious conduct",
  linked_to_artifact: "Linked to artefact",
  unrelated_lie: "False statement",
};
let health = { claude_configured: false, groq_configured: false, model: "", groq_model: "" };
let view = { caseId: null, tab: "summary", overview: null, state: null };
let evidence = { docId: null, passage: null, query: "", results: null };
let timelineRange = "window";
let autoLeads = true;
let poller = null;

/* --- helpers ----------------------------------------------------------- */

function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else if (key === "class") el.className = value;
    else if (key === "value") el.value = value;
    else if (key === "disabled" || key === "checked" || key === "required" || key === "selected") el[key] = Boolean(value);
    else el.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat(Infinity)) {
    if (child === null || child === undefined || child === false) continue;
    el.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return el;
}

function svg(tag, attrs, ...children) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
    else el.setAttribute(key, value);
  }
  el.append(...children);
  return el;
}

async function api(path, options = {}) {
  const headers = { "x-detective-client": "web", ...(options.headers || {}) };
  if (options.json !== undefined) {
    headers["content-type"] = "application/json";
    options.body = JSON.stringify(options.json);
  }
  const response = await fetch(path, { ...options, headers });
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d) => `${(d.loc || []).slice(1).join(" ")}: ${d.msg}`).join("; ");
    } catch { /* keep the status message */ }
    throw new Error(detail);
  }
  if (response.status === 204) return null;
  const type = response.headers.get("content-type") || "";
  return type.includes("json") ? response.json() : response.text();
}

function toast(message, bad = false) {
  const el = document.getElementById("toast");
  el.textContent = message;
  el.className = bad ? "toast bad" : "toast";
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => { el.hidden = true; }, bad ? 7000 : 3000);
}

function guarded(fn) {
  return async (...args) => {
    try { return await fn(...args); } catch (error) { toast(error.message, true); }
  };
}

const clock = (iso) => (iso || "").split("T")[1] || iso || "";
const when = (iso) => (iso || "").replace("T", " ");
const docOf = (passageId) => passageId.split(":")[0];
const enc = encodeURIComponent;

function cite(citation) {
  return h("button", {
    class: "cite",
    type: "button",
    title: citation.quote,
    "aria-label": `Open source ${citation.passage_id}`,
    onclick: () => openPassage(citation.passage_id),
  }, citation.passage_id);
}

function cites(citations) {
  const seen = new Map();
  for (const citation of citations || []) if (!seen.has(citation.passage_id)) seen.set(citation.passage_id, citation);
  return [...seen.values()].map(cite);
}

function openPassage(passageId) {
  evidence = { ...evidence, docId: docOf(passageId), passage: passageId, results: null, query: "" };
  location.hash = `#/case/${enc(view.caseId)}/evidence`;
  if (view.tab === "evidence") renderCase();
}

function empty(text) {
  return h("div", { class: "panel empty" }, text);
}

/* --- routing ----------------------------------------------------------- */

async function route() {
  clearInterval(poller);
  const [, section, id, tab] = location.hash.split("/");
  try {
    if (section === "case" && id) {
      await openCase(decodeURIComponent(id), tab || "summary");
    } else if (section === "new") {
      view.caseId = null;
      renderNewCase();
    } else {
      view.caseId = null;
      await renderHome();
    }
  } catch (error) {
    app.replaceChildren(h("div", { class: "panel error-box" }, error.message, " ", h("a", { href: "#/" }, "Back to cases")));
  }
  window.scrollTo(0, 0);
}

async function openCase(caseId, tab) {
  if (view.caseId !== caseId) {
    view = { caseId, tab, overview: null, state: null };
    evidence = { docId: null, passage: null, query: "", results: null };
  }
  view.tab = tab;
  [view.overview, view.state] = await Promise.all([api(`/api/cases/${enc(caseId)}`), api(`/api/cases/${enc(caseId)}/state`)]);
  renderCase();
  if (view.state.status === "running") startPolling();
}

function startPolling() {
  clearInterval(poller);
  poller = setInterval(guarded(async () => {
    const state = await api(`/api/cases/${enc(view.caseId)}/state`);
    view.state = state;
    if (state.status === "running") {
      renderProgress();
      return;
    }
    clearInterval(poller);
    renderCase();
    if (state.status === "done") toast("Investigation updated.");
  }), 500);
}

/* --- home -------------------------------------------------------------- */

async function renderHome() {
  const cases = await api("/api/cases");
  const importInput = h("input", { type: "file", accept: ".zip,application/zip", hidden: true, "aria-label": "Case archive", onchange: guarded(importCase) });
  app.replaceChildren(
    h("div", { class: "page-head" },
      h("div", {},
        h("h1", {}, "Case files"),
        h("p", { class: "lede" }, "Choose a case and investigate it. Every conclusion links back to the words in its source document.")),
      h("div", { class: "row" },
        importInput,
        h("button", { class: "btn", type: "button", onclick: () => importInput.click() }, "Import zip"),
        h("a", { class: "btn primary", href: "#/new" }, "New case"))),
    h("div", { class: "case-grid" }, cases.map(caseCard)));
}

function caseCard(item) {
  return h("a", { class: "case-card", href: `#/case/${enc(item.id)}/summary` },
    h("div", { class: "row" }, h("span", { class: "tag" }, item.difficulty)),
    h("h2", {}, item.title),
    h("p", {}, item.briefing),
    h("span", { class: "small muted" }, `${item.documents} documents · ${item.suspects} suspects · ${item.leads} sealed leads`),
    item.culprit ? h("span", { class: "verdict-line" }, `Finding: ${item.culprit} (${item.confidence})`) : null);
}

async function importCase(event) {
  const file = event.target.files[0];
  if (!file) return;
  const created = await api("/api/cases/import", { method: "POST", body: file, headers: { "content-type": "application/zip" } });
  toast("Case imported.");
  location.hash = `#/case/${enc(created.id)}/summary`;
}

/* --- case shell -------------------------------------------------------- */

function renderCase() {
  const { overview, state } = view;
  const tabs = TABS.filter(([id]) => id !== "key" || overview.has_answer_key);
  const body = h("div", { id: "tab-body" });
  const parts = [
    h("a", { class: "back", href: "#/" }, "← All cases"),
    h("div", { class: "page-head" },
      h("div", {},
        h("div", { class: "row" }, h("h1", {}, overview.title), h("span", { class: "tag" }, overview.difficulty)),
        h("p", { class: "lede" }, overview.briefing))),
    runPanel(),
    h("div", { id: "progress" }),
    state.status === "failed" ? h("div", { class: "panel error-box", role: "alert" }, h("strong", {}, "The run failed. "), state.error) : null,
    h("nav", { class: "tabs", "aria-label": "Case sections" }, tabs.map(([id, label]) =>
      h("a", { class: "tab", href: `#/case/${enc(view.caseId)}/${id}`, "aria-current": id === view.tab ? "page" : null }, label))),
    body,
  ];
  app.replaceChildren(...parts.filter(Boolean));
  renderProgress();
  guarded(fillTab)(body);
}

function runPanel() {
  const { overview, state } = view;
  const running = state.status === "running";
  const finished = Boolean(state.conclusion);
  const available = { claude: health.claude_configured, groq: health.groq_configured, offline: true };
  const preferred = state.rounds.length && available[state.backend] ? state.backend : ["claude", "groq", "offline"].find((id) => available[id]);
  const option = (id, label) => h("option", { value: id, disabled: !available[id], selected: id === preferred }, available[id] || id === "offline" ? label : `${label} (needs an API key)`);
  const backend = h("select", { id: "backend", "aria-label": "Investigator", disabled: running },
    option("claude", "Claude"), option("groq", "Groq"), option("offline", "Offline rules"));
  const auto = h("input", { type: "checkbox", id: "auto-leads", checked: autoLeads, disabled: running, onchange: (event) => { autoLeads = event.target.checked; } });
  return h("div", { class: "panel" },
    h("div", { class: "controls" },
      h("label", {}, "Investigator", backend),
      h("label", {}, auto, "Follow sealed leads automatically"),
      h("button", { class: "btn primary", type: "button", disabled: running, onclick: guarded(() => startRun(backend.value, auto.checked)) },
        running ? "Investigating…" : finished ? "Investigate again" : "Investigate"),
      finished ? h("a", { class: "btn", href: `/api/cases/${enc(view.caseId)}/report.md`, download: "" }, "Download report") : null,
      finished || state.status === "failed" ? h("button", { class: "btn", type: "button", disabled: running, onclick: guarded(resetCase) }, "Clear results") : null,
      overview.builtin ? null : h("button", { class: "btn danger", type: "button", disabled: running, onclick: guarded(deleteCase) }, "Delete case")),
    !overview.builtin && !health.claude_configured && !health.groq_configured
      ? h("p", { class: "notice" }, "Offline rules were written for the three sample cases and will miss a lot in your own. Set ANTHROPIC_API_KEY or GROQ_API_KEY and restart the server to use a model.")
      : null);
}

function renderProgress() {
  const slot = document.getElementById("progress");
  if (!slot) return;
  const { state } = view;
  if (state.status !== "running") {
    slot.replaceChildren();
    return;
  }
  const recent = state.steps.slice(-6).reverse();
  slot.replaceChildren(h("div", { class: "panel progress", role: "status" },
    h("div", { class: "row" }, h("span", { class: "spinner", "aria-hidden": "true" }), h("strong", {}, state.activity || "Working")),
    h("ol", {}, recent.map((step) => h("li", {}, step.observation)))));
}

async function startRun(backend, autoLeads) {
  await api(`/api/cases/${enc(view.caseId)}/runs`, { method: "POST", json: { backend, auto_leads: autoLeads } });
  view.state = await api(`/api/cases/${enc(view.caseId)}/state`);
  evidence.results = null;
  renderCase();
  startPolling();
}

async function resetCase() {
  if (!confirm("Clear this case's results, questions and opened leads?")) return;
  await api(`/api/cases/${enc(view.caseId)}/reset`, { method: "POST" });
  evidence = { docId: null, passage: null, query: "", results: null };
  view.state = await api(`/api/cases/${enc(view.caseId)}/state`);
  renderCase();
}

async function deleteCase() {
  if (!confirm("Delete this case and its results? This cannot be undone.")) return;
  await api(`/api/cases/${enc(view.caseId)}`, { method: "DELETE" });
  toast("Case deleted.");
  location.hash = "#/";
}

async function uncoverLead(leadId) {
  await api(`/api/cases/${enc(view.caseId)}/leads/${enc(leadId)}/uncover`, { method: "POST" });
  view.state = await api(`/api/cases/${enc(view.caseId)}/state`);
  renderCase();
  startPolling();
}

async function fillTab(body) {
  const fill = { summary: tabSummary, evidence: tabEvidence, timeline: tabTimeline, suspects: tabSuspects, contradictions: tabContradictions, reasoning: tabReasoning, ask: tabAsk, report: tabReport, key: tabKey }[view.tab] || tabSummary;
  const content = await fill();
  body.replaceChildren(...[content].flat().filter(Boolean));
  if (view.tab === "evidence" && evidence.passage) {
    document.getElementById(`passage-${evidence.passage}`)?.scrollIntoView({ block: "center" });
  }
}

const notYet = () => empty(view.state.status === "running" ? "The investigation is running. Results appear here when it finishes." : "Run an investigation to fill this section.");

/* --- summary ----------------------------------------------------------- */

function tabSummary() {
  const { state, overview } = view;
  const crime = overview.crime;
  const brief = h("div", { class: "panel" },
    h("h2", {}, "The crime"),
    h("p", {}, crime.summary),
    h("p", { class: "small muted" }, `Scene: ${crime.location} · Window: ${when(crime.window_start)} to ${when(crime.window_end)}`),
    h("div", { class: "row small" }, h("span", { class: "muted" }, "Suspects:"), overview.suspect_list.map((p) => h("span", { class: "tag" }, `${p.name}, ${p.role}`))));
  if (!state.conclusion) return [brief, leadsPanel()];
  const c = state.conclusion;
  const first = state.rounds[0];
  const verdict = h("section", { class: "panel verdict", "aria-label": "Conclusion" },
    h("div", {}, h("div", { class: "eyebrow" }, "Most likely suspect"), h("div", { class: "name" }, c.culprit || "None named")),
    h("div", { class: "confidence" }, h("div", { class: "eyebrow" }, "Confidence"), h("strong", { class: c.confidence }, c.confidence)),
    h("div", { class: "body" },
      h("p", {}, c.narrative),
      h("p", { class: "small muted" }, c.confidence_explanation),
      c.accomplices.length ? h("p", {}, h("strong", {}, "Possible accomplices: "), c.accomplices.join(", ")) : null,
      !c.engine_agrees ? h("p", { class: "notice" }, h("strong", {}, "Disagreement with the rule engine. "), c.engine_note) : null,
      first?.face_value_leader && first.face_value_leader !== c.culprit
        ? h("p", { class: "notice" }, `Counting the open evidence at face value points to ${first.face_value_leader}. That reading was rejected.`) : null));
  const final = state.rounds[state.rounds.length - 1];
  return [
    verdict,
    h("div", { class: "columns" }, roundsPanel(), [leadsPanel(), final?.open_questions.length ? h("div", { class: "panel" }, h("h2", {}, "Open questions"), h("ul", {}, final.open_questions.map((q) => h("li", {}, q)))) : null]),
    findingsPanel("fact", "Facts", "Directly supported by the sources."),
    findingsPanel("assumption", "Assumptions", "Reasonable inferences that go beyond the sources."),
    findingsPanel("speculation", "Speculation", "Possibilities with thin support. These never support the conclusion."),
    integrityPanel(),
    brief,
  ];
}

function roundsPanel() {
  const rounds = view.state.rounds;
  return h("div", { class: "panel" }, h("h2", {}, "How the assessment changed"),
    h("ol", { class: "rounds" }, rounds.map((round, index) => {
      const changed = index > 0 && rounds[index - 1].leader !== round.leader;
      return h("li", {},
        h("span", { class: "round-no" }, round.round),
        h("div", {}, h("div", { class: changed ? "flip" : null }, round.leader || "No leader", changed ? " (new leader)" : ""), h("div", { class: "small muted" }, round.trigger)),
        confidenceTag(round.confidence));
    })));
}

function confidenceTag(level) {
  return h("span", { class: `tag ${{ HIGH: "good", MEDIUM: "warn", LOW: "bad" }[level]}` }, level);
}

function leadsPanel() {
  const { state } = view;
  if (!state.pending_leads.length && !state.opened_leads.length) return null;
  const canOpen = Boolean(state.rounds.length) && state.status !== "running";
  return h("div", { class: "panel" },
    h("h2", {}, "Sealed leads"),
    state.pending_leads.length
      ? state.pending_leads.map((lead) => h("div", { class: "lead" },
          h("span", {}, lead.title),
          h("button", { class: "btn", type: "button", disabled: !canOpen, onclick: guarded(() => uncoverLead(lead.id)) }, "Open lead")))
      : h("p", { class: "muted" }, "All leads have been opened."),
    !canOpen && state.status !== "running" ? h("p", { class: "small muted" }, "Investigate first, then open leads one at a time to watch the assessment change.") : null);
}

function findingsPanel(status, title, note) {
  const found = view.state.findings.filter((f) => f.status === status);
  const supporting = new Set(view.state.conclusion.supporting_findings);
  return h("section", { class: "panel" }, h("h2", {}, title), h("p", { class: "small muted" }, note),
    found.length
      ? h("ul", { class: "findings" }, found.map((f) => h("li", {},
          h("span", {}, f.statement, supporting.has(f.id) ? h("span", { class: "tag", style: "margin-left:6px" }, "supports conclusion") : null),
          f.rationale ? h("span", { class: "rationale" }, f.rationale) : null,
          h("div", {}, cites(f.citations)))))
      : h("p", { class: "muted" }, "None recorded."));
}

function integrityPanel() {
  const { state } = view;
  return h("div", { class: "panel" }, h("h2", {}, "Integrity checks"),
    h("p", {}, `${state.rejections.length} extracted items and ${state.finding_rejections.length} findings were rejected for failing verification against the sources.`),
    [...state.rejections.slice(0, 5).map((r) => r.reason), ...state.finding_rejections.slice(0, 5)].map((text) => h("p", { class: "small muted" }, text)));
}

/* --- evidence ---------------------------------------------------------- */

async function tabEvidence() {
  const documents = await api(`/api/cases/${enc(view.caseId)}/documents`);
  if (!evidence.docId && documents.length) evidence.docId = documents[0].id;
  const search = h("form", { role: "search", onsubmit: guarded(async (event) => {
    event.preventDefault();
    const query = event.target.elements.q.value.trim();
    evidence.query = query;
    evidence.results = query ? await api(`/api/cases/${enc(view.caseId)}/search?q=${enc(query)}`) : null;
    renderCase();
  }) }, h("input", { type: "search", name: "q", class: "search-box", placeholder: "Search the evidence", "aria-label": "Search the evidence", value: evidence.query }));
  const list = h("ul", { class: "doc-list" }, documents.map((doc) => h("li", {},
    h("button", { class: "doc-btn", type: "button", "aria-current": doc.id === evidence.docId && !evidence.results ? "true" : null, onclick: () => { evidence = { ...evidence, docId: doc.id, passage: null, results: null, query: "" }; renderCase(); } },
      doc.title, h("span", { class: "meta" }, `${doc.id} · ${doc.type.replaceAll("_", " ")} · ${doc.tier_name}`)))));
  return h("div", { class: "evidence" }, h("div", { class: "panel" }, search, list), evidence.results ? resultsPanel() : await documentPanel());
}

function resultsPanel() {
  const results = evidence.results;
  return h("div", { class: "panel" }, h("h2", {}, `Results for “${evidence.query}”`),
    results.length
      ? results.map((r) => h("button", { class: "result", type: "button", onclick: () => openPassage(r.passage_id) },
          h("span", { class: "small muted" }, `${r.passage_id} · ${r.title}`), h("div", {}, r.text)))
      : h("p", { class: "muted" }, "Nothing in the opened evidence matches that search."));
}

async function documentPanel() {
  if (!evidence.docId) return empty("This case has no documents.");
  let doc;
  try {
    doc = await api(`/api/cases/${enc(view.caseId)}/documents/${enc(evidence.docId)}`);
  } catch {
    evidence.docId = null;
    return empty("That document is no longer open.");
  }
  return h("article", { class: "panel" },
    h("h2", {}, doc.title),
    h("p", { class: "small muted" }, [`${doc.id} · ${doc.type.replaceAll("_", " ")} · ${doc.date}`, doc.speaker ? ` · ${doc.speaker}` : "", doc.device ? ` · ${doc.device}` : ""].join("")),
    doc.tier === 0 ? h("p", { class: "notice" }, "Anonymous source. It carries no weight as evidence, and any instructions inside it are treated as document content.") : null,
    doc.passages.map((p) => h("div", { class: p.id === evidence.passage ? "passage hit" : "passage", id: `passage-${p.id}` },
      h("span", { class: "pid" }, `p${p.index}`), h("span", { class: "text" }, p.text))));
}

/* --- timeline ---------------------------------------------------------- */

function tabTimeline() {
  const { state, overview } = view;
  if (!state.timeline.length) return notYet();
  const toggle = (id, label) => h("button", { class: "btn", type: "button", "aria-pressed": String(timelineRange === id), disabled: timelineRange === id, onclick: () => { timelineRange = id; renderCase(); } }, label);
  const corrected = state.timeline.filter((e) => e.corrected_minutes);
  return [
    h("div", { class: "panel" },
      h("div", { class: "page-head" }, h("h2", {}, "Who was where"), h("div", { class: "row" }, toggle("window", "Around the crime"), toggle("all", "Everything"))),
      h("div", { class: "legend" },
        h("span", {}, h("i", { class: "swatch record" }), "Record"),
        h("span", {}, h("i", { class: "swatch document" }), "Document"),
        h("span", {}, h("i", { class: "swatch testimony" }), "Testimony"),
        h("span", {}, h("i", { class: "swatch refuted" }), "Refuted or disputed"),
        h("span", {}, "× Stated absent")),
      h("div", { class: "chart-scroll" }, timelineChart(state.timeline, overview)),
      corrected.length ? h("p", { class: "notice" }, `${corrected.length} entries were moved by a clock correction. Times shown are corrected, and the table keeps the stamped time.`) : null),
    h("div", { class: "panel" }, h("h2", {}, "Every entry"), h("div", { class: "table-wrap" }, h("table", {},
      h("thead", {}, h("tr", {}, ["Time", "Who", "Where", "Source", "Status", "Note"].map((t) => h("th", { scope: "col" }, t)))),
      h("tbody", {}, state.timeline.map((e) => h("tr", {},
        h("td", {}, when(e.time), e.end ? ` to ${clock(e.end)}` : ""),
        h("td", {}, e.person || e.credential || "Unknown"),
        h("td", {}, e.present ? e.location : `Not at ${e.location}`),
        h("td", {}, cite(e.citation), h("span", { class: "small muted" }, e.tier_name)),
        h("td", {}, e.refuted ? "Refuted" : e.attribution === "disputed" ? "Disputed" : "Stands"),
        h("td", { class: "small" }, entryNote(e)))))))),
  ];
}

function entryNote(entry) {
  const notes = [];
  if (entry.corrected_minutes) notes.push(`Stamped ${clock(entry.stated_time)}`);
  if (entry.attribution !== "direct") notes.push(`Via ${entry.credential}`);
  return notes.join(". ");
}

function timelineChart(entries, overview) {
  const { window_start: start, window_end: end } = overview.crime;
  const ms = (iso) => new Date(`${iso}:00`).getTime();
  const windowStart = ms(start), windowEnd = ms(end);
  const pad = Math.max((windowEnd - windowStart) * 0.5, 45 * 60000);
  let lo = windowStart - pad, hi = windowEnd + pad;
  if (timelineRange === "all") {
    lo = Math.min(...entries.map((e) => ms(e.time)));
    hi = Math.max(...entries.map((e) => ms(e.end || e.time)));
    const margin = (hi - lo) * 0.02 || 60000;
    lo -= margin; hi += margin;
  }
  const visible = entries.filter((e) => e.person && ms(e.time) <= hi && ms(e.end || e.time) >= lo && e.location);
  const suspects = overview.suspect_list.map((p) => p.name);
  const people = [...new Set(visible.map((e) => e.person))].sort((a, b) => (suspects.indexOf(a) + 1 || 99) - (suspects.indexOf(b) + 1 || 99));
  const left = 150, right = 985, top = 46, rowHeight = 18, width = 1000;
  const x = (t) => left + ((Math.min(hi, Math.max(lo, t)) - lo) / (hi - lo)) * (right - left);
  const lanes = people.map((person) => layoutLane(visible.filter((e) => e.person === person), (e) => x(ms(e.time)), (e) => x(ms(e.end || e.time))));
  const laneHeight = (rows) => Math.max(rows * rowHeight + 10, 36);
  const height = top + lanes.reduce((sum, placed) => sum + laneHeight(placed.rows), 0) + 12;
  const root = svg("svg", { viewBox: `0 0 ${width} ${height}`, role: "img", "aria-label": "Timeline of each person's movements. The table below holds the same data." });
  const ticks = 6;
  for (let i = 0; i <= ticks; i += 1) {
    const t = lo + ((hi - lo) * i) / ticks;
    const px = x(t);
    const stamp = new Date(t);
    const label = `${String(stamp.getHours()).padStart(2, "0")}:${String(stamp.getMinutes()).padStart(2, "0")}`;
    root.append(svg("line", { class: "tl-grid", x1: px, x2: px, y1: top - 6, y2: height - 8 }));
    root.append(svg("text", { class: "tl-tick", x: px, y: top - 12, "text-anchor": i === ticks ? "end" : i === 0 ? "start" : "middle" }, document.createTextNode(label)));
  }
  root.append(svg("rect", { class: "tl-window", x: x(windowStart), y: top - 6, width: Math.max(x(windowEnd) - x(windowStart), 3), height: height - top - 2 }));
  root.append(svg("text", { class: "tl-window-label", x: (x(windowStart) + x(windowEnd)) / 2, y: 12, "text-anchor": "middle" }, document.createTextNode("Crime window")));
  let y = top;
  people.forEach((person, index) => {
    const placed = lanes[index];
    const lane = laneHeight(placed.rows);
    root.append(svg("line", { class: "tl-grid", x1: 0, x2: right, y1: y + lane, y2: y + lane }));
    root.append(svg("text", { class: "tl-lane", x: 0, y: y + lane / 2 + 4 }, document.createTextNode(person)));
    for (const [entry, rowIndex] of placed.entries) {
      const row = y + (lane - placed.rows * rowHeight) / 2 + rowHeight * rowIndex + rowHeight / 2;
      const x1 = x(ms(entry.time));
      const doubtful = entry.refuted || entry.attribution === "disputed";
      const cls = `tl-mark ${entry.tier_name}${doubtful ? " refuted" : ""}`;
      let mark, hit;
      if (entry.end) {
        const barWidth = Math.max(x(ms(entry.end)) - x1, 4);
        mark = svg("rect", { class: cls, x: x1, y: row - 6, width: barWidth, height: 12, rx: 4 });
        hit = svg("rect", { class: "tl-hit", x: x1, y: row - 7, width: barWidth, height: 14 });
      } else if (!entry.present) {
        mark = svg("path", { class: "tl-cross", d: `M${x1 - 5} ${row - 5}L${x1 + 5} ${row + 5}M${x1 + 5} ${row - 5}L${x1 - 5} ${row + 5}` });
        hit = svg("circle", { class: "tl-hit", cx: x1, cy: row, r: 9 });
      } else {
        mark = svg("circle", { class: cls, cx: x1, cy: row, r: 6 });
        hit = svg("circle", { class: "tl-hit", cx: x1, cy: row, r: 9 });
      }
      hit.addEventListener("mousemove", (event) => showTooltip(event, entry));
      hit.addEventListener("mouseleave", hideTooltip);
      hit.addEventListener("click", () => openPassage(entry.citation.passage_id));
      root.append(mark, hit);
    }
    y += lane;
  });
  if (!people.length) root.append(svg("text", { class: "tl-tick", x: left, y: top + 20 }, document.createTextNode("No entries in this range.")));
  return root;
}

/* Stronger sources take the top rows, and marks that would overlap move down a row. */
function layoutLane(entries, startX, endX) {
  const order = [...entries].sort((a, b) => b.tier - a.tier || a.time.localeCompare(b.time));
  const rowEnds = [];
  const placed = order.map((entry) => {
    const from = startX(entry) - 8;
    let row = rowEnds.findIndex((end) => end < from);
    if (row === -1) row = rowEnds.length;
    rowEnds[row] = endX(entry) + 8;
    return [entry, row];
  });
  return { entries: placed, rows: Math.max(rowEnds.length, 1) };
}

function showTooltip(event, entry) {
  const tip = document.getElementById("tooltip");
  const status = entry.refuted ? "Refuted by a stronger source" : entry.attribution === "disputed" ? "Attribution disputed" : entry.tier_name;
  tip.replaceChildren(
    h("strong", {}, entry.person), h("br"),
    `${entry.present ? "" : "Not at "}${entry.location}`, h("br"),
    `${clock(entry.time)}${entry.end ? ` to ${clock(entry.end)}` : ""} · ${entry.doc_id} · ${status}`);
  tip.hidden = false;
  tip.style.left = `${Math.min(event.clientX + 14, innerWidth - 336)}px`;
  tip.style.top = `${event.clientY + 14}px`;
}

function hideTooltip() {
  document.getElementById("tooltip").hidden = true;
}

/* --- suspects, contradictions, reasoning ------------------------------- */

function tabSuspects() {
  const { state } = view;
  if (!state.suspects.length) return notYet();
  const conclusion = state.conclusion;
  return state.suspects.map((s) => h("article", { class: "panel suspect" },
    h("div", {},
      h("h2", {}, s.name),
      h("div", { class: "small muted" }, s.role),
      h("div", { class: "row", style: "margin-top:6px" },
        s.name === conclusion?.culprit ? h("span", { class: "tag bad" }, "Named") : null,
        conclusion?.accomplices.includes(s.name) ? h("span", { class: "tag warn" }, "Possible accomplice") : null,
        h("span", { class: `tag ${s.opportunity === "excluded" ? "good" : s.opportunity === "direct" ? "bad" : ""}` }, `Opportunity: ${s.opportunity}`),
        h("span", { class: `tag ${s.alibi === "verified" ? "good" : s.alibi === "broken" ? "bad" : ""}` }, `Alibi: ${s.alibi}`))),
    h("div", { class: "score", title: "Rule engine ranking score, not a probability" }, s.score),
    h("div", { class: "detail" },
      h("div", { class: "breakdown" }, Object.entries(s.breakdown).map(([key, value]) => h("span", { class: "tag" }, `${LABELS[key] || key} ${value > 0 ? "+" : ""}${value}`))),
      s.opportunity_note ? h("p", { class: "small" }, h("strong", {}, "Opportunity: "), s.opportunity_note) : null,
      s.alibi_note ? h("p", { class: "small" }, h("strong", {}, "Alibi: "), s.alibi_note) : null,
      signals("Motive", s.motives), signals("Means", s.means), signals("Suspicious conduct", s.suspicious),
      s.links.map((link) => h("div", {}, h("strong", { class: "small" }, "Connection"), h("div", { class: "chain" }, link.path.map((node, i) => [i ? "→" : null, h("code", {}, node)]), cites(link.citations)))),
      conclusion?.eliminated[s.name] ? h("p", { class: "small muted" }, conclusion.eliminated[s.name]) : null)));
}

function signals(label, items) {
  return items.map((item) => h("p", { class: "signal" }, h("strong", {}, `${label}: `), item.detail, " ", cite(item.citation)));
}

function tabContradictions() {
  const { state } = view;
  if (!state.conclusion && !state.contradictions.length) return notYet();
  const undermined = [
    ...state.defeaters.map((d) => h("li", {}, d.detail, " ", cite(d.citation))),
    ...Object.entries(state.compromised).map(([credential, reason]) => h("li", {}, `${credential}: ${reason}`)),
  ];
  return [
    state.contradictions.length ? state.contradictions.map((c) => h("article", { class: "panel" },
      h("div", { class: "row" }, h("h3", {}, c.person), h("span", { class: "tag" }, c.kind.replaceAll("_", " "))),
      h("p", { style: "margin-top:6px" }, c.summary),
      h("p", { class: "small" }, h("strong", {}, "Resolution: "), c.resolution),
      cites(c.citations))) : empty("No contradictions were found."),
    undermined.length ? h("div", { class: "panel" }, h("h2", {}, "Evidence that was undermined"), h("ul", {}, undermined)) : null,
  ];
}

function tabReasoning() {
  const { state } = view;
  if (!state.steps.length) return notYet();
  const interesting = state.steps.filter((s) => s.action !== "extract");
  const extracted = state.steps.length - interesting.length;
  return h("div", { class: "panel" },
    h("h2", {}, "Reasoning log"),
    h("p", { class: "small muted" }, `${extracted} documents were read first. Every later step is shown in order.`),
    h("ol", { class: "steps" }, interesting.map((s) => h("li", {},
      h("span", { class: "action" }, s.action.replaceAll("_", " ")),
      h("div", {}, s.observation, " ", cites(s.citations))))));
}

/* --- ask, report, answer key ------------------------------------------- */

function tabAsk() {
  const { state } = view;
  if (!state.rounds.length) return notYet();
  const input = h("input", { type: "text", name: "question", placeholder: "Ask about the evidence", "aria-label": "Question", required: true, maxlength: 1000, style: "flex:1;min-width:220px" });
  const submit = h("button", { class: "btn primary", type: "submit", disabled: state.status === "running" }, "Ask");
  const form = h("form", { class: "row", onsubmit: guarded(async (event) => {
    event.preventDefault();
    submit.disabled = true;
    submit.textContent = "Checking the file…";
    try {
      await api(`/api/cases/${enc(view.caseId)}/questions`, { method: "POST", json: { question: input.value } });
      view.state = await api(`/api/cases/${enc(view.caseId)}/state`);
      renderCase();
    } finally {
      submit.disabled = false;
      submit.textContent = "Ask";
    }
  }) }, input, submit);
  const history = [...state.questions].reverse();
  return [
    h("div", { class: "panel" }, h("h2", {}, "Follow-up questions"), h("p", { class: "small muted" }, "Answers come only from opened evidence. If the file has no answer, the detective says so."), form),
    history.length ? h("div", { class: "panel" }, history.map((q) => h("div", { class: "answer" },
      h("div", { class: "q" }, q.question),
      h("div", { class: "a" }, q.answer),
      h("div", { class: "row" }, h("span", { class: "tag" }, q.status), cites(q.citations))))) : null,
  ];
}

async function tabReport() {
  if (!view.state.conclusion) return notYet();
  const text = await api(`/api/cases/${enc(view.caseId)}/report.md`);
  return h("div", { class: "panel report" }, markdown(text));
}

function markdown(text) {
  const out = [];
  const lines = text.split("\n");
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (line.startsWith("# ")) out.push(h("h1", {}, inline(line.slice(2))));
    else if (line.startsWith("## ")) out.push(h("h2", {}, inline(line.slice(3))));
    else if (line.startsWith("- ")) {
      const items = [];
      while (i < lines.length && lines[i].startsWith("- ")) items.push(h("li", {}, inline(lines[i++].slice(2))));
      i -= 1;
      out.push(h("ul", {}, items));
    } else if (line.startsWith("|")) {
      const rows = [];
      while (i < lines.length && lines[i].startsWith("|")) rows.push(lines[i++].split("|").slice(1, -1).map((c) => c.trim()));
      i -= 1;
      out.push(h("div", { class: "table-wrap" }, h("table", {},
        h("thead", {}, h("tr", {}, rows[0].map((c) => h("th", {}, c)))),
        h("tbody", {}, rows.slice(2).map((r) => h("tr", {}, r.map((c) => h("td", {}, inline(c)))))))));
    } else if (line.trim()) out.push(h("p", {}, inline(line.replace(/^> /, ""))));
  }
  return out;
}

function inline(text) {
  return text.split(/(\*\*[^*]+\*\*|\*[^*]+\*|\[[A-Za-z0-9]+:p\d+\])/).filter(Boolean).map((part) => {
    if (part.startsWith("**")) return h("strong", {}, part.slice(2, -2));
    if (part.startsWith("*")) return h("em", {}, part.slice(1, -1));
    if (/^\[[A-Za-z0-9]+:p\d+\]$/.test(part)) return cite({ passage_id: part.slice(1, -1), quote: "" });
    return part;
  });
}

function tabKey() {
  if (!view.state.conclusion) return notYet();
  const slot = h("div", { id: "key-slot" }, h("div", { class: "panel" },
    h("h2", {}, "Check against the answer key"),
    h("p", {}, "This compares the current finding with the case's predefined solution and reveals the culprit."),
    h("button", { class: "btn primary", type: "button", onclick: guarded(async () => slot.replaceChildren(keyResult(await api(`/api/cases/${enc(view.caseId)}/evaluation`)))) }, "Reveal and score")));
  return slot;
}

function keyResult(score) {
  const percent = (value) => `${Math.round(value * 100)}%`;
  const yes = (value) => (value ? "Yes" : "No");
  const rows = [
    ["Culprit", `${score.named} (expected ${score.expected})`, score.culprit_correct],
    ["Confidence", score.confidence, score.confidence_ok],
    ["Accomplices correct", yes(score.accomplices_correct), score.accomplices_correct],
    ["Contradiction recall", percent(score.contradiction_recall), score.contradiction_recall === 1],
    ["Timeline recall", percent(score.timeline_recall), score.timeline_recall === 1],
    ["Key evidence cited", percent(score.key_evidence_recall), score.key_evidence_recall === 1],
    ["Retrieval recall at 5", percent(score.retrieval_recall), score.retrieval_recall === 1],
    ["Citation validity", percent(score.citation_validity), score.citation_validity === 1],
    ["Red herrings avoided", yes(score.red_herrings_clean), score.red_herrings_clean],
  ];
  return h("div", { class: "panel" },
    h("div", { class: "row" }, h("h2", {}, "Score"), h("span", { class: `tag ${score.passed ? "good" : "bad"}` }, score.passed ? "Pass" : "Fail")),
    h("div", { class: "table-wrap", style: "margin-top:10px" }, h("table", {}, h("tbody", {}, rows.map(([name, value, ok]) => h("tr", {},
      h("th", { scope: "row" }, name), h("td", {}, value), h("td", {}, h("span", { class: `tag ${ok ? "good" : "warn"}` }, ok ? "Met" : "Missed"))))))),
    score.misses.length ? [h("h3", { style: "margin-top:14px" }, "Missed"), h("ul", { class: "small" }, score.misses.map((m) => h("li", {}, m)))] : null);
}

/* --- new case ---------------------------------------------------------- */

function renderNewCase() {
  const documents = h("div", {});
  const form = h("form", { onsubmit: guarded(submitCase) },
    h("div", { class: "panel" }, h("h2", {}, "The case"),
      h("div", { class: "form-grid" },
        field("Title", h("input", { type: "text", name: "title", required: true })),
        field("Crime scene", h("input", { type: "text", name: "scene", required: true }), "Must match a location below."),
        field("Crime window starts", h("input", { type: "datetime-local", name: "window_start", required: true })),
        field("Crime window ends", h("input", { type: "datetime-local", name: "window_end", required: true })),
        field("Briefing", h("textarea", { name: "briefing", required: true }), "What happened, in a sentence or two.", true),
        field("What was done", h("textarea", { name: "crime_summary", required: true }), null, true))),
    h("div", { class: "panel" }, h("h2", {}, "People and places"),
      h("div", { class: "form-grid" },
        field("Suspects", h("textarea", { name: "suspects", class: "code", required: true, placeholder: "Ana Bell | Sister | Ana, A. Bell\nBen Cole | Brother | Ben" }), "One per line: name | role | other names."),
        field("Other people", h("textarea", { name: "people", class: "code", placeholder: "Mara Okafor | Victim | Mara" }), "Same format. Optional."),
        field("Locations", h("textarea", { name: "locations", class: "code", required: true, placeholder: "House\nKitchen | House\nGarden | House" }), "One per line: name | the place it sits inside."))),
    h("div", { class: "panel" }, h("div", { class: "page-head", style: "margin-bottom:0" }, h("h2", {}, "Evidence"),
        h("button", { class: "btn", type: "button", onclick: () => documents.append(documentDraft()) }, "Add document")),
      documents),
    h("div", { class: "row" }, h("button", { class: "btn primary", type: "submit" }, "Create case"), h("a", { class: "btn", href: "#/" }, "Cancel")));
  documents.append(documentDraft());
  app.replaceChildren(
    h("a", { class: "back", href: "#/" }, "← All cases"),
    h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "New case"),
      h("p", { class: "lede" }, "Enter the people, places and evidence. For logs, one line per event works best, such as 15:10 | Ana Bell | Kitchen | ENTRY. To include sealed leads, import a zip instead."))),
    form);
}

function field(label, control, hint, wide) {
  return h("label", { class: wide ? "field wide" : "field" }, label, control, hint ? h("span", { class: "hint" }, hint) : null);
}

function documentDraft() {
  const text = h("textarea", { name: "doc_text", required: true, style: "min-height:130px" });
  const title = h("input", { type: "text", name: "doc_title", required: true });
  const file = h("input", { type: "file", accept: ".txt,.md,text/plain", "aria-label": "Load document from a text file", onchange: guarded(async (event) => {
    const chosen = event.target.files[0];
    if (!chosen) return;
    text.value = await chosen.text();
    if (!title.value) title.value = chosen.name.replace(/\.[^.]+$/, "");
  }) });
  const draft = h("div", { class: "doc-draft" },
    h("div", { class: "form-grid" },
      field("Document title", title),
      field("Type", h("select", { name: "doc_type" }, DOC_TYPES.map(([value, label]) => h("option", { value }, label)))),
      field("Date", h("input", { type: "text", name: "doc_date", required: true, placeholder: "2025-01-10" })),
      field("Speaker", h("input", { type: "text", name: "doc_speaker" }), "For statements: must match a person above."),
      field("Device", h("input", { type: "text", name: "doc_device" }), "For logs and cameras, such as Lobby camera."),
      field("Load from file", file),
      field("Text", text, null, true)),
    h("button", { class: "btn danger", type: "button", style: "margin-top:8px", onclick: () => draft.remove() }, "Remove document"));
  return draft;
}

function parsePeople(text) {
  return text.split("\n").map((line) => line.trim()).filter(Boolean).map((line) => {
    const [name, role = "", aliases = ""] = line.split("|").map((part) => part.trim());
    return { name, role, aliases: aliases.split(",").map((a) => a.trim()).filter(Boolean) };
  });
}

function parseLocations(text) {
  return text.split("\n").map((line) => line.trim()).filter(Boolean).map((line) => {
    const [name, within] = line.split("|").map((part) => part.trim());
    return { name, within: within || null, aliases: [] };
  });
}

async function submitCase(event) {
  event.preventDefault();
  const form = event.target;
  const value = (name) => form.elements[name].value.trim();
  const documents = [...form.querySelectorAll(".doc-draft")].map((draft) => {
    const part = (name) => draft.querySelector(`[name="${name}"]`).value.trim();
    return { title: part("doc_title"), type: part("doc_type"), date: part("doc_date"), speaker: part("doc_speaker") || null, device: part("doc_device") || null, text: part("doc_text") };
  });
  const created = await api("/api/cases", { method: "POST", json: {
    title: value("title"),
    briefing: value("briefing"),
    crime_summary: value("crime_summary"),
    scene: value("scene"),
    window_start: value("window_start").slice(0, 16),
    window_end: value("window_end").slice(0, 16),
    suspects: parsePeople(value("suspects")),
    people: parsePeople(value("people")),
    locations: parseLocations(value("locations")),
    documents,
  } });
  toast("Case created.");
  location.hash = `#/case/${enc(created.id)}/summary`;
}

/* --- start ------------------------------------------------------------- */

async function start() {
  try {
    health = await api("/api/health");
    const ready = [health.claude_configured ? `Claude (${health.model})` : null, health.groq_configured ? `Groq (${health.groq_model})` : null].filter(Boolean);
    document.getElementById("engine-status").textContent = ready.length ? `Ready: ${ready.join(", ")}` : "Offline mode: no API key set";
  } catch {
    document.getElementById("engine-status").textContent = "Server unavailable";
  }
  window.addEventListener("hashchange", route);
  route();
}

start();
