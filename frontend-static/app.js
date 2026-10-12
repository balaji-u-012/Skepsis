// Skepsis frontend — vanilla JS, no build step, no framework.
// Talks to the Skepsis backend (default http://127.0.0.1:8000).

const API_BASE = window.SKEPSIS_API_BASE || "http://127.0.0.1:8000";

const els = {
  form: document.getElementById("searchForm"),
  topicInput: document.getElementById("topicInput"),
  yearFrom: document.getElementById("yearFrom"),
  yearTo: document.getElementById("yearTo"),
  searchBtn: document.getElementById("searchBtn"),
  pipelineStatus: document.getElementById("pipelineStatus"),
  pipelineSteps: document.getElementById("pipelineSteps"),
  errorBanner: document.getElementById("errorBanner"),
  resultsMeta: document.getElementById("resultsMeta"),
  paperResults: document.getElementById("paperResults"),
  comparisonSection: document.getElementById("comparisonSection"),
  comparisonTable: document.getElementById("comparisonTable"),
  similaritiesList: document.getElementById("similaritiesList"),
  differencesList: document.getElementById("differencesList"),
  gapsList: document.getElementById("gapsList"),
  detailBackdrop: document.getElementById("detailBackdrop"),
  detailModal: document.getElementById("detailModal"),
  detailContent: document.getElementById("detailContent"),
  detailClose: document.getElementById("detailClose"),
  statusDot: document.getElementById("statusDot"),
  statusText: document.getElementById("statusText"),
};

const SOURCE_SLUG = {
  "arXiv": "arxiv",
  "Semantic Scholar": "semantic-scholar",
  "OpenAlex": "openalex",
  "Crossref": "crossref",
};

const NOT_REPORTED = "Not reported";

// Interval IDs stored at module scope so we can always clear the previous
// run before starting a new one (otherwise overlapping searches would
// leave orphaned intervals manipulating the same DOM).
let pipelineIntervalId = null;

// ---------------------------------------------------------------- init

checkHealth();
// Retry the health check a few times so a just-started backend is picked
// up without requiring a page reload.
setTimeout(checkHealth, 3000);
setTimeout(checkHealth, 8000);

els.form.addEventListener("submit", onSearchSubmit);
els.detailClose.addEventListener("click", closeDetail);
els.detailBackdrop.addEventListener("click", (e) => {
  if (e.target === els.detailBackdrop) closeDetail();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeDetail();
});

async function checkHealth() {
  try {
    const res = await fetch(`${API_BASE}/api/health`);
    if (!res.ok) throw new Error("bad status");
    const data = await res.json();
    els.statusDot.classList.remove("off");
    els.statusDot.classList.add("ok");
    els.statusText.textContent = data.llm_enabled
      ? "backend connected · LLM enrichment on"
      : "backend connected · heuristic mode";
  } catch (err) {
    els.statusDot.classList.remove("ok");
    els.statusDot.classList.add("off");
    els.statusText.textContent =
      "backend unreachable — run `python -m backend.main`";
  }
}

// ---------------------------------------------------------------- search

async function onSearchSubmit(event) {
  event.preventDefault();
  const topic = els.topicInput.value.trim();
  if (!topic) return;

  const yearFrom = parseYearOrNull(els.yearFrom.value, "From year");
  if (yearFrom === false) return;
  const yearTo = parseYearOrNull(els.yearTo.value, "To year");
  if (yearTo === false) return;

  if (yearFrom !== null && yearTo !== null && yearFrom > yearTo) {
    showError("\"From year\" must not be greater than \"To year\".");
    return;
  }

  setLoading(true);
  showPipeline();
  hideError();
  els.comparisonSection.classList.add("hidden");
  els.paperResults.innerHTML = "";
  els.resultsMeta.classList.add("hidden");

  try {
    startPipelineAnimation();
    const res = await fetch(`${API_BASE}/api/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        topic,
        year_from: yearFrom,
        year_to: yearTo,
        max_results: 5,
      }),
    });

    if (!res.ok) {
      const problem = await safeJson(res);
      throw new Error(problem?.detail || `Search failed (HTTP ${res.status}).`);
    }

    const data = await res.json();
    completePipeline();
    renderResults(data);
  } catch (err) {
    stopPipelineAnimation();
    resetPipeline();
    showError(err.message || "Something went wrong while searching.");
  } finally {
    setLoading(false);
  }
}

function parseYearOrNull(rawValue, label) {
  const value = (rawValue || "").trim();
  if (!value) return null;
  const parsed = Number.parseInt(value, 10);
  if (!Number.isFinite(parsed) || parsed < 1900 || parsed > 2100) {
    showError(`${label} must be a year between 1900 and 2100.`);
    return false;  // sentinel: caller should abort
  }
  return parsed;
}

function setLoading(isLoading) {
  els.searchBtn.disabled = isLoading;
  els.searchBtn.textContent = isLoading ? "Searching…" : "Search the literature";
}

// ---------------------------------------------------------------- pipeline visual

// Cosmetic progression while the backend runs the real pipeline in a single
// request/response. Slowed to ~800ms/step so it doesn't finish long before
// the actual search (which can take 5-30s with LLM enrichment).
const PIPELINE_STEP_MS = 800;

function showPipeline() {
  els.pipelineStatus.classList.remove("hidden");
  resetPipeline();
}

function resetPipeline() {
  [...els.pipelineSteps.children].forEach((li) => li.classList.remove("active", "done"));
}

function startPipelineAnimation() {
  stopPipelineAnimation();  // clear any previous run
  let i = 0;
  const items = [...els.pipelineSteps.children];
  pipelineIntervalId = setInterval(() => {
    if (i > 0) items[i - 1].classList.replace("active", "done");
    if (i >= items.length) {
      stopPipelineAnimation();
      return;
    }
    items[i].classList.add("active");
    i += 1;
  }, PIPELINE_STEP_MS);
}

function stopPipelineAnimation() {
  if (pipelineIntervalId !== null) {
    clearInterval(pipelineIntervalId);
    pipelineIntervalId = null;
  }
}

function completePipeline() {
  stopPipelineAnimation();
  [...els.pipelineSteps.children].forEach((li) => {
    li.classList.remove("active");
    li.classList.add("done");
  });
}

// ---------------------------------------------------------------- errors

function showError(message) {
  els.errorBanner.textContent = message;
  els.errorBanner.classList.remove("hidden");
}
function hideError() {
  els.errorBanner.classList.add("hidden");
}

async function safeJson(res) {
  try { return await res.json(); } catch { return null; }
}

// ---------------------------------------------------------------- rendering results

function renderResults(data) {
  els.resultsMeta.textContent =
    `${data.papers.length} of ${data.total_candidates} deduplicated candidates shown, ` +
    `ranked by relevance to "${data.topic}" and recency.`;
  els.resultsMeta.classList.remove("hidden");

  els.paperResults.innerHTML = "";
  data.papers.forEach((ap, index) => {
    els.paperResults.appendChild(renderPaperCard(ap, index));
  });

  renderComparison(data.comparison, data.papers);
  els.comparisonSection.classList.remove("hidden");
}

function renderPaperCard(analyzedPaper, index) {
  const { paper, analysis } = analyzedPaper;
  const slug = SOURCE_SLUG[paper.source]
    || (paper.source || "unknown").toLowerCase().replace(/\s+/g, "-");

  const card = document.createElement("article");
  card.className = "paper-card";
  card.tabIndex = 0;
  card.setAttribute("role", "button");
  card.setAttribute("aria-label", `Open details for ${paper.title}`);

  const scorePct = paper.final_score != null ? Math.round(paper.final_score * 100) : null;

  card.innerHTML = `
    <div class="paper-card__spine spine-${slug}"></div>
    <div class="paper-card__body">
      <div class="paper-card__top">
        <span class="paper-card__rank">#${index + 1}</span>
        ${scorePct != null ? `<span class="paper-card__score">${scorePct}% match</span>` : ""}
      </div>
      <h3 class="paper-card__title">${escapeHtml(paper.title)}</h3>
      <p class="paper-card__meta">
        <span class="paper-card__source-tag tag-${slug}">${escapeHtml(paper.source)}</span>
        ${paper.year ? escapeHtml(String(paper.year)) + " · " : ""}${escapeHtml(formatAuthors(paper.authors))}
      </p>
      <p class="paper-card__summary">${escapeHtml(truncate(analysis.abstract, 220))}</p>
    </div>
  `;

  const open = () => openDetail(analyzedPaper);
  card.addEventListener("click", open);
  card.addEventListener("keydown", (e) => {
    if (e.key === "Enter" || e.key === " ") { e.preventDefault(); open(); }
  });

  return card;
}

function formatAuthors(authors) {
  if (!authors || authors.length === 0) return NOT_REPORTED;
  // Deduplicate while preserving order.
  const unique = [...new Set(authors)];
  if (unique.length <= 3) return unique.join(", ");
  return `${unique.slice(0, 3).join(", ")} +${unique.length - 3} more`;
}

// ---------------------------------------------------------------- detail modal

function openDetail(analyzedPaper) {
  const { paper, analysis } = analyzedPaper;
  els.detailContent.innerHTML = `
    <h2 class="detail-title" id="detailTitle">${escapeHtml(paper.title)}</h2>
    <p class="detail-meta">${escapeHtml(formatAuthors(paper.authors))}</p>
    <p class="detail-meta">${escapeHtml(paper.source)}${paper.year ? " · " + escapeHtml(String(paper.year)) : ""}${paper.venue ? " · " + escapeHtml(paper.venue) : ""}</p>
    <div class="detail-links">
      ${paper.url ? `<a href="${escapeAttr(paper.url)}" target="_blank" rel="noopener">Original paper ↗</a>` : ""}
      ${paper.doi ? `<a href="https://doi.org/${escapeAttr(paper.doi)}" target="_blank" rel="noopener">DOI: ${escapeHtml(paper.doi)}</a>` : ""}
    </div>
    <dl class="detail-fields">
      ${detailField("Abstract", analysis.abstract)}
      ${detailField("Problem", analysis.problem)}
      ${detailField("Methodology", analysis.methodology)}
      ${detailField("Algorithm / model", analysis.algorithm)}
      ${detailField("Dataset", analysis.dataset)}
      ${detailField("Results", analysis.results)}
      ${detailField("Limitations", analysis.limitations)}
      ${detailField("Novelty", analysis.novelty)}
    </dl>
  `;
  els.detailBackdrop.classList.remove("hidden");
  els.detailClose.focus();
}

function detailField(label, value) {
  const isNotReported = value === NOT_REPORTED || !value;
  return `
    <div class="detail-field">
      <dt>${escapeHtml(label)}</dt>
      <dd class="${isNotReported ? "not-reported" : ""}">${escapeHtml(value || NOT_REPORTED)}</dd>
    </div>
  `;
}

function closeDetail() {
  els.detailBackdrop.classList.add("hidden");
}

// ---------------------------------------------------------------- comparison

function renderComparison(comparison, papers) {
  const rows = comparison.rows;
  const fieldDefs = [
    ["Problem", "problem"],
    ["Methodology", "methodology"],
    ["Results", "results"],
    ["Limitations", "limitations"],
    ["Novelty", "novelty"],
  ];

  let html = "<thead><tr><th>Paper</th>";
  rows.forEach((row, i) => {
    html += `<th>#${i + 1} ${escapeHtml(truncate(row.title, 60))}</th>`;
  });
  html += "</tr></thead><tbody>";

  fieldDefs.forEach(([label, key]) => {
    html += `<tr><th>${escapeHtml(label)}</th>`;
    rows.forEach((row) => {
      const value = row[key] || NOT_REPORTED;
      const isNotReported = value === NOT_REPORTED;
      html += `<td class="${isNotReported ? "not-reported" : ""}">${escapeHtml(value)}</td>`;
    });
    html += "</tr>";
  });
  html += "</tbody>";

  els.comparisonTable.innerHTML = html;

  renderList(els.similaritiesList, comparison.similarities, "No clear similarities identified from the available text.");
  renderList(els.differencesList, comparison.differences, "No clear differences identified from the available text.");
  renderList(els.gapsList, comparison.research_gaps, "No research gaps identified from the available text.");
}

function renderList(el, items, emptyMessage) {
  el.innerHTML = "";
  const list = items && items.length ? items : [emptyMessage];
  list.forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    el.appendChild(li);
  });
}

// ---------------------------------------------------------------- helpers

function truncate(text, max) {
  if (!text) return NOT_REPORTED;
  return text.length > max ? text.slice(0, max - 1).trimEnd() + "…" : text;
}

function escapeHtml(str) {
  if (str == null) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function escapeAttr(str) {
  return escapeHtml(str);
}