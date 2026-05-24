const API_BASE = "http://localhost:8000/api";

let activeQuoteIndex = 0;
let quoteRotationTimer = null;

const state = {
  ticker: null,
  profile: null,
  financials: [],
  ratios: null,
  peers: null,
  valuation: null,
};

const quotes = [
  {
    text: "The stock market is filled with individuals who know the price of everything, but the value of nothing.",
    author: "Philip Fisher",
  },
  {
    text: "Price is what you pay. Value is what you get.",
    author: "Warren Buffett",
  },
  {
    text: "In investing, what is comfortable is rarely profitable.",
    author: "Robert Arnott",
  },
  {
    text: "The intelligent investor is a realist who sells to optimists and buys from pessimists.",
    author: "Benjamin Graham",
  },
];

document.addEventListener("DOMContentLoaded", () => {
  initTheme();

  if (document.querySelector(".home-page")) {
    initHomePage();
  }

  if (document.querySelector("#analysisPage")) {
    initAnalysisPage();
  }
});

function initTheme() {
  const storedTheme = localStorage.getItem("investiq-theme");
  const theme = storedTheme || "dark";
  document.documentElement.setAttribute("data-theme", theme);
  updateThemeIcon(theme);

  const toggle = document.getElementById("themeToggle");
  if (!toggle) return;

  toggle.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    document.documentElement.setAttribute("data-theme", next);
    localStorage.setItem("investiq-theme", next);
    updateThemeIcon(next);

    if (document.querySelector("#analysisPage")) {
      redrawAllCharts();
    }
  });
}

function updateThemeIcon(theme) {
  const icon = document.getElementById("themeIcon");
  if (icon) icon.textContent = theme === "dark" ? "☾" : "☼";
}

function initHomePage() {
  const form = document.getElementById("homeSearchForm");
  const input = document.getElementById("tickerInput");
  const error = document.getElementById("homeError");
  const suggestions = document.getElementById("suggestions");

  setupQuoteCarousel();

  document.querySelectorAll(".ticker-pill").forEach((button) => {
    button.addEventListener("click", () => {
      input.value = button.dataset.ticker || "";
      goToAnalysis(input.value, error);
    });
  });

  let debounceId = null;
  input.addEventListener("input", () => {
    clearTimeout(debounceId);
    debounceId = setTimeout(async () => {
      const q = input.value.trim();
      suggestions.innerHTML = "";
      if (q.length < 1) return;

      try {
        const data = await apiGet(`/autocomplete?q=${encodeURIComponent(q)}&limit=6`);
        renderSuggestions(data, suggestions, input, error);
      } catch {
        // Autocomplete is helpful but not required.
      }
    }, 220);
  });

  form.addEventListener("submit", (event) => {
    event.preventDefault();
    goToAnalysis(input.value, error);
  });
}

function setupQuoteCarousel() {
  const quoteCard = document.querySelector(".quote-card");
  const quoteEl = document.getElementById("investorQuote");
  const authorEl = document.getElementById("quoteAuthor");
  if (!quoteCard || !quoteEl || !authorEl) return;

  let dotsContainer = document.getElementById("quoteDots");
  if (!dotsContainer) {
    dotsContainer = document.createElement("div");
    dotsContainer.id = "quoteDots";
    dotsContainer.className = "quote-dots";
    quoteCard.appendChild(dotsContainer);
  }

  dotsContainer.innerHTML = quotes
    .map((_, index) => `<button class="quote-dot" type="button" data-quote-index="${index}" aria-label="Show quote ${index + 1}"></button>`)
    .join("");

  dotsContainer.querySelectorAll(".quote-dot").forEach((dot) => {
    dot.addEventListener("click", () => {
      const index = Number(dot.dataset.quoteIndex || 0);
      showQuote(index, true);
      startQuoteRotation();
    });
  });

  showQuote(0, false);
  startQuoteRotation();
}

function startQuoteRotation() {
  clearInterval(quoteRotationTimer);
  quoteRotationTimer = setInterval(() => {
    const nextIndex = (activeQuoteIndex + 1) % quotes.length;
    showQuote(nextIndex, true);
  }, 6500);
}

function showQuote(index, animate = true) {
  const quoteCard = document.querySelector(".quote-card");
  const quoteEl = document.getElementById("investorQuote");
  const authorEl = document.getElementById("quoteAuthor");
  const dots = document.querySelectorAll(".quote-dot");

  if (!quoteCard || !quoteEl || !authorEl) return;

  const applyQuote = () => {
    quoteEl.textContent = `“${quotes[index].text}”`;
    authorEl.textContent = quotes[index].author;
    activeQuoteIndex = index;

    dots.forEach((dot, dotIndex) => {
      dot.classList.toggle("active", dotIndex === index);
    });
  };

  if (!animate) {
    applyQuote();
    return;
  }

  quoteCard.classList.add("is-switching");

  setTimeout(() => {
    applyQuote();
    quoteCard.classList.remove("is-switching");
  }, 260);
}

function renderSuggestions(data, container, input, error) {
  container.innerHTML = "";

  if (!Array.isArray(data) || data.length === 0) return;

  data.forEach((item) => {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "suggestion-button";
    button.textContent = `${item.ticker} · ${item.company_name || "Company"}`;
    button.addEventListener("click", () => {
      input.value = item.ticker;
      goToAnalysis(item.ticker, error);
    });
    container.appendChild(button);
  });
}

function goToAnalysis(rawTicker, errorEl) {
  const ticker = normalizeTicker(rawTicker);
  if (!ticker) {
    showInlineError(errorEl, "Enter a valid ticker symbol.");
    return;
  }

  window.location.href = `analysis.html?ticker=${encodeURIComponent(ticker)}`;
}

function initAnalysisPage() {
  initTabs();

  const params = new URLSearchParams(window.location.search);
  const ticker = normalizeTicker(params.get("ticker") || "AAPL");
  state.ticker = ticker;

  const input = document.getElementById("analysisTickerInput");
  if (input) input.value = ticker;

  const form = document.getElementById("analysisSearchForm");
  form?.addEventListener("submit", (event) => {
    event.preventDefault();
    const nextTicker = normalizeTicker(input.value);
    if (!nextTicker) return;
    window.location.href = `analysis.html?ticker=${encodeURIComponent(nextTicker)}`;
  });

  const backtestForm = document.getElementById("backtestForm");
  backtestForm?.addEventListener("submit", async (event) => {
    event.preventDefault();
    await runBacktest();
  });

  loadAnalysis(ticker);
}

function initTabs() {
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.addEventListener("click", () => {
      const tab = button.dataset.tab;
      document.querySelectorAll(".tab-button").forEach((item) => item.classList.remove("active"));
      document.querySelectorAll(".tab-panel").forEach((item) => item.classList.remove("active"));
      button.classList.add("active");
      document.getElementById(`tab-${tab}`)?.classList.add("active");
      setTimeout(redrawAllCharts, 50);
    });
  });
}

async function loadAnalysis(ticker) {
  setLoading(true);
  hidePageError();

  try {
    const [profile, financials, ratios, peers, valuation] = await Promise.all([
      apiGet(`/company_profile?ticker=${encodeURIComponent(ticker)}`),
      apiGet(`/historical_financials?ticker=${encodeURIComponent(ticker)}&years=8`),
      apiGet(`/financial_ratios?ticker=${encodeURIComponent(ticker)}`),
      apiGet(`/peer_comparison?ticker=${encodeURIComponent(ticker)}&limit=8`),
      apiGet(`/ml_classify?ticker=${encodeURIComponent(ticker)}`),
    ]);

    state.profile = profile;
    state.financials = Array.isArray(financials) ? financials : [];
    state.ratios = ratios;
    state.peers = peers;
    state.valuation = valuation;

    renderHeader(profile);
    renderProfile(profile);
    renderFinancials(state.financials);
    renderRatios(ratios);
    renderPeers(peers);
    renderValuation(valuation);
  } catch (error) {
    showPageError(error.message || "Unable to load analysis.");
  } finally {
    setLoading(false);
  }
}

async function apiGet(path) {
  const response = await fetch(`${API_BASE}${path}`);
  const payload = await response.json();

  if (!response.ok || payload.success === false) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }

  return payload.data;
}

async function apiPost(path, body) {
  const response = await fetch(`${API_BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  const payload = await response.json();

  if (!response.ok || payload.success === false) {
    throw new Error(payload.error || `Request failed with status ${response.status}`);
  }

  return payload.data;
}

function renderHeader(profile) {
  const title = document.getElementById("companyTitle");
  const subtitle = document.getElementById("companySubtitle");

  title.textContent = `${profile.company_name || profile.ticker} (${profile.ticker})`;
  subtitle.textContent = `${profile.sector || "Sector unavailable"} · ${profile.industry || "Industry unavailable"} · Latest year ${profile.year || "N/A"}`;
}

function renderProfile(profile) {
  const grid = document.getElementById("profileGrid");
  if (!grid) return;

  const rows = [
    ["Company", profile.company_name || "N/A"],
    ["Ticker", profile.ticker],
    ["Sector", profile.sector || "N/A"],
    ["Industry", profile.industry || "N/A"],
    ["Latest year", profile.year || "N/A"],
    ["Revenue", formatCurrency(profile.revenue)],
    ["Net income", formatCurrency(profile.net_income)],
    ["Total assets", formatCurrency(profile.total_assets)],
    ["Total equity", formatCurrency(profile.total_equity)],
    ["Market cap", formatCurrency(profile.market_cap)],
    ["P/E", formatMultiple(profile.pe)],
    ["ROE", formatPercent(normalizePercentRatio(profile.roe))],
  ];

  grid.innerHTML = rows.map(([label, value]) => `
    <article class="profile-card">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value))}</strong>
    </article>
  `).join("");
}

function renderFinancials(rows) {
  if (!rows || rows.length === 0) {
    document.getElementById("financialsTable").outerHTML = `<div class="empty-state">No historical financials available.</div>`;
    return;
  }

  const sorted = [...rows].sort((a, b) => Number(a.year) - Number(b.year));
  const years = sorted.map((row) => row.year);

  drawLineChart("revenueChart", "Revenue", years, sorted.map((row) => row.revenue), "Revenue");
  drawLineChart("netIncomeChart", "Net Income", years, sorted.map((row) => row.net_income), "Net Income");
  drawLineChart("assetsChart", "Total Assets", years, sorted.map((row) => row.total_assets), "Total Assets");

  renderTable("financialsTable", [
    { key: "year", label: "Year" },
    { key: "revenue", label: "Revenue", format: formatCurrency },
    { key: "net_income", label: "Net Income", format: formatCurrency },
    { key: "total_assets", label: "Assets", format: formatCurrency },
    { key: "total_equity", label: "Equity", format: formatCurrency },
    { key: "gross_profit", label: "Gross Profit", format: formatCurrency },
    { key: "total_debt", label: "Debt", format: formatCurrency },
  ], sorted);
}

function renderRatios(payload) {
  const scoreCard = document.getElementById("healthScoreCard");
  const grid = document.getElementById("ratioGrid");
  if (!scoreCard || !grid) return;

  scoreCard.innerHTML = `
    <div>
      <p class="eyebrow">Financial health score</p>
      <h3>${escapeHtml(payload.company_name || payload.ticker)}</h3>
      <p>${escapeHtml(payload.summary || "")}</p>
    </div>
    <div class="score-number">${payload.financial_health_score ?? "N/A"}</div>
  `;

  const ratios = payload.ratios || [];
  grid.innerHTML = ratios.map((ratio) => `
    <article class="ratio-card status-${escapeHtml(ratio.status || "grey")}">
      <span class="ratio-label"><span class="status-dot"></span>${escapeHtml(ratio.label)}</span>
      <strong>${escapeHtml(ratio.formatted_value || "N/A")}</strong>
      <span>Sector percentile: ${formatNumber(ratio.sector_percentile)}%</span>
      <p class="ratio-range">Good range: ${escapeHtml(ratio.good_range || "N/A")} · Weak range: ${escapeHtml(ratio.bad_range || "N/A")}</p>
      <p>${escapeHtml(ratio.explanation || "")}</p>
    </article>
  `).join("");
}

function renderPeers(payload) {
  if (!payload) return;

  renderTable("peersTable", [
    { key: "ticker", label: "Ticker" },
    { key: "company_name", label: "Company" },
    { key: "industry", label: "Industry" },
    { key: "market_cap", label: "Market Cap", format: formatCurrency },
    { key: "pe", label: "P/E", format: formatMultiple },
    { key: "pb", label: "P/B", format: formatMultiple },
    { key: "roe", label: "ROE", format: formatPercent },
    { key: "roa", label: "ROA", format: formatPercent },
    { key: "de", label: "D/E", format: formatNumber },
    { key: "current_ratio", label: "Current", format: formatNumber },
    { key: "gross_margin", label: "Gross Margin", format: formatPercent },
  ], payload.peers || []);

  drawPeerBarChart(payload.bar_chart_data);
  drawPeerRadarChart(payload.radar_chart_data);
}

function renderValuation(payload) {
  const card = document.getElementById("valuationCard");
  if (!card) return;

  const probabilities = payload.probabilities || {};
  const positive = payload.top_positive_signals || [];
  const negative = payload.top_negative_signals || [];

  card.innerHTML = `
    <span class="verdict">${escapeHtml(payload.verdict || "N/A")}</span>
    <h3>${escapeHtml(payload.company_name || payload.ticker)}</h3>
    <p>${escapeHtml(payload.explanation || "")}</p>
    <div class="probability-list">
      ${Object.entries(probabilities).map(([label, value]) => `
        <div class="prob-row">
          <span>${escapeHtml(label)}</span>
          <div class="prob-bar"><span style="width: ${Math.max(0, Math.min(100, Number(value || 0) * 100))}%"></span></div>
          <strong>${formatPercent(value)}</strong>
        </div>
      `).join("")}
    </div>
    <h4>Positive signals</h4>
    ${renderSignalList(positive)}
    <h4>Negative signals</h4>
    ${renderSignalList(negative)}
  `;

  drawFeatureImportanceChart(payload.feature_importance || []);
}

function renderSignalList(items) {
  if (!items || items.length === 0) {
    return `<div class="empty-state">No strong signals found.</div>`;
  }

  return `
    <ul class="signal-list">
      ${items.map((item) => `
        <li><strong>${escapeHtml(item.label)} ${escapeHtml(item.formatted_value || "")}</strong> · ${escapeHtml(item.message || "")}</li>
      `).join("")}
    </ul>
  `;
}

async function runBacktest() {
  const error = document.getElementById("backtestError");
  showInlineError(error, "");

  const tickers = document.getElementById("backtestTickers").value
    .split(",")
    .map((item) => normalizeTicker(item))
    .filter(Boolean);

  const startDate = document.getElementById("backtestStart").value;
  const endDate = document.getElementById("backtestEnd").value;
  const initialCapital = Number(document.getElementById("backtestCapital").value || 10000);

  if (tickers.length === 0) {
    showInlineError(error, "Enter at least one ticker.");
    return;
  }

  try {
    setLoading(true);
    const result = await apiPost("/backtest", {
      tickers,
      start_date: startDate,
      end_date: endDate,
      initial_capital: initialCapital,
    });

    renderBacktest(result);
  } catch (err) {
    showInlineError(error, err.message || "Backtest failed.");
  } finally {
    setLoading(false);
  }
}

function renderBacktest(payload) {
  const metrics = payload.metrics || {};
  const summary = payload.summary || {};

  const metricGrid = document.getElementById("backtestMetrics");
  metricGrid.innerHTML = [
    ["Initial Capital", formatFullCurrency(summary.initial_capital)],
    ["Final Value", formatFullCurrency(summary.final_value)],
    ["Total Return", formatPercent(metrics.total_return)],
    ["CAGR", formatPercent(metrics.cagr)],
    ["Max Drawdown", formatPercent(metrics.max_drawdown)],
    ["Win Rate", formatPercent(metrics.win_rate)],
    ["Executed Trades", summary.executed_trades ?? 0],
  ].map(([label, value]) => `
    <article class="metric-card">
      <span>${escapeHtml(label)}</span>
      <strong>${escapeHtml(String(value))}</strong>
    </article>
  `).join("");

  drawPortfolioChart(payload.portfolio_history || []);

  const methodology = document.getElementById("backtestMethodology");
  methodology.innerHTML = `
    <p><strong>Strategy:</strong> ${escapeHtml(payload.methodology?.strategy || "")}</p>
    <p><strong>Sell rule:</strong> ${escapeHtml(payload.methodology?.sell_rule || "")}</p>
    <p><strong>Data alignment:</strong> ${escapeHtml(payload.methodology?.fundamental_alignment || "")}</p>
    <p><strong>Disclaimer:</strong> ${escapeHtml(payload.disclaimer || "")}</p>
  `;

  renderTable("tradesTable", [
    { key: "ticker", label: "Ticker" },
    { key: "status", label: "Status" },
    { key: "reason", label: "Reason" },
    { key: "entry_date", label: "Entry" },
    { key: "exit_date", label: "Exit" },
    { key: "entry_price", label: "Entry Price", format: formatCurrency },
    { key: "exit_price", label: "Exit Price", format: formatCurrency },
    { key: "return_pct", label: "Return", format: formatPercent },
    { key: "profit_loss", label: "P/L", format: formatCurrency },
    { key: "fundamental_year_used", label: "Fund. Year" },
  ], payload.trades || []);
}

function drawLineChart(id, title, x, y, name) {
  if (!window.Plotly) return;

  Plotly.react(id, [{
    x,
    y,
    type: "scatter",
    mode: "lines+markers",
    name,
    line: { width: 3 },
    marker: { size: 7 },
    hovertemplate: "%{x}<br>%{y:$,.0f}<extra></extra>",
  }], layout(title));
}

function drawPeerBarChart(data) {
  if (!window.Plotly || !data) return;

  const labels = data.labels || [];
  const traces = [
    { name: "Target", y: data.target || [] },
    { name: "Peer Avg", y: data.peer_average || [] },
    { name: "Sector Avg", y: data.sector_average || [] },
    { name: "Industry Avg", y: data.industry_average || [] },
  ].map((series) => ({
    x: labels,
    y: series.y,
    name: series.name,
    type: "bar",
    hovertemplate: "%{x}<br>%{y:.2f}<extra></extra>",
  }));

  Plotly.react("peerBarChart", traces, {
    ...layout("Peer Metrics"),
    barmode: "group",
    xaxis: { ...axisStyle(), tickangle: -25 },
  });
}

function drawPeerRadarChart(data) {
  if (!window.Plotly || !data) return;

  const labels = [...(data.labels || [])];
  const close = (arr) => {
    const copy = [...(arr || [])];
    if (copy.length > 0) copy.push(copy[0]);
    return copy;
  };
  const theta = labels.length ? [...labels, labels[0]] : [];

  const traces = [
    { name: "Target", r: data.target },
    { name: "Peer Avg", r: data.peer_average },
    { name: "Sector Avg", r: data.sector_average },
  ].map((series) => ({
    type: "scatterpolar",
    r: close(series.r),
    theta,
    fill: "toself",
    name: series.name,
    hovertemplate: "%{theta}<br>%{r:.1f}/100<extra></extra>",
  }));

  Plotly.react("peerRadarChart", traces, {
    ...layout("Normalized Radar Score"),
    polar: {
      bgcolor: "rgba(0,0,0,0)",
      radialaxis: {
        visible: true,
        range: [0, 100],
        gridcolor: cssVar("--line-strong"),
        tickfont: { color: cssVar("--muted") },
      },
      angularaxis: {
        gridcolor: cssVar("--line"),
        tickfont: { color: cssVar("--muted") },
      },
    },
  });
}

function drawFeatureImportanceChart(items) {
  if (!window.Plotly) return;

  const sorted = [...items].sort((a, b) => Number(a.importance) - Number(b.importance));

  Plotly.react("featureImportanceChart", [{
    x: sorted.map((item) => item.importance),
    y: sorted.map((item) => item.label),
    type: "bar",
    orientation: "h",
    hovertemplate: "%{y}<br>%{x:.3f}<extra></extra>",
  }], {
    ...layout("Feature Importance"),
    height: 410,
    margin: { t: 54, r: 36, b: 38, l: 165 },
    showlegend: false,
    yaxis: {
      ...axisStyle(),
      automargin: true,
      tickfont: { color: cssVar("--muted"), size: 12 },
    },
    xaxis: {
      ...axisStyle(),
      automargin: true,
      tickfont: { color: cssVar("--muted"), size: 12 },
    },
  }, {
    responsive: true,
    displayModeBar: false,
  });
}

function drawPortfolioChart(rows) {
  if (!window.Plotly) return;

  if (!rows || rows.length === 0) {
    document.getElementById("portfolioChart").innerHTML = `<div class="empty-state">Run a backtest to see portfolio history.</div>`;
    return;
  }

  const x = rows.map((row) => row.date);
  const y = rows.map((row) => roundMoney(row.portfolio_value));

  Plotly.react("portfolioChart", [{
    x,
    y,
    type: "scatter",
    mode: "lines",
    line: { width: 3 },
    hovertemplate: "%{x}<br>Portfolio value: $%{y:,.2f}<extra></extra>",
  }], {
    ...layout("Portfolio Value"),
    margin: { t: 56, r: 28, b: 52, l: 90 },
    yaxis: {
      ...axisStyle(),
      tickprefix: "$",
      tickformat: ",.0f",
      automargin: true,
    },
    xaxis: {
      ...axisStyle(),
      automargin: true,
    },
  }, {
    responsive: true,
    displayModeBar: false,
  });
}

function renderTable(tableId, columns, rows) {
  const table = document.getElementById(tableId);
  if (!table) return;

  if (!rows || rows.length === 0) {
    table.innerHTML = `<tbody><tr><td>No data available.</td></tr></tbody>`;
    return;
  }

  table.innerHTML = `
    <thead>
      <tr>${columns.map((col) => `<th>${escapeHtml(col.label)}</th>`).join("")}</tr>
    </thead>
    <tbody>
      ${rows.map((row) => `
        <tr>
          ${columns.map((col) => {
            const value = row[col.key];
            const formatted = col.format ? col.format(value) : (value ?? "N/A");
            return `<td>${escapeHtml(String(formatted))}</td>`;
          }).join("")}
        </tr>
      `).join("")}
    </tbody>
  `;
}

function redrawAllCharts() {
  if (state.financials?.length) renderFinancials(state.financials);
  if (state.peers) renderPeers(state.peers);
  if (state.valuation) drawFeatureImportanceChart(state.valuation.feature_importance || []);
}

function layout(title) {
  return {
    title: {
      text: title,
      font: { color: cssVar("--text"), size: 18 },
      x: 0,
    },
    paper_bgcolor: "rgba(0,0,0,0)",
    plot_bgcolor: "rgba(0,0,0,0)",
    margin: { t: 56, r: 24, b: 52, l: 64 },
    font: { color: cssVar("--text") },
    legend: {
      orientation: "h",
      y: -0.22,
      font: { color: cssVar("--muted") },
    },
    xaxis: axisStyle(),
    yaxis: axisStyle(),
  };
}

function axisStyle() {
  return {
    gridcolor: cssVar("--line"),
    zerolinecolor: cssVar("--line-strong"),
    tickfont: { color: cssVar("--muted") },
  };
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function setLoading(isLoading) {
  const bar = document.getElementById("loadingBar");
  if (bar) bar.hidden = !isLoading;
}

function showPageError(message) {
  const el = document.getElementById("analysisError");
  if (!el) return;
  el.textContent = message;
  el.hidden = false;
}

function hidePageError() {
  const el = document.getElementById("analysisError");
  if (!el) return;
  el.textContent = "";
  el.hidden = true;
}

function showInlineError(el, message) {
  if (!el) return;
  el.textContent = message || "";
}

function normalizeTicker(value) {
  const ticker = String(value || "").trim().toUpperCase();
  if (!/^[A-Z][A-Z0-9.\-]{0,9}$/.test(ticker)) return "";
  return ticker;
}


function normalizePercentRatio(value) {
  if (value === null || value === undefined || value === "") return value;

  const number = Number(value);
  if (!Number.isFinite(number)) return value;

  // Some source values store 36 as 36%, not 0.36.
  if (Math.abs(number) > 1.5) {
    return number / 100;
  }

  return number;
}

function formatCurrency(value) {
  if (value === null || value === undefined || value === "") return "N/A";

  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";

  const abs = Math.abs(number);
  if (abs >= 1_000_000_000_000) return `$${(number / 1_000_000_000_000).toFixed(2)}T`;
  if (abs >= 1_000_000_000) return `$${(number / 1_000_000_000).toFixed(2)}B`;
  if (abs >= 1_000_000) return `$${(number / 1_000_000).toFixed(2)}M`;
  return `$${number.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
}

function formatPercent(value) {
  if (value === null || value === undefined || value === "") return "N/A";

  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  return `${(number * 100).toFixed(2)}%`;
}

function formatMultiple(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  return `${number.toFixed(2)}x`;
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  return number.toLocaleString(undefined, { maximumFractionDigits: 2 });
}

function roundMoney(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) return null;
  return Math.round(number * 100) / 100;
}

function formatFullCurrency(value) {
  if (value === null || value === undefined || value === "") return "N/A";

  const number = Number(value);
  if (!Number.isFinite(number)) return "N/A";
  return `$${number.toLocaleString(undefined, {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}
