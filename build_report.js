'use strict';
const fs = require('fs');
const path = require('path');
const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell, ImageRun,
  Header, Footer, AlignmentType, LevelFormat, BorderStyle, WidthType, ShadingType,
  HeadingLevel, PageNumber, PageBreak, TableOfContents
} = require('/opt/homebrew/lib/node_modules/docx');

// ── Paths ─────────────────────────────────────────────────────────────────────
const REPO = '/Users/nigelli/Desktop/Columbia MAFN/26Spring/MATH5320/Project/MATH5320';
const THUMB = path.join(REPO, 'docs/screenshots/thumb');
const OUT   = path.join(REPO, 'FINAL_REPORT.docx');

// ── Helpers ───────────────────────────────────────────────────────────────────
const W = 9360; // content width DXA (US Letter 8.5" - 2" margins)
const GRAY  = 'CCCCCC';
const BLUE  = 'D5E8F0';
const DBLUE = '2E75B6';
const border = { style: BorderStyle.SINGLE, size: 1, color: GRAY };
const borders = { top: border, bottom: border, left: border, right: border };

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: DBLUE, space: 4 } },
    children: [new TextRun({ text, font: 'Arial', size: 32, bold: true, color: '1F3864' })]
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, font: 'Arial', size: 26, bold: true, color: '2E75B6' })]
  });
}
function h3(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_3,
    spacing: { before: 160, after: 80 },
    children: [new TextRun({ text, font: 'Arial', size: 24, bold: true, color: '404040' })]
  });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { before: 80, after: 80 },
    children: [new TextRun({ text, font: 'Arial', size: 22, ...opts })]
  });
}
function code(text) {
  return new Paragraph({
    spacing: { before: 40, after: 40 },
    indent: { left: 720 },
    shading: { fill: 'F2F2F2', type: ShadingType.CLEAR },
    children: [new TextRun({ text, font: 'Courier New', size: 18, color: '333333' })]
  });
}
function codeBlock(lines) {
  return lines.split('\n').map(line => code(line));
}
function blank(n = 1) {
  return Array.from({ length: n }, () => new Paragraph({ children: [] }));
}
function pageBreak() {
  return new Paragraph({ children: [new PageBreak()] });
}

// ── Table builder ─────────────────────────────────────────────────────────────
function makeTable(headers, rows, colWidths) {
  const total = colWidths.reduce((a, b) => a + b, 0);
  function cell(text, isHeader = false, colW) {
    return new TableCell({
      borders,
      width: { size: colW, type: WidthType.DXA },
      shading: { fill: isHeader ? BLUE : 'FFFFFF', type: ShadingType.CLEAR },
      margins: { top: 80, bottom: 80, left: 120, right: 120 },
      children: [new Paragraph({
        children: [new TextRun({ text: String(text), font: 'Arial', size: 18, bold: isHeader })]
      })]
    });
  }
  const headerRow = new TableRow({
    tableHeader: true,
    children: headers.map((h, i) => cell(h, true, colWidths[i]))
  });
  const dataRows = rows.map(row =>
    new TableRow({ children: row.map((c, i) => cell(c, false, colWidths[i])) })
  );
  return new Table({
    width: { size: total, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...dataRows]
  });
}

// ── Screenshot embed ──────────────────────────────────────────────────────────
function screenshot(filename, caption) {
  const imgPath = path.join(THUMB, filename);
  if (!fs.existsSync(imgPath)) return [p(`[Screenshot: ${filename}]`, { italics: true })];
  const imgData = fs.readFileSync(imgPath);
  return [
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 120, after: 60 },
      children: [new ImageRun({
        type: 'png',
        data: imgData,
        transformation: { width: 600, height: 375 },
        altText: { title: caption, description: caption, name: caption }
      })]
    }),
    new Paragraph({
      alignment: AlignmentType.CENTER,
      spacing: { before: 0, after: 160 },
      children: [new TextRun({ text: caption, font: 'Arial', size: 18, italics: true, color: '666666' })]
    })
  ];
}

// ══════════════════════════════════════════════════════════════════════════════
// DOCUMENT CONTENT
// ══════════════════════════════════════════════════════════════════════════════

const children = [

  // ── Cover page ─────────────────────────────────────────────────────────────
  ...blank(4),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 240 },
    children: [new TextRun({ text: 'MATH GR 5320', font: 'Arial', size: 28, bold: true, color: DBLUE })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 240 },
    children: [new TextRun({ text: 'Portfolio Risk Management System', font: 'Arial', size: 40, bold: true, color: '1F3864' })]
  }),
  new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 0, after: 480 },
    border: { bottom: { style: BorderStyle.SINGLE, size: 6, color: DBLUE, space: 4 } },
    children: [new TextRun({ text: 'Final Project Report', font: 'Arial', size: 28, color: '404040' })]
  }),
  ...blank(2),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'Authors', font: 'Arial', size: 22, bold: true, color: '606060' })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'Nigel Li  ·  Michael Adegbite  ·  Stella', font: 'Arial', size: 24, bold: true })] }),
  ...blank(1),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'Columbia University', font: 'Arial', size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'MATH GR 5320 — Financial Risk Management', font: 'Arial', size: 22 })] }),
  new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ text: 'May 10, 2026', font: 'Arial', size: 22 })] }),
  pageBreak(),

  // ── Executive Summary ──────────────────────────────────────────────────────
  h1('Executive Summary'),
  p('We designed and implemented a complete portfolio risk-measurement system for MATH GR 5320. The system accepts mixed portfolios of equities and European options, loads historical price data from Yahoo Finance or CSV files, and computes Value at Risk (VaR) and Expected Shortfall (ES) under three methodologies: historical simulation, parametric delta-normal, and Monte Carlo. Walk-forward VaR backtesting with Kupiec unconditional coverage, Christoffersen independence, and conditional-coverage tests provides model validation diagnostics.'),
  p('We extended the required engine with formula-sheet modules for exact lognormal VaR/ES, reduced-form credit risk, the Merton structural default model, CDS pricing, CVA with counterparty mitigation, and illustrative regulatory capital with DFAST-style stress-test projection. The entire system is exposed through an eight-tab Streamlit application and through reusable Python modules.'),
  ...blank(1),
  h2('Live Demonstration Results'),
  p('Portfolio: AAPL (100 shares) + MSFT (50 shares) + AAPL call (10 contracts, K=$200), Yahoo Finance data 2021-05-11 to 2026-05-08 (1,255 trading days), one-day 99% VaR:'),
  makeTable(
    ['Method', 'VaR ($)', 'ES ($)', 'VaR / Portfolio'],
    [
      ['Historical Simulation', '$11,095.68', '$11,247.45', '7.61%'],
      ['Parametric Delta-Normal', '$1,299.44', '$1,306.07', '0.89%'],
      ['Monte Carlo (10,000 paths)', '$10,488.89', '$10,439.94', '7.19%'],
    ],
    [3120, 2080, 2080, 2080]
  ),
  ...blank(1),
  p('Backtesting over 1,001 hold-out days produced exactly 10 VaR exceptions (1.00% observed vs. 1.00% expected). The Kupiec LR statistic was 0.0000 (p-value 0.9975), confirming that H₀ of correct unconditional coverage is not rejected.'),
  p('Our test suite passed 569 tests with 92% statement coverage across src/ (no network required). Two additional integration scripts exercise live Yahoo Finance data paths.'),
  pageBreak(),

  // ── Requirement Coverage ───────────────────────────────────────────────────
  h1('1  Requirement Coverage Matrix'),
  makeTable(
    ['Project Requirement', 'Source File(s)', 'Test Evidence'],
    [
      ['Portfolio of stocks and options', 'src/schemas.py, src/ui/portfolio_editor.py', 'test_backend.py, test_ui_panels.py'],
      ['Historical data + parameter input', 'src/data/market_data.py', 'test_market_data.py'],
      ['Historical VaR', 'src/risk/historical.py', 'test_backend.py, test_homework_cases.py'],
      ['Parametric VaR', 'src/risk/parametric.py', 'test_backend.py, test_es_confidence_split.py'],
      ['Monte Carlo VaR', 'src/risk/monte_carlo.py', 'test_backend.py, test_coverage_gaps.py'],
      ['Historical ES', 'src/risk/historical.py', 'test_backend.py, test_es_confidence_split.py'],
      ['Monte Carlo ES', 'src/risk/monte_carlo.py', 'test_backend.py, test_es_confidence_split.py'],
      ['VaR Backtesting', 'src/risk/backtest.py', 'test_backend.py, test_backtest_extensions.py'],
      ['European option pricing', 'src/pricing/black_scholes.py', 'test_backend.py, test_homework_cases.py'],
      ['Model documentation', 'This report', '—'],
      ['Software design documentation', 'README.md, src/ structure', 'test_ui_panels.py'],
      ['Test plan', 'tests/ directory', '569 tests, 92% coverage'],
      ['Software', 'app.py, src/, notebooks/', 'Full test suite + integration'],
      ['Test results', '§8 of this report', 'pytest + coverage run'],
    ],
    [3600, 3160, 2600]
  ),
  pageBreak(),

  // ── Introduction ───────────────────────────────────────────────────────────
  h1('2  Introduction and Scope'),
  h2('2.1  System Name'),
  p('MATH5320 Portfolio Risk Management System'),
  h2('2.2  Business Purpose'),
  p('The system is an academic risk-calculation platform for MATH GR 5320. Its purpose is to allow students and analysts to value portfolios of stocks and European options, compare risk methodologies side-by-side, validate formula implementations against course-derived fixtures, and document modelling assumptions and limitations in a validation-oriented format modelled on the Stein (2014) municipal-bond validation report structure.'),
  h2('2.3  Intended Users'),
  p('Students and instructors using the Streamlit application (app.py), quantitative analysts importing from the src/ package, and researchers using the Jupyter notebooks under notebooks/.'),
  h2('2.4  Intended Use'),
  p('Define portfolios of stocks and European options; load historical data from CSV or Yahoo Finance; compute and compare VaR/ES under historical, parametric, and Monte Carlo methods; run walk-forward VaR backtesting with Kupiec, Christoffersen, and conditional-coverage tests; validate course-formula modules; produce model documentation for course deliverables.'),
  h2('2.5  Non-Intended Use'),
  p('The system is NOT intended for: production trading or risk management; regulatory capital filing or CCAR/DFAST submission; production XVA or enterprise-wide risk aggregation; pricing exotics, American options, or volatility-surface-sensitive products.'),
  h2('2.6  Scope'),
  makeTable(
    ['Area', 'In Scope', 'Out of Scope'],
    [
      ['Instruments', 'Stocks, European calls/puts', 'American options, exotics'],
      ['VaR methods', 'Historical, parametric, Monte Carlo', 'EVT, filtered historical, copula'],
      ['ES methods', 'Historical, parametric, MC, exact GBM', 'Full regulatory ES framework'],
      ['Pricing', 'Black-Scholes (constant vol)', 'Local/stochastic vol, early exercise'],
      ['Credit', 'Hazard, Merton, CDS, CVA modules', 'Production credit portfolio model'],
      ['Regulation', 'RWA, capital ratio, illustrative DFAST', 'Official Fed CCAR/DFAST model'],
    ],
    [2000, 3680, 3680]
  ),
  pageBreak(),

  // ── Screenshots ────────────────────────────────────────────────────────────
  h1('3  Application Screenshots'),
  p('The following screenshots were captured from the live application running at localhost:8502 against five years of Yahoo Finance data (AAPL + MSFT, 2021-05-11 to 2026-05-08, 1,255 trading days).'),
  ...blank(1),
  h2('3.1  Tab 1 — Portfolio Input'),
  p('Stock positions (AAPL × 100, MSFT × 50) and one option position (10 AAPL call contracts, strike $200) are entered. The editor validates every field and shows a live position summary.'),
  ...screenshot('01_portfolio_input.png', 'Figure 1: Portfolio Input — AAPL, MSFT stocks with AAPL call option'),
  h2('3.2  Tab 2 — Market Data'),
  p('Data downloaded from Yahoo Finance: 1,255 rows × 2 tickers (2021-05-11 to 2026-05-08). A local parquet cache avoids repeated downloads. Most recent prices: AAPL $270.71, MSFT $424.82.'),
  ...screenshot('02_market_data.png', 'Figure 2: Market Data — 1,255 rows, 2 tickers loaded from Yahoo Finance'),
  h2('3.3  Tab 3 — Risk Settings'),
  p('Lookback window 252 days, horizon 1 day, VaR confidence 99%, ES confidence 97.5%, rolling-window estimator, 10,000 Monte Carlo paths.'),
  ...screenshot('03_risk_settings.png', 'Figure 3: Risk Settings — 252-day lookback, 1-day horizon, 99% VaR confidence'),
  h2('3.4  Tab 4 — Run Analysis'),
  p('One click runs all three VaR/ES engines simultaneously. Portfolio value $145,864.49. Historical VaR $11,095.68, Parametric VaR $1,299.44, Monte Carlo VaR $10,488.89.'),
  ...screenshot('04_run_analysis.png', 'Figure 4: Run Analysis — VaR/ES comparison table and bar chart'),
  h2('3.5  Tab 5 — Backtesting'),
  p('Walk-forward backtesting over 1,001 days: 10 exceptions, 1.00% observed rate (expected 1.00%). Kupiec LR = 0.0000, p = 0.9975 — H₀ not rejected.'),
  ...screenshot('05_backtesting.png', 'Figure 5: Backtesting — exception chart, Kupiec statistics, 1.00% observed rate'),
  h2('3.6  Tab 6 — Credit Risk'),
  p('Reduced-form hazard panel: λ = 3%, R = 40%, LGD = 60%, CDS approximation = 180.0 bps (§14 landmark value). Merton structural model below for firm-value analysis.'),
  ...screenshot('06_credit_risk.png', 'Figure 6: Credit Risk — hazard/survival table, 180 bps CDS spread confirmed'),
  h2('3.7  Tab 7 — CDS / CVA'),
  p('Par-spread curve under constant hazard: approx 180.0 bps, full-formula par spread at 10-year tenor 180.7 bps. CVA section builds exposure profile from MC engine.'),
  ...screenshot('07_cds_cva.png', 'Figure 7: CDS/CVA — par spread curve and CVA computation'),
  h2('3.8  Tab 8 — Capital & Stress'),
  p('RWA computed from portfolio exposures with unit Basel risk weights. Equity capital = $11,669.16 (8% of portfolio value). Capital ratio = 22.84% — PASS ✅.'),
  ...screenshot('08_capital_stress.png', 'Figure 8: Capital & Stress — RWA, capital ratio 22.84%, PASS'),
  pageBreak(),

  // ── Product Description ────────────────────────────────────────────────────
  h1('4  Product / System Description'),
  h2('4.1  User Workflow'),
  makeTable(
    ['Tab', 'Name', 'Purpose'],
    [
      ['1', 'Portfolio Input', 'Enter stock and option positions'],
      ['2', 'Market Data', 'Load CSV or download from Yahoo Finance'],
      ['3', 'Risk Settings', 'Configure lookback, horizon, confidence, estimator, MC paths'],
      ['4', 'Run Analysis', 'Execute all VaR/ES engines; view tables and charts'],
      ['5', 'Backtesting', 'Walk-forward backtest with exception diagnostics'],
      ['6', 'Credit Risk', 'Reduced-form hazard and Merton structural model'],
      ['7', 'CDS / CVA', 'CDS par-spread curve and CVA / mitigated CVA'],
      ['8', 'Capital & Stress', 'RWA, capital ratio, DFAST stress scenarios'],
    ],
    [800, 2400, 6160]
  ),
  ...blank(1),
  h2('4.2  Input Schema'),
  p('Three dataclasses defined in src/schemas.py:'),
  ...codeBlock('StockPosition:  ticker, quantity (positive=long, negative=short)\nOptionPosition: ticker, underlying_ticker, option_type, quantity,\n                strike, maturity_date, volatility, risk_free_rate,\n                dividend_yield=0.0, contract_multiplier=100.0\nPortfolio:      stocks: list[StockPosition], options: list[OptionPosition]'),
  h2('4.3  Market Data'),
  p('Two loading paths: (1) CSV upload via load_price_history_csv — expects a date column and numeric price columns; (2) Yahoo Finance via download_adjusted_close_cached — downloads adjusted close with exponential-backoff retry and local parquet cache keyed on (sorted(tickers), start, end). A fetch_risk_free_rate(asof) helper pulls the 10-year Treasury yield (^TNX).'),
  h2('4.4  Outputs'),
  makeTable(
    ['Output', 'Format', 'Delivery'],
    [
      ['VaR and ES per method', 'Metric tiles + comparison table', 'Streamlit UI'],
      ['Loss distributions', 'Plotly histogram with VaR/ES lines', 'Streamlit UI'],
      ['Return correlation matrix', 'Plotly heatmap', 'Streamlit UI'],
      ['Backtest chart', 'Plotly time series with exceptions marked', 'Streamlit UI'],
      ['Kupiec / Christoffersen / CC statistics', 'Formatted table', 'Streamlit UI'],
      ['Risk results', 'JSON download button', 'File download'],
      ['Backtest results', 'CSV download button', 'File download'],
      ['Credit hazard table', 'Styled DataFrame', 'Streamlit UI'],
      ['CDS par-spread curve', 'Plotly line chart', 'Streamlit UI'],
      ['DFAST 9-quarter capital path', 'Plotly multi-line chart', 'Streamlit UI'],
    ],
    [3000, 3360, 3000]
  ),
  pageBreak(),

  // ── Model Description ──────────────────────────────────────────────────────
  h1('5  Model Description'),
  h2('5.1  Overview'),
  p('The core market-risk engine consists of: (1) portfolio valuation under Black-Scholes, (2) log-return computation and estimation, (3) historical simulation VaR/ES, (4) parametric delta-normal VaR/ES, (5) Monte Carlo VaR/ES, and (6) walk-forward VaR backtesting. Formula-sheet extensions add exact GBM VaR/ES, credit risk, CDS, CVA, and regulatory capital.'),
  h2('5.2  Return and Scenario Construction'),
  p('We work with daily log-returns throughout: r_t = log(S_t / S_{t-1}). For an h-day horizon we aggregate h consecutive log-returns. Horizon scaling: μ_h = h·μ, Σ_h = h·Σ.'),
  h2('5.3  Black-Scholes Option Pricing'),
  ...codeBlock('d₁ = [log(S/K) + (r − q + ½σ²)T] / (σ√T),   d₂ = d₁ − σ√T\nCall = S·e^{−qT}·N(d₁) − K·e^{−rT}·N(d₂)\nPut  = K·e^{−rT}·N(−d₂) − S·e^{−qT}·N(−d₁)\nΔ_call = e^{−qT}·N(d₁),   Δ_put = e^{−qT}·(N(d₁) − 1)'),
  p('Key limitation: volatility σ is fixed at the user-supplied value. We do not shock the volatility surface when repricing options under stressed spots, meaning vega risk is not captured.'),
  h2('5.4  Historical VaR and ES'),
  p('Implemented in src/risk/historical.py. The algorithm shocks all underlyings with S_shocked = S₀·exp(R) for each historical scenario R in the lookback window, reprices the full portfolio (including options), and computes VaR and ES from the empirical loss distribution.'),
  ...codeBlock('VaR_α = empirical α-quantile of {V₀ − V_sim}\nES_α  = E[loss | loss > ES threshold]'),
  p('Live result (252-day lookback, 1-day horizon, 99%): VaR = $11,095.68 (7.61%), ES = $11,247.45.'),
  h2('5.5  Parametric Delta-Normal VaR and ES'),
  p('Implemented in src/risk/parametric.py. We build a dollar-exposure vector x from stock positions and option deltas, estimate μ and Σ, scale to horizon, then:'),
  ...codeBlock('m = x⊤μ_h,   s² = x⊤Σ_h x\nVaR_α = −m + s·Φ⁻¹(α)\nES_α  = −m + s·φ(z_α)/(1 − α_ES)'),
  p('Live result: VaR = $1,299.44 (0.89%), ES = $1,306.07. The large discrepancy vs. historical VaR illustrates the first-order approximation limitation for nonlinear option payoffs.'),
  h2('5.6  Monte Carlo VaR and ES'),
  p('Implemented in src/risk/monte_carlo.py. We simulate R_h ~ N(μ_h, Σ_h) for 10,000 paths, shock S_sim = S₀·exp(R_h), reprice the portfolio, and compute VaR/ES from the simulated loss distribution.'),
  p('Live result: VaR = $10,488.89 (7.19%), ES = $10,439.94. Default seed = 42 for reproducibility.'),
  h2('5.7  Estimation Methods'),
  p('Rolling window: μ̂ = sample mean, Σ̂ = sample covariance over lookback_days. EWMA: λ = (N−1)/(N+1) — exactly the convention from the course formula sheet. Rolling window is the default for interpretability; EWMA is available for faster regime adaptation.'),
  h2('5.8  VaR Backtesting'),
  p('Implemented in src/risk/backtest.py. Walk-forward procedure: fit the model on data up to t, forecast VaR, compute realized loss t to t+1, flag exception I_t = 1{loss_t > VaR_t}.'),
  p('Kupiec unconditional coverage: LR_uc = −2·[log L₀ − log L₁] ~ χ²(1). Christoffersen independence: LR_ind ~ χ²(1). Conditional coverage: LR_cc = LR_uc + LR_ind ~ χ²(2).'),
  ...blank(1),
  h3('Live Backtest Result (Historical, 252-day lookback, 1-day horizon, 99%, 1,001 observations)'),
  makeTable(
    ['Metric', 'Value'],
    [
      ['Observations', '1,001'],
      ['Exceptions', '10'],
      ['Observed exception rate', '1.00%'],
      ['Expected exception rate', '1.00%'],
      ['Kupiec LR statistic', '0.0000'],
      ['Kupiec p-value', '0.9975'],
      ['Reject H₀ at 5%?', 'No'],
      ['Basel zone', 'Yellow (10 exceptions)'],
    ],
    [4680, 4680]
  ),
  ...blank(1),
  h2('5.9  Formula-Sheet Extensions'),
  makeTable(
    ['Module', 'Purpose', 'Key Formula'],
    [
      ['src/risk/lognormal.py', 'Exact GBM VaR/ES', 'VaR = V₀·[1 − exp(m_h + s_h·z_{1−p})]'],
      ['src/credit/hazard.py', 'Reduced-form credit', 'S(t) = exp(−λt), spread ≈ (1−R)·λ'],
      ['src/credit/merton.py', 'Structural default', 'PD = N(−d₂), d₂ = [log(V₀/B) + ...]/σ√T'],
      ['src/credit/cds.py', 'CDS par spread', 'LGD·∑q_i·DF_i / ∑Δt_i·S(t_i)·DF_i'],
      ['src/credit/cva.py', 'CVA + mitigated CVA', 'CVA = (1−R)·∑Ē_i·p̄_i'],
      ['src/risk/regulatory.py', 'RWA + DFAST capital path', 'RWA = ∑|exp_i|·w_i, ratio = Equity/RWA'],
    ],
    [2800, 3160, 3400]
  ),
  pageBreak(),

  // ── Software Design ────────────────────────────────────────────────────────
  h1('6  Software Design and Implementation'),
  h2('6.1  Architecture'),
  p('We designed the system in distinct layers so that business logic is completely separated from the UI and can be tested independently:'),
  ...codeBlock('UI Layer:      app.py + src/ui/*  (Streamlit only — no math)\n                   |\nService Layer: RiskEngineService · credit_service · regulatory_service\n                   |\nCore Risk:     historical · parametric · monte_carlo · backtest · estimators\nExtensions:    credit/* · risk/lognormal.py · risk/regulatory.py\n                   |\nPricing/Data:  black_scholes.py · portfolio.py · market_data.py\n                   |\nSchemas:       StockPosition · OptionPosition · Portfolio'),
  h2('6.2  Key Design Decisions'),
  p('1. Thin UI — All quantitative logic sits below the Streamlit layer. UI panels call service objects only.'),
  p('2. Pure functions — Risk modules are stateless functions of their arguments, making unit testing simple.'),
  p('3. Service objects — RiskEngineService and *_service.py provide single orchestration entry points.'),
  p('4. Phased development — Notebooks 01–05 cover the required market-risk engine; notebooks 06–11 cover formula-sheet extensions.'),
  p('5. EWMA parameterisation — λ = (N-1)/(N+1) exactly as defined in the course formula sheet.'),
  h2('6.3  Module Map'),
  makeTable(
    ['Module', 'Role'],
    [
      ['src/schemas.py', 'Dataclass definitions: StockPosition, OptionPosition, Portfolio'],
      ['src/data/market_data.py', 'CSV loader, Yahoo Finance, cached download, risk-free rate helper'],
      ['src/pricing/black_scholes.py', 'Option pricing (call/put) and delta computation'],
      ['src/portfolio/portfolio.py', 'Portfolio value and dollar-exposure aggregation'],
      ['src/risk/estimators.py', 'Rolling-window and EWMA mean/covariance estimators'],
      ['src/risk/historical.py', 'Historical simulation VaR and ES'],
      ['src/risk/parametric.py', 'Delta-normal VaR and ES'],
      ['src/risk/monte_carlo.py', 'Monte Carlo VaR and ES (10,000 paths default)'],
      ['src/risk/backtest.py', 'Walk-forward backtest, Kupiec, Christoffersen, CC, Basel light'],
      ['src/risk/lognormal.py', 'Exact GBM/lognormal VaR and ES (long and short)'],
      ['src/risk/regulatory.py', 'RWA, capital ratio, DFAST stress, balance-sheet helpers'],
      ['src/credit/hazard.py', 'Reduced-form: survival, default density, risky ZCB, credit spread'],
      ['src/credit/merton.py', 'Structural default, Merton timing defect modelled explicitly'],
      ['src/credit/cds.py', 'CDS par spread (constant and piecewise hazard)'],
      ['src/credit/cva.py', 'CVA and discounted CVA'],
      ['src/credit/mitigation.py', 'Netting and CSA collateral mitigation'],
      ['src/services/risk_engine_service.py', 'Orchestrates full market-risk run_all()'],
      ['src/services/credit_service.py', 'Orchestrates credit summaries'],
      ['src/services/regulatory_service.py', 'Orchestrates RWA and DFAST capital path'],
    ],
    [4680, 4680]
  ),
  pageBreak(),

  // ── Validation ─────────────────────────────────────────────────────────────
  h1('7  Validation Methodology and Scope'),
  h2('7.1  Validation Types'),
  makeTable(
    ['Validation Type', 'Files', 'Purpose'],
    [
      ['Analytical goldens', 'test_backend.py, test_lognormal.py', 'Formula correctness vs. known exact values'],
      ['Homework fixtures', 'test_homework_cases.py', 'Alignment with course HW solutions (1% tol.)'],
      ['Course validation sheet', 'test_course_validation.py', 'Formula-sheet section coverage'],
      ['Edge cases', 'test_config_and_validation.py, test_strict_numerics.py', 'Validation logic, numerical boundaries'],
      ['Service integration', 'test_backend.py (run_all), integration_test.py', 'End-to-end workflow'],
      ['UI smoke tests', 'test_ui_panels.py, test_charts.py', 'App renders and responds correctly'],
      ['Coverage-gap regression', 'test_coverage_gaps.py, test_es_confidence_split.py', 'Branch coverage and numerical discipline'],
      ['Backtest diagnostics', 'test_backtest_extensions.py', 'Kupiec, Christoffersen, CC, Basel, severity'],
      ['Credit modules', 'test_credit.py, test_merton_timing.py, test_cva_mitigants.py', 'Hazard, Merton timing defect, CVA, mitigation'],
      ['Regulatory', 'test_regulatory.py, test_dfast_pathing.py, test_balance_sheet.py', 'RWA, DFAST path, solvency'],
    ],
    [2800, 3160, 3400]
  ),
  ...blank(1),
  h2('7.2  Tolerances'),
  makeTable(
    ['Tolerance Type', 'Value', 'Applied To'],
    [
      ['Homework / course fixtures', '1% relative', 'All HW-derived test fixtures'],
      ['Analytical formulas', 'Machine precision', 'Deterministic formula outputs'],
      ['Structural assertions', 'Sign / monotonicity', 'ES ≥ VaR, short VaR > long VaR'],
      ['Monte Carlo', 'Up to 5% relative', 'Stochastic simulation outputs'],
      ['Live integration', 'Positive / finite', 'Yahoo Finance end-to-end check'],
    ],
    [3120, 2080, 4160]
  ),
  pageBreak(),

  // ── Validation Results ─────────────────────────────────────────────────────
  h1('8  Validation Results'),
  h2('8.1  Unit and Integration Test Results'),
  p('Run command (no network required):'),
  ...codeBlock('python -m pytest tests/ --ignore=tests/integration_test.py\n  --ignore=tests/integration_test_formula_sheet.py -q'),
  p('Observed result:'),
  ...codeBlock('569 passed in 13.40s'),
  p('Coverage run observed result: 569 passed, 92% statement coverage across src/.'),
  ...blank(1),
  h2('8.2  Homework Validation (23/23 PASS)'),
  p('Notebook 11_end_to_end_demo.ipynb validates all course homework cases at 1% tolerance:'),
  makeTable(
    ['HW', 'Case', 'Expected', 'Actual', 'Status'],
    [
      ['HW2', 'Call price (§5 BS)', '10.450', '10.450', 'PASS'],
      ['HW3', 'Historical VaR 99% 1-stock', '0.0294', '0.0293', 'PASS'],
      ['HW4', '2-stock Historical VaR', '0.00931', '0.00931', 'PASS'],
      ['HW5', 'd₂ = 0 exact', '0.0000', '0.0000', 'PASS'],
      ['HW6', 'EWMA λ = 0.941', '0.9410', '0.9410', 'PASS'],
      ['HW7', 'Lognormal VaR long', '0.2831', '0.2831', 'PASS'],
      ['HW8', 'Hazard survival 1yr', '0.9704', '0.9704', 'PASS'],
      ['HW9', 'Merton Q-PD', '0.0456', '0.0456', 'PASS'],
      ['HW10', 'CDS par spread', '180.0 bps', '179.7 bps', 'PASS'],
      ['HW11', 'CVA (gross)', '0.02841', '0.02841', 'PASS'],
      ['...', '(13 additional cases)', '—', '—', 'All PASS'],
    ],
    [800, 3200, 1800, 1800, 1760]
  ),
  ...blank(1),
  h2('8.3  Formula-Sheet Sanity Values'),
  makeTable(
    ['§14 Landmark', 'Target', 'Observed', 'Status'],
    [
      ['CDS spread (λ=3%, R=40%)', '≈ 180 bps', '180.0 bps', 'PASS'],
      ['Merton P-PD > Q-PD (μ < r)', 'Structural', 'Confirmed', 'PASS'],
      ['Short VaR > Long VaR', 'Structural', 'Confirmed', 'PASS'],
      ['CVA > 0', 'Structural', 'Confirmed', 'PASS'],
      ['Capital ratio > 8% (PASS)', 'Structural', '22.84%', 'PASS'],
    ],
    [3000, 2120, 2120, 2120]
  ),
  pageBreak(),

  // ── Limitations ────────────────────────────────────────────────────────────
  h1('9  Limitations'),
  makeTable(
    ['Limitation', 'Description', 'Impact', 'Mitigation'],
    [
      ['Static option volatility', 'σ fixed; surface not shocked', 'Vega risk not captured in VaR', 'Document; use implied vol from market data'],
      ['Delta-normal approximation', 'First-order exposure; no gamma', 'Understates risk for nonlinear portfolios', 'Use historical or MC for option-heavy books'],
      ['Multivariate normal MC', 'Gaussian simulation', 'Fat-tail risk understated', 'Use historical simulation as primary method'],
      ['Historical scenario assumption', 'Past shocks may not repeat', 'Slow reaction to regime change', 'Use EWMA estimator for faster adaptation'],
      ['Estimation window sensitivity', '252-day default', 'May be too long in fast markets', 'Allow EWMA; consider regime filtering'],
      ['European options only', 'No early exercise', 'American options out of scope', 'Add Longstaff-Schwartz MC in future'],
      ['Constant hazard in credit', 'Piecewise-constant λ', 'No stochastic intensity', 'Add Hull-White or CIR extension'],
      ['Merton maturity default only', 'Default only at T', 'No first-passage default', 'Add Black-Cox barrier model'],
      ['Illustrative DFAST', 'Not official Fed numbers', 'Academic use only', 'Label clearly in UI and report'],
    ],
    [2400, 2400, 2400, 2160]
  ),
  pageBreak(),

  // ── Recommendations ────────────────────────────────────────────────────────
  h1('10  Recommendations'),
  p('1. Shock the volatility surface — Replace static σ with a term-structure shocked alongside spot, capturing vega risk.'),
  p('2. Use implied volatility — Pull ATM implied vol from market data instead of historical volatility for option repricing.'),
  p('3. Add fat-tail simulation — Implement t-distributed or GARCH-filtered scenario generation for the MC engine.'),
  p('4. First-passage Merton — Extend merton.py with the Black-Cox barrier model to allow default before maturity T.'),
  p('5. Full ISDA CVA — Implement a proper EPE profile from full MC simulation, consistent with ISDA/Basel CVA charge requirements.'),
  p('6. American options — Add Longstaff-Schwartz Monte Carlo to support early-exercise instruments.'),
  p('7. Increase test coverage — The current 92% leaves some branches untested; target 95%+ for regulatory-relevant modules.'),
  pageBreak(),

  // ── Bibliography ───────────────────────────────────────────────────────────
  h1('11  Bibliography'),
  p('Black, F. and Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy, 81(3), 637–654.'),
  p('Christoffersen, P. (1998). Evaluating Interval Forecasts. International Economic Review, 39(4), 841–862.'),
  p('Kupiec, P. (1995). Techniques for Verifying the Accuracy of Risk Measurement Models. Journal of Derivatives, 3(2), 73–84.'),
  p('McNeil, A., Frey, R., and Embrechts, P. (2015). Quantitative Risk Management. Princeton University Press.'),
  p('Merton, R. (1974). On the Pricing of Corporate Debt: The Risk Structure of Interest Rates. Journal of Finance, 29(2), 449–470.'),
  p('Stein, H. J. (2014). Model Validation for Municipal Bonds. Bloomberg Portfolio Risk Analytics. [Project report template]'),
  p('Columbia University MATH GR 5320 Formula Sheet (2025–2026). [Internal course material]'),
  pageBreak(),

  // ── Appendix ───────────────────────────────────────────────────────────────
  h1('12  Appendices'),
  h2('Appendix A  Installation and Execution'),
  ...codeBlock('# Create environment\nconda create -n math5320 python=3.10\nconda activate math5320\npip install -r requirements.txt\n\n# Run the Streamlit application\nstreamlit run app.py\n\n# Run unit tests (no network)\npython -m pytest tests/ \\\n  --ignore=tests/integration_test.py \\\n  --ignore=tests/integration_test_formula_sheet.py -v\n\n# Run with coverage\npython -m pytest tests/ --cov=src --cov-report=term-missing \\\n  --ignore=tests/integration_test.py \\\n  --ignore=tests/integration_test_formula_sheet.py\n\n# Network integration tests\npython tests/integration_test.py\npython tests/integration_test_formula_sheet.py'),
  h2('Appendix B  Deliverable Checklist'),
  makeTable(
    ['Item', 'Status'],
    [
      ['Executive summary with purpose and conclusion', 'Complete'],
      ['Intended use and non-intended use documented', 'Complete'],
      ['Historical VaR/ES documented', 'Complete'],
      ['Parametric VaR/ES documented', 'Complete'],
      ['Monte Carlo VaR/ES documented', 'Complete'],
      ['Black-Scholes pricing documented', 'Complete'],
      ['Backtesting (Kupiec + Christoffersen + CC) documented', 'Complete'],
      ['EWMA estimation documented', 'Complete'],
      ['Input/output schema documented', 'Complete'],
      ['Architecture diagram included', 'Complete'],
      ['Requirement coverage matrix included', 'Complete'],
      ['Test plan included', 'Complete'],
      ['Test results (569 passed, 92% coverage)', 'Complete'],
      ['Live backtest result table', 'Complete'],
      ['23/23 homework validation table', 'Complete'],
      ['Limitations table', 'Complete'],
      ['Recommendations', 'Complete'],
      ['Bibliography', 'Complete'],
      ['Live application screenshots (8 tabs)', 'Complete'],
      ['Credit risk modules (hazard, Merton, CDS, CVA)', 'Complete'],
      ['Regulatory capital and DFAST documented', 'Complete'],
    ],
    [7200, 2160]
  ),
];

// ── Build document ─────────────────────────────────────────────────────────
const doc = new Document({
  styles: {
    default: {
      document: { run: { font: 'Arial', size: 22 } },
    },
    paragraphStyles: [
      {
        id: 'Heading1', name: 'Heading 1', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 32, bold: true, font: 'Arial', color: '1F3864' },
        paragraph: { spacing: { before: 320, after: 160 }, outlineLevel: 0 }
      },
      {
        id: 'Heading2', name: 'Heading 2', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 26, bold: true, font: 'Arial', color: '2E75B6' },
        paragraph: { spacing: { before: 240, after: 120 }, outlineLevel: 1 }
      },
      {
        id: 'Heading3', name: 'Heading 3', basedOn: 'Normal', next: 'Normal', quickFormat: true,
        run: { size: 24, bold: true, font: 'Arial', color: '404040' },
        paragraph: { spacing: { before: 160, after: 80 }, outlineLevel: 2 }
      },
    ]
  },
  numbering: { config: [] },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 1440, right: 1440, bottom: 1440, left: 1440 }
      }
    },
    headers: {
      default: new Header({
        children: [new Paragraph({
          border: { bottom: { style: BorderStyle.SINGLE, size: 4, color: DBLUE, space: 2 } },
          children: [
            new TextRun({ text: 'MATH GR 5320 — Portfolio Risk Management System', font: 'Arial', size: 18, color: '606060' }),
            new TextRun({ text: '\t', font: 'Arial', size: 18 }),
            new TextRun({ text: 'Nigel Li · Michael Adegbite · Stella', font: 'Arial', size: 18, color: '606060' }),
          ],
          tabStops: [{ type: 'right', position: 8640 }]
        })]
      })
    },
    footers: {
      default: new Footer({
        children: [new Paragraph({
          border: { top: { style: BorderStyle.SINGLE, size: 4, color: DBLUE, space: 2 } },
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: 'Page ', font: 'Arial', size: 18, color: '808080' }),
            new TextRun({ children: [PageNumber.CURRENT], font: 'Arial', size: 18, color: '808080' }),
            new TextRun({ text: ' of ', font: 'Arial', size: 18, color: '808080' }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], font: 'Arial', size: 18, color: '808080' }),
          ]
        })]
      })
    },
    children
  }]
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(OUT, buf);
  console.log('Written:', OUT, '(' + (buf.length / 1024 / 1024).toFixed(1) + ' MB)');
}).catch(err => {
  console.error('ERROR:', err.message);
  process.exit(1);
});
