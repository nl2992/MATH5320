<div class="titlepage">

**Software Design Documentation**

MATH GR 5320 Portfolio Risk Management System

Columbia University, Financial Risk Management, Spring 2026

<div class="tabular">

L4cmL9cm **Field** & **Value**\
Deliverable & 2 of 5 (15 points)\
Authors & Nigel Li, Michael Adegbite, Stella\
Reference Commit & `a4aa9e9` (main branch, May 2026)\
Submission Date & May 13, 2026\
Python Version & 3.12.2\
Test Suite & 644 tests, 0 failures\
Statement Coverage & 95%\

</div>

<div class="minipage">

*This document describes the architecture, module design, data flow, validation strategy, and known limitations of the MATH5320 risk engine. Per Lecture 5 requirements, it gives primary emphasis to the justification of every model choice, the mathematical specification of each technique, analysis of assumptions and limitations, comparison with alternatives, and documentation of subjective calibration decisions.*

</div>

</div>

# Executive Summary

The `MATH5320` repository implements a modular Python and Streamlit risk engine for portfolios of stocks and European options. The design separates user-interface code, data loading, portfolio representation, pricing logic, risk models, orchestration, and validation tests into distinct layers, keeping quantitative functions independently testable and treating the UI as a thin presentation layer over a reusable analytical core.

Three VaR/ES methods (historical simulation, delta-normal parametric, Monte Carlo) are implemented alongside Black-Scholes pricing, EWMA covariance estimation, walk-forward backtesting with Kupiec and Christoffersen diagnostics, and extension modules for credit, CVA, and regulatory capital. Each method is chosen for documented reasons, specified in full mathematical detail, and compared against the alternatives that were considered and not adopted.

The software architecture is justified on model-risk grounds: pure functions have no side effects and no Streamlit imports, which is the only design that makes independent validation tractable.

# System Purpose and Scope

## Purpose

The system is an educational portfolio risk application for MATH GR 5320. Its core purpose is to enable a user to:

- define a portfolio of stocks and European options;

- load historical market data via CSV upload or Yahoo Finance;

- compute VaR and ES under historical simulation, delta-normal parametric, and Monte Carlo methods;

- choose historical or manual calibration of market-risk parameters;

- compare methods under common data and parameter assumptions;

- run walk-forward VaR backtests with Kupiec and Christoffersen diagnostics;

- inspect supporting analytics and download structured outputs.

Extension modules cover exact GBM/lognormal risk, hazard-rate credit, Merton structural default, CDS, CVA, counterparty mitigation, and illustrative regulatory capital calculations. These are explicitly documented as course-formula extensions, not production models.

## Scope

**In scope:** stocks; European calls and puts; historical, parametric, and Monte Carlo VaR and ES; walk-forward backtesting with Kupiec and Christoffersen diagnostics; CSV and Yahoo Finance data loading; downloadable risk outputs; and course-formula validation modules.

**Out of scope:** production trading or risk management; enterprise authorization and audit; American or path-dependent option pricing; full volatility-surface or stochastic-volatility repricing; official supervisory DFAST/CCAR modeling; and production XVA systems.

## Lecture 5 Design Compliance Matrix

<div class="center">

<div class="tabular">

L6.5cmL7.5cm **Lecture 5 requirement** & **Where satisfied**\
Clear statement of purpose & Section 2\
Model choice justification by published research & Section 2.4\
Mathematical specification in detail & Section 2.5\
Assumptions, merits, and limitations & Section 2.6\
Comparison with alternatives & Section 2.7\
Subjective component documentation & Section 2.8\
Software design documentation & Sections 3–6\
Data analysis & Sections 4, 7, and 9\
Testing & Section 11\
System analysis and testing & Sections 4, 8, and 11\
Separation of concerns & Section 3\
Architecture justification & Section 2.9\

</div>

</div>

## Model Choice Justification and Literature Basis

The three VaR/ES methods and the supporting components were not chosen arbitrarily. Each maps to a recognised industry or academic standard. The table below links each method to its primary reference and states the core reason it was chosen over alternatives.

<div class="center">

<div class="tabular">

L2.6cmL4.5cmL5.8cm **Method** & **Reason for selection** & **Primary reference**\
Black-Scholes pricing & Industry-standard closed-form for European options; delta feeds parametric VaR directly & Black & Scholes (1973), *JPE* 81(3); Hull, *Options, Futures, and Other Derivatives*, 11th ed., Ch. 15\
Historical simulation & Non-parametric; no distributional assumption; captures observed fat tails and skewness in full repricing & McNeil, Frey & Embrechts, *Quantitative Risk Management*, Princeton, 2015, Ch. 2\
Delta-normal parametric & Closed-form; analytically tractable; serves as a fast linear baseline and a benchmark for the nonlinear engines & *RiskMetrics Technical Document*, J.P. Morgan/Reuters, 4th ed., 1996\
Monte Carlo VaR/ES & Full repricing of nonlinear payoffs; the standard for option-heavy portfolios; convergence to exact GBM VaR is verified in tests & Glasserman, *Monte Carlo Methods in Financial Engineering*, Springer, 2003, Ch. 9\
EWMA covariance & Down-weights old data; faster regime adaptation than rolling equal-weight; directly addresses volatility clustering & *RiskMetrics Technical Document*, 1996, Section 5; Zivot & Wang, *Modeling Financial Time Series with S-PLUS*, 2nd ed., Ch. 6\
Kupiec backtest & Likelihood-ratio test of unconditional exception frequency; the Basel II/III standard regulatory backtest & Kupiec (1995), *FEDS Discussion Paper 95-24*\
Christoffersen test & Conditional coverage; detects clustering of exceptions in time; complements Kupiec; directly relevant to our RED zone result & Christoffersen (1998), *Int. Economic Review* 39(4):841–862\
Hazard / Merton credit & Two canonical credit risk frameworks; reduced-form for market-implied default and structural for balance-sheet default & Lando, *Credit Risk Modeling*, Princeton, 2004; Merton (1974), *JF* 29(2)\
Regulatory capital & Basel III-inspired RWA and capital ratio; illustrative DFAST stress path; consistent with course formula sheet & BCBS, *Basel III: A Global Regulatory Framework*, 2010\

</div>

</div>

## Mathematical Specification and Numerical Techniques

This section gives the full mathematical formulation of each model and documents the numerical technique used in the implementation.

### Black-Scholes Option Pricing

The Black-Scholes-Merton (1973) formula prices a European call and put on an underlying following geometric Brownian motion with continuous dividend yield $`q`$:
``` math
\begin{align}
C &= S\,e^{-qT}\,N(d_1) - K\,e^{-rT}\,N(d_2) \label{eq:bs-call}\\
P &= K\,e^{-rT}\,N(-d_2) - S\,e^{-qT}\,N(-d_1) \label{eq:bs-put}
\end{align}
```
where
``` math
d_1 = \frac{\ln(S/K)+(r-q+\tfrac{1}{2}\sigma^2)T}{\sigma\sqrt{T}},
\qquad
d_2 = d_1 - \sigma\sqrt{T}
```
and $`N(\cdot)`$ is the standard normal CDF. The delta sensitivities that feed the parametric VaR engine are:
``` math
\Delta_C = e^{-qT}\,N(d_1), \qquad \Delta_P = -e^{-qT}\,N(-d_1)
```

**Numerical technique.** Formulas <a href="#eq:bs-call" data-reference-type="eqref" data-reference="eq:bs-call">[eq:bs-call]</a>–<a href="#eq:bs-put" data-reference-type="eqref" data-reference="eq:bs-put">[eq:bs-put]</a> are evaluated directly using `scipy.stats.norm.cdf`. No numerical integration or lattice is used. At $`T \to 0^+`$ the implementation returns discounted intrinsic value; negative volatility or maturity raises a controlled exception rather than returning a NaN.

### Historical Simulation VaR and ES

Let $`\mathbf{S}_t \in \mathbb{R}^m`$ be the current spot vector and $`\mathbf{r}_{t-k} \in \mathbb{R}^m`$ the log-return vectors for $`k=1,\ldots,n_{\mathrm{lb}}`$. Form $`h`$-day cumulative return scenarios by summing overlapping returns:
``` math
\mathbf{R}_k^{(h)} = \sum_{j=0}^{h-1} \mathbf{r}_{t-k-j},
\qquad k = 1,\ldots,n_{\mathrm{lb}}-h+1
```

Apply each scenario to the current spot vector and reprice the full portfolio:
``` math
\tilde{V}_k = V_{\text{portfolio}}\!\left(\mathbf{S}_t \odot e^{\mathbf{R}_k^{(h)}}\right),
\qquad L_k = V_t - \tilde{V}_k
```

Sort losses in ascending order $`L_{(1)} \leq \cdots \leq L_{(n)}`$. The empirical quantile estimators are:
``` math
\widehat{\mathrm{VaR}}_\alpha = L_{(\lceil\alpha n\rceil)},
\qquad
\widehat{\mathrm{ES}}_\alpha
  = \frac{1}{|\mathcal{E}|}\sum_{k:\,L_k>\widehat{\mathrm{VaR}}_\alpha} L_k
```

**Numerical technique.** Full repricing means option values are recomputed via Black-Scholes at each shocked spot. When `vol_shock_mode` is active, a proportional volatility shock $`\sigma_{\text{shocked}} = \sigma_0(1 + \beta \cdot R_{\text{underlying}})`$ is applied before repricing, capturing first-order vol-spot comovement.

### Delta-Normal Parametric VaR and ES

Define the delta-dollar exposure vector $`\boldsymbol{\delta}\in\mathbb{R}^m`$:
``` math
\delta_i = \begin{cases}
  q_i \cdot S_{i,t}
    & \text{stock with signed quantity }q_i\\
  q_j \cdot \Delta_j \cdot S_{j,t} \cdot M_j
    & \text{option with multiplier }M_j,\text{ delta }\Delta_j
\end{cases}
```

Under multivariate normality of $`h`$-day log returns, portfolio loss is normally distributed with:
``` math
L \sim \mathcal{N}\!\left(
  -\boldsymbol{\delta}^\top\boldsymbol{\mu}_h,\;
  \boldsymbol{\delta}^\top\Sigma_h\boldsymbol{\delta}
\right)
```
where $`\boldsymbol{\mu}_h = h\boldsymbol{\mu}`$ and $`\Sigma_h = h\Sigma`$ (square-root-of-time rule, exact under i.i.d. returns). VaR and ES follow in closed form:
``` math
\begin{align}
\mathrm{VaR}_\alpha
  &= -\boldsymbol{\delta}^\top\boldsymbol{\mu}_h
     + \sqrt{\boldsymbol{\delta}^\top\Sigma_h\boldsymbol{\delta}}
       \cdot\Phi^{-1}(\alpha) \label{eq:pvar}\\
\mathrm{ES}_\alpha
  &= -\boldsymbol{\delta}^\top\boldsymbol{\mu}_h
     + \sqrt{\boldsymbol{\delta}^\top\Sigma_h\boldsymbol{\delta}}
       \cdot\frac{\phi\!\left(\Phi^{-1}(\alpha)\right)}{1-\alpha}
       \label{eq:pes}
\end{align}
```

**Numerical technique.** The portfolio variance $`\boldsymbol{\delta}^\top\Sigma_h\boldsymbol{\delta}`$ is computed via `numpy` matrix-vector products. The covariance matrix is validated as symmetric and positive semi-definite before use.

### Monte Carlo VaR and ES

Draw $`n`$ independent $`h`$-day return scenarios from the estimated distribution:
``` math
\mathbf{R}^{(i)} = \boldsymbol{\mu}_h + L_c\,\mathbf{z}^{(i)},
\qquad \mathbf{z}^{(i)} \sim \mathcal{N}(\mathbf{0},I),
\qquad i=1,\ldots,n
```
where $`L_c`$ is the lower Cholesky factor of $`\Sigma_h = L_c L_c^\top`$. Reprice the full portfolio at each shocked spot vector to obtain scenario losses $`L^{(i)} = V_0 - V(\mathbf{S}_0 \odot e^{\mathbf{R}^{(i)}})`$, then estimate VaR and ES as in historical simulation.

The standard error of the VaR estimator scales as $`O(n^{-1/2})`$. At $`n=10{,}000`$ and $`\alpha=0.99`$, approximately 100 scenarios fall in the tail, giving a coefficient of variation of roughly 10% on the VaR point estimate. Tests confirm convergence to within 2% of the exact GBM VaR at $`n=100{,}000`$.

**Numerical technique.** Cholesky decomposition via `numpy.linalg.cholesky`. A fixed `random_seed` is used for all regression tests; the seed is stored alongside results for reproducibility.

### EWMA Covariance Estimation

The exponentially weighted moving average (EWMA) estimator with decay parameter $`\lambda = 1 - 1/N`$ updates daily as:
``` math
\begin{align}
\hat{\sigma}_{i,t}^2
  &= \lambda\,\hat{\sigma}_{i,t-1}^2 + (1-\lambda)\,r_{i,t-1}^2
  \label{eq:ewma-var}\\
\hat{\sigma}_{ij,t}
  &= \lambda\,\hat{\sigma}_{ij,t-1} + (1-\lambda)\,r_{i,t-1}\,r_{j,t-1}
  \label{eq:ewma-cov}
\end{align}
```

The effective memory of this estimator is $`N`$ days: half the total weight is carried by the most recent $`N\ln 2 \approx 0.693N`$ observations. For the default $`N=60`$, $`\lambda \approx 0.9833`$ and the effective look-back is roughly one calendar quarter.

Equations <a href="#eq:ewma-var" data-reference-type="eqref" data-reference="eq:ewma-var">[eq:ewma-var]</a>–<a href="#eq:ewma-cov" data-reference-type="eqref" data-reference="eq:ewma-cov">[eq:ewma-cov]</a> are vectorised over all asset pairs and initialised with sample estimates from the first 30 return observations (burn-in). The resulting matrix is positive semi-definite by construction because it is a convex combination of positive semi-definite matrices.

### Walk-Forward Backtesting and Statistical Diagnostics

The backtest estimates a VaR forecast at each date $`t`$ using only information available up to and including $`t`$, then observes the realised $`h`$-day loss. Define the exception indicator:
``` math
I_t = \mathbf{1}\{L_t > \widehat{\mathrm{VaR}}_\alpha^t\}
```

**Kupiec (1995) unconditional coverage test.** Under $`H_0`$: exceptions are i.i.d. Bernoulli$`(\alpha)`$, the log-likelihood ratio statistic:
``` math
\mathrm{LR}_{\mathrm{uc}}
= 2\!\left[
    x\ln\frac{x}{T\alpha}
  + (T-x)\ln\frac{T-x}{T(1-\alpha)}
  \right] \xrightarrow{d} \chi^2_1
```
where $`x = \sum_t I_t`$ is the exception count and $`T`$ is the number of forecasts. This test is sensitive to the *frequency* of exceptions only.

**Christoffersen (1998) independence test.** Define the $`2\times2`$ transition count matrix $`n_{ij} = \#\{t: I_{t-1}=i, I_t=j\}`$ and estimated transition probabilities $`\hat\pi_{ij} = n_{ij}/(n_{i0}+n_{i1})`$. The independence statistic:
``` math
\mathrm{LR}_{\mathrm{ind}}
= -2\ln L(\hat\alpha)
  +2\ln L(\hat\pi_{00},\hat\pi_{10})
\xrightarrow{d} \chi^2_1
```
tests whether the probability of an exception today depends on whether an exception occurred yesterday. For our run: $`\mathrm{LR}_{\mathrm{ind}} = 62.20`$, $`p < 10^{-14}`$, a strong rejection indicating that historical simulation produces clustered exceptions during high-volatility sub-periods.

## Assumptions, Merits, and Limitations

For each model, we state the key assumptions explicitly, describe what market risks are captured, and document where the model is expected to underperform.

<div class="center">

<div class="tabular">

L1.8cmL3.5cmL3.0cmL3.5cmL2.5cm **Model** & **Key assumptions** & **Risks captured** & **Risks not captured / gaps** & **Main merit**\
Black-Scholes & Constant $`\sigma`$; GBM underlying; continuous trading; no jumps; flat rates & European payoffs exactly; delta sensitivity & Volatility smile; jumps; stochastic vol; discrete dividends & Closed-form; fast; industry standard\
Historical simulation & Past returns representative of future; i.i.d. returns within window; constant composition & Fat tails, skewness, and correlation as observed; option nonlinearity via full repricing & Cannot extrapolate beyond worst historical loss; slow to adapt to new regimes (confirmed by Christoffersen rejection) & No distributional assumption; most honest representation of observed data\
Delta-normal parametric & Multivariate normal returns; delta approximation valid for options; constant delta over horizon & Equity market risk; correlation structure; fast closed-form sensitivity & Fat tails; gamma; vega; any nonlinear payoff beyond first-order delta & Closed-form; analytically tractable; fast for large portfolios\
Monte Carlo & Same MVN as parametric; Cholesky factor exists (PSD covariance) & Option nonlinearity via full repricing; same fat-tail limitation as parametric unless distribution changed & Model risk from MVN assumption; seed-to-seed variance in tail estimates & Handles complex nonlinear portfolios; framework is extensible to non-normal distributions\
EWMA & Exponential decay of information; no structural breaks; stationary covariance & Volatility clustering; recent regime shifts & Optimal $`\lambda`$ is data-dependent and not calibrated here; abrupt structural breaks not handled & Responsive to recent data; computationally efficient\
Kupiec & Exceptions are i.i.d. Bernoulli$`(\alpha)`$ under $`H_0`$ & Exception frequency relative to model confidence level & Exception clustering; tail severity; conditional coverage & Standard regulatory test; interpretable p-value\
Christoffersen & First-order Markov exception sequence under $`H_1`$ & Exception clustering / independence from previous day & Higher-order dependence; severity of exceptions & Directly addresses historical simulation’s known clustering failure\

</div>

</div>

**Overall risk coverage.** The system captures equity market risk (delta and gamma via full repricing), volatility clustering (EWMA estimator), observed fat tails (historical simulation), option nonlinearity (MC and historical engines), credit default risk (hazard-rate and Merton modules), counterparty exposure (CVA module), and illustrative regulatory capital adequacy (Basel III-inspired RWA and DFAST stress path).

The system does *not* capture: liquidity risk or bid-ask spreads; intraday price moves (daily closes only); jump diffusion or sudden gap risk; model parameter estimation error (no confidence intervals around VaR estimates); basis risk between position and hedge; or operational risk. These gaps are documented rather than silently ignored.

## Comparison with Alternative Approaches

Three alternative families of methods were explicitly considered and not adopted. Each decision is documented below.

<div class="center">

<div class="tabular">

L2.6cmL3.0cmL3.2cmL4.0cm **Property** & **Historical simulation** & **Delta-normal parametric** & **Monte Carlo**\
Distributional assumption & None (empirical) & Multivariate normal & Multivariate normal (extendable)\
Option handling & Full repricing & Delta approximation & Full repricing\
Computation & Moderate; $`O(n_{\text{lb}} \cdot m)`$ & Very fast, $`O(m^2)`$ closed-form & Slow, $`O(n \cdot m)`$\
Fat-tail capture & Yes, from data & No & Partial; MVN distributional\
Volatility clustering & No; equally weighted & Yes with EWMA & Yes with EWMA\
Extrapolation beyond history & No & Yes (distributional) & Yes (distributional)\
Main limitation & Regime-change blindness & Underestimates gamma/vega risk & MVN assumption; seed variance\

</div>

</div>

**Alternatives considered and not adopted:**

- **Age-weighted historical simulation (Boudoukh, Richardson & Whitelaw, 1998).** Assigns exponentially decaying weights to historical scenarios, directly addressing regime-change blindness. Not implemented because the EWMA covariance estimator on the parametric and Monte Carlo engines achieves a comparable effect on second-moment adaptation without modifying the historical engine’s scenario set. For a production build, age-weighting would be the first enhancement to the historical engine.

- **Extreme Value Theory (EVT, Pickands-Balkema-de Haan theorem; McNeil & Frey, 2000).** Fits a Generalised Pareto Distribution to the tail of the loss distribution, providing superior tail estimation beyond the range of observed data. Not implemented because it requires a minimum of several hundred tail observations to calibrate reliably and the pedagogical gain at course scale is limited. Documented as the correct next step for production tail risk estimation.

- **Filtered Historical Simulation (FHS; Barone-Adesi, Giannopoulos & Vosper, 1999).** Applies GARCH or EGARCH filtering to standardise returns before drawing scenarios from the filtered empirical distribution. This directly addresses the Christoffersen independence rejection: GARCH-filtered residuals are closer to i.i.d. than raw returns. Not implemented because the course scope covers only EWMA, but FHS is the appropriate production replacement for the historical engine given our backtest results.

- **Binomial tree option pricing.** The CRR binomial tree converges to Black-Scholes in the limit and is more flexible for American options, but is $`O(N^2)`$ per evaluation versus $`O(1)`$ for Black-Scholes. Since the scope is European options only, Black-Scholes is the dominant choice.

- **SABR / Heston stochastic volatility.** Would capture the volatility smile and allow consistent implied-vol surface interpolation. Not implemented because implied-vol market data is not part of the course data layer, and the added complexity is out of scope for a daily-VaR risk engine rather than an options market-making system.

## Subjective Design Choices and Calibration Parameters

Several parameters required expert judgment. They are documented here with their justification so that a reviewer can reproduce, challenge, or override each choice without reading the source code.

<div class="center">

<div class="tabular">

L3.0cmL1.8cmL8.0cm **Parameter** & **Default** & **Justification and sensitivity**\
Lookback window & 252 days & One calendar year; the standard RiskMetrics and Basel II convention. Shorter windows (e.g. 60 days) react faster to recent volatility but produce noisier estimates; longer windows (e.g. 504 days) are more stable but may mix fundamentally different market regimes. Robustness tests confirm finite, monotone VaR across 60–504 days.\
EWMA half-life $`N`$ & 60 & Corresponds to $`\lambda = 59/60 \approx 0.9833`$ decay per day and an effective memory of roughly one calendar quarter. RiskMetrics (1996) recommends $`\lambda=0.94`$ ($`N\approx17`$) for daily equities; we choose $`N=60`$ for greater stability, accepting slower regime adaptation. Robustness tests show PSD covariance and smooth VaR across $`N \in
    \{10, 20, 60, 120\}`$.\
Monte Carlo paths & 10 000 & Balances stability against runtime at academic scale. At $`n=10{,}000`$, approximately 100 scenarios fall in the 99% tail, giving a VaR coefficient of variation of roughly 10%. A seeded test at $`n=100{,}000`$ confirms convergence to within 2% of the exact GBM VaR; gains beyond 10 000 paths are marginal for coursework.\
Vol floor & 1% & Applied only during the vol-shock repricing path. Prevents numerical instability when a short history produces near-zero estimated volatility. Has no effect when $`\sigma > 0.01`$.\
Confidence levels & 99% VaR; 97.5% ES & Aligns with Basel III IMA: 99% 10-day VaR for market risk; 97.5% ES under FRTB. Both are user-configurable; tests cover 95%, 97.5%, 99%, and 99.9%.\
Risk horizon & 5 days & The Basel III standard for daily VaR reporting and backtesting. The square-root-of-time rule is used for horizon scaling ($`\Sigma_h = h\Sigma`$), which is exact under i.i.d. returns and is an approximation otherwise.\
Backtest lookback & 252 days & Enough observations to make the Kupiec and Christoffersen statistics well-powered, consistent with Basel guidance of at least one year of daily VaR forecasts.\

</div>

</div>

**Validation of subjective choices.** Each parameter above is user-configurable through the risk-settings UI. Robustness tests in `tests/test_coverage_gaps.py` and `tests/test_strict_numerics.py` sweep the key parameters (lookback 60–504 days; EWMA $`N`$ 10–120; MC paths 100–10 000; confidence 95%–99.9%) and confirm that results are finite, positive, and monotone throughout the tested ranges.

## Software Architecture Justification

The layered architecture is not a stylistic preference; it follows directly from the model-risk requirements of Lecture 5 and from broadly adopted industry guidance on model governance.

**Pure functions with no side effects.** Every module in `src/risk/`, `src/pricing/`, and `src/credit/` is a pure function of its inputs: given the same arguments it returns the same output, with no network calls, no global state, and no Streamlit imports. This is the only design that makes independent validation tractable. A test can call `historical_var_es()` directly from a pytest fixture without starting the application, which means the model layer can be validated by anyone with Python, not just users of the Streamlit UI. This matches the SR 11-7 principle that model validation should be independent of the production environment.

**Separation of data gathering, calibration, and computation.** Lecture 5 explicitly identifies mixing of these concerns as a model-risk weakness. In this repository: `src/data/` handles all data acquisition and cleaning; `src/risk/estimators.py` handles parameter estimation; `src/risk/historical.py`, `parametric.py`, and `monte_carlo.py` perform risk computation; and `src/ui/` handles presentation. None of these modules imports from another layer’s responsibility. The benefit is that the covariance estimator can be tested without market data, and the VaR engine can be tested with any estimated covariance, whether from EWMA, rolling window, or a manually specified matrix.

**Service layer as model governance boundary.** `src/services/risk_engine_service.py` is the single orchestration point for the standard risk workflow. A UI panel or notebook that calls the service layer cannot accidentally change how the model is calibrated or which assumptions are applied, because those decisions live entirely inside the service and its downstream model modules. This mirrors the “model boundary” concept in model risk governance: the production environment should only interact with a model through its defined interface, never by reaching inside it.

**Why not a monolithic script?** A single script that loads data, estimates parameters, computes VaR, and renders charts would be functionally equivalent but would require the full application to run in order to test any one piece of it. The 644-test suite, which runs in under 30 seconds without any network calls, is only possible because the model layer has no Streamlit or network dependencies.

# Software Architecture

## Layered Architecture

The system follows a strict layered architecture in which each layer has a single responsibility and well-defined interfaces with adjacent layers:

<div class="center">

</div>

## Why This Architecture Fits a Risk Engine

Pure model functions can be called directly from the test suite and from the Jupyter notebook environment without any Streamlit state, which makes independent validation straightforward. Module responsibilities are narrow enough to be read and understood in isolation. Extension modules for credit, CVA, and regulatory calculations can be added without modifying the application shell or the core stock-and-option risk path.

## Separation of Concerns

<div class="center">

<div class="tabular">

L2.5cmL5.5cmL5.5cm **Layer** & **Responsibility** & **Must not do**\
Data layer & Load, parse, clean, align, and validate prices & Compute VaR or ES\
Estimator layer & Estimate means, volatilities, and covariances & Download data or render UI\
Pricing layer & Price stocks/options; compute option sensitivities & Portfolio-level workflow decisions\
Portfolio layer & Aggregate positions, values, and exposures & Estimate risk-model parameters\
Risk layer & Compute VaR, ES, scenarios, and backtests & Render charts or collect user input\
Service layer & Orchestrate the end-to-end workflow & Hide quantitative formulas in UI code\
UI layer & Collect inputs and display outputs & Mutate model logic or silently change assumptions\

</div>

</div>

# Data Flow and Control Flow

## End-to-End Data-Flow Diagram

<div class="center">

</div>

## Core Modeling Conventions

- Daily log returns: $`r_{i,t} = \log(S_{i,t}/S_{i,t-1})`$

- Overlapping horizon returns: $`R_t^{(h)} = \sum_{k=0}^{h-1} r_{t-k}`$

- Price shock: $`S_{\mathrm{shocked}} = S_0 \cdot e^R`$

- Portfolio PnL: $`V_T - V_0`$; loss: $`V_0 - V_T`$

- Horizon scaling: $`\boldsymbol{\mu}_h = h\boldsymbol{\mu}`$, $`\Sigma_h = h\Sigma`$

- Option pricing: Black-Scholes with continuous dividends

- Kupiec backtest statistic: $`\mathrm{LR}_{\mathrm{uc}} \sim \chi^2_1`$

## Backtesting Control-Flow Diagram

<div class="center">

</div>

Backtesting is implemented as a strictly out-of-sample walk-forward process in `src/risk/backtest.py`. The implementation never uses future data to estimate the VaR forecast at the current date: it slices prices up to the forecast date, fits on that historical subset, and then compares the resulting VaR with the realized future loss over the chosen horizon.

# Module-by-Module Design

## Module Inventory

<div class="center">

<div class="tabular">

L2.4cmL3.8cmL2.4cmL2.2cmL2.4cm **Module** & **Purpose** & **Inputs** & **Outputs** & **Test evidence**\
Schemas & Define stock, option, and portfolio objects & User inputs & Structured portfolio objects & `test_config_and_validation.py`\
Market data & CSV and Yahoo Finance loading & Tickers, dates, CSVs & Aligned price data & `test_market_data.py`\
Data validation & Validate price and input data & Prices, settings & Errors or clean acceptance & `test_config_and_validation.py`\
Black-Scholes & European option pricing and delta & $`S,K,T,r,q,\sigma`$, type & Price, delta & `test_backend.py`, `test_homework_cases.py`\
Position valuation & Per-position value, vol shock, delta-dollar sensitivity & Position and market inputs & Value, exposure & `test_backend.py`\
Portfolio valuation & Aggregate positions and exposures & Portfolio, spot vector & Total value, exposure vector & `test_backend.py`\
Returns & Log and horizon return construction & Price matrix & Return matrix & `test_backend.py`, `test_coverage_gaps.py`\
Estimators & Rolling, EWMA, and manual covariance assembly & Return matrix or manual params & Mean, covariance & `test_backend.py`, `test_homework_cases.py`\
Historical risk & Historical VaR/ES with full repricing and vol shock & Portfolio, history, settings & VaR, ES, losses & `test_backend.py`, `test_course_validation.py`\
Parametric risk & Delta-normal VaR/ES & Exposure vector, covariance & VaR, ES & `test_backend.py`, `test_es_confidence_split.py`\
Monte Carlo risk & Simulated VaR/ES & Mean/cov, portfolio, vol-shock settings & VaR, ES, losses & `test_backend.py`, `test_coverage_gaps.py`\
Backtesting & Walk-forward VaR validation & History, model settings & Exceptions, Kupiec, Christoffersen & `test_backend.py`, `test_backtest_extensions.py`\
Exact GBM & Formula-sheet GBM VaR/ES & GBM parameters & Exact VaR, ES & `test_lognormal.py`, `test_course_validation.py`\
Hazard & Reduced-form default & Hazard, recovery, maturity & Survival, density, spread & `test_credit.py`, `test_course_validation.py`\
Merton & Structural default & $`V_0,B,T,r,\mu,\sigma`$ & PD, equity, debt & `test_credit.py`, `test_course_validation.py`\
CDS & Par spread calculation & Hazard, recovery, discounting & Spread, protection/premium values & `test_credit.py`, `test_course_validation.py`\
CVA & Counterparty valuation adjustment & Exposure, PD, recovery & CVA & `test_credit.py`, `test_cva_mitigants.py`\
Regulatory & RWA, capital ratio, DFAST-style calculations & Assets, losses, weights & Ratios, stress metrics & `test_regulatory.py`, `test_dfast_pathing.py`\
Service & Orchestrate end-to-end core risk run & Portfolio, data, settings & Unified result object & `test_backend.py`, `integration_test.py`\
UI & Streamlit display and input logic & User interaction & Rendered panels & `test_ui_panels.py`, `test_charts.py`\

</div>

</div>

## Public Interfaces and Contracts

<div class="center">

<div class="tabular">

L3.5cmL4.5cmL3.0cmL2.5cm **Interface** & **Inputs** & **Outputs** & **Failure mode**\
`black_scholes_price` & $`S,K,T,\sigma,r,q`$, option type & Option price & Reject invalid type or numerical domain\
`black_scholes_delta` & Same inputs & Delta & Reject invalid type or numerical domain\
`portfolio_value` & Portfolio object, spot vector & Portfolio value & Fail visibly if required underlying missing\
`portfolio_delta_dollar` & Portfolio, spot vector & Exposure vector & Fail visibly if option input inconsistent\
`historical_var_es` & Portfolio, price history, horizon, confidences & VaR, ES, losses & Return explicit reason if history insufficient\
`parametric_var_es` & Exposure vector, mean/covariance, horizon & VaR, ES & Reject invalid covariance or confidence\
`monte_carlo_var_es` & Portfolio, mean/cov, paths, seed & VaR, ES, losses & Reject invalid covariance or path count\
`run_backtest` & Portfolio, price history, model choice, lookback & Exception series, summary & Return explicit no-result reason when infeasible\
`RiskEngineService` & Portfolio, prices, pricing date, settings & Unified result dict & Surface errors to the UI rather than swallowing\

</div>

</div>

## Core Data Structures

<div class="center">

<div class="tabular">

L3.5cmL5.5cmL4.5cm **Object** & **Main fields** & **Design role**\
`StockPosition` & Ticker, quantity (signed) & Represents signed equity holdings\
`OptionPosition` & Underlying, type, strike, maturity, volatility, rate, dividend yield, multiplier, quantity & Represents European option contracts\
`Portfolio` & Stock positions, option positions & Single object passed through valuation and risk workflows\
Risk settings bundle & Lookback, horizon, confidences, estimator type, calibration mode, MC paths, vol-shock controls & Standardizes user configuration across risk engines\
`ManualMarketParams` & Daily mean vector, volatility/covariance inputs & Allows parameter-driven parametric and MC runs\
Risk result dictionary & Method-level VaR, ES, losses, portfolio value, chart payloads & Feeds the UI and test artifacts\
`BacktestResult` & Forecast dates, realized losses, VaR forecasts, exceptions, diagnostics & Records outcome-analysis evidence\

</div>

</div>

## Operational Paths

**Current risk run:**

1.  Validate portfolio and market-data availability.

2.  Compute current stock and option values.

3.  Estimate or accept market-risk parameters.

4.  Run historical, parametric, and Monte Carlo models.

5.  Aggregate VaR, ES, losses, and charts into one result object.

**Backtest run:**

1.  Move through time using a rolling estimation window.

2.  Re-estimate the selected model at each date.

3.  Forecast VaR using only information available at that date.

4.  Observe the realized future loss.

5.  Record exceptions and compute Kupiec and Christoffersen diagnostics.

# Input and Output Schemas

## Stock Input

<div class="center">

<div class="tabular">

L2.5cmL1.8cmL4cmL5cm **Field** & **Type** & **Rule** & **Validation behaviour**\
`ticker` & string & Non-empty symbol & UI prevents empty ticker\
`quantity` & numeric & Positive or negative (signed) & Numeric handling enforced in data-entry flow\

</div>

</div>

## Option Input

<div class="center">

<div class="tabular">

L2.8cmL1.8cmL3.5cmL4.5cm **Field** & **Type** & **Rule** & **Validation behaviour**\
`label`/`ticker` & string & Non-empty & UI avoids blank partial rows\
`underlying_ticker` & string & Must exist in price data & Explicit ticker-existence check\
`option_type` & string & `call` or `put` & Pricing layer rejects unknown types\
`quantity` & numeric & Signed & Numeric handling enforced\
`strike` & numeric & Positive & Invalid values fail in pricing\
`maturity` & date & Future-dated for live pricing & Expired options handled; malformed dates fail upstream\
`volatility` & numeric & Positive decimal & Pricing layer rejects non-positive volatility\
`risk_free_rate` & numeric & Numeric decimal & Numeric input expected\
`dividend_yield` & numeric & Numeric decimal & Numeric input expected\
`multiplier` & numeric & Positive & Expected positive\

</div>

</div>

## Risk Settings

<div class="center">

<div class="tabular">

L2.8cmL1.8cmL3cmL4.5cm **Field** & **Type** & **Rule** & **Validation behaviour**\
Lookback window & integer & Positive, sufficiently large & Infeasible windows surface as insufficient-history error\
Horizon & integer & Positive & Return helpers reject invalid horizons\
VaR confidence & float & $`(0,1)`$ & Domain covered in tests\
ES confidence & float & $`(0,1)`$ & Separate-confidence tests exist\
Estimator type & enum & `window` or `ewma` & UI constrains\
Calibration mode & enum & `historical` or `manual` & UI constrains; tests cover manual mode\
MC simulations & integer & Positive & Positive value expected\
Random seed & integer or `None` & Fixed for reproducibility & Both modes supported\
Vol shock mode & enum & `fixed` or `underlying_beta` & Unknown modes raise visibly\

</div>

</div>

## Output Schema

<div class="center">

<div class="tabular">

L3.5cmL9cm **Output** & **Contents**\
Risk summary (JSON) & Method, VaR, ES, VaR/ES confidence levels, horizon, portfolio value, assumptions\
Losses CSV & Scenario id or date, method, loss under convention $`V_0 - V_T`$\
Backtest CSV & Model, observations, exceptions, expected exceptions, exception rate, Kupiec LR, p-value\

</div>

</div>

# Risk Engine Orchestration

The core orchestration path is implemented in `src/services/risk_engine_service.py`.

## Core Steps

1.  Accept a validated `Portfolio`, aligned `prices` DataFrame, a `pricing_date`, and a risk-settings bundle.

2.  Compute current portfolio value from the latest spot vector.

3.  Dispatch to `historical_var_es`, `parametric_var_es`, and `monte_carlo_var_es`.

4.  Aggregate results into a single dictionary keyed by method.

5.  On backtest requests, call `run_backtest` and `kupiec_test`.

6.  Return service-level objects to the UI for display and download.

## Why the Service Layer Matters

The service layer isolates each Streamlit panel from knowledge of portfolio repricing, return estimation, scenario generation, and output aggregation. Acting as a single orchestration boundary makes the application’s model governance tractable: all model-selection decisions are made inside the service, not scattered across UI callbacks.

## Extension Orchestration

Credit and regulatory logic use separate service modules (`src/services/credit_service.py` and `src/services/regulatory_service.py`), keeping the required stock/option risk workflow independent from the course-formula extensions.

# Data Validation and Error Handling

## Error-Handling Table

<div class="center">

<div class="tabular">

L4.5cmL3.5cmL5.5cm **Error case** & **Detection layer** & **Required behaviour**\
Missing ticker in price history & Data validation & Raise or display explicit error\
Too few observations & Risk module & Return explicit empty reason\
Negative or zero prices & Data validation & Reject explicitly\
Duplicate dates & Data loader & Reject explicitly\
NaN prices & Data loader/validation & All-NaN columns flagged\
Invalid option maturity & Schema/pricing & Expired options handled; malformed fail visibly\
Negative volatility & Schema/pricing & Pricing fails visibly\
Invalid confidence & Risk settings & Constrained by UI or fails visibly\
Non-PSD covariance & Estimator/MC & Manual inputs rejected explicitly\
Empty portfolio & Schema/service & Prevented or fails visibly\
Download failure & Data layer/UI & Clear user-facing error message\
Monte Carlo seed missing & MC layer & Randomize and record, or require seed for regression\

</div>

</div>

## Data Validation Design

- `src/data/validation.py`: emptiness, index type, all-NaN columns, positive prices, stale-price runs.

- `src/data/market_data.py`: CSV parsing, numeric coercion, Yahoo Finance retrieval, exponential-backoff retry, parquet caching.

- `src/ui/market_data_panel.py`: surfaces errors immediately to the user rather than silently proceeding.

# Numerical Implementation Controls

<div class="center">

<div class="tabular">

L6cmL7.5cm **Numerical risk** & **Control**\
Floating-point equality & `pytest.approx` / `np.isclose` in all tests\
Overflow in lognormal shocks & Validate inputs; use analytical formulas where available\
Underflow in deep OTM option prices & Permit small positive values; reject negative option prices\
Catastrophic cancellation & Prefer analytical Greeks over unstable finite differences\
Non-PSD covariance & Test structure and document error or repair expectations\
Monte Carlo randomness & Fixed seed for all regression-style tests\
Stale or missing data & Validate before calculation\
Wrong loss sign & Explicit PnL/loss conventions plus sign-aware tests\
VaR/ES confidence confusion & Separate-confidence tests in `test_es_confidence_split.py`\
Delta/exposure unit mismatch & Dedicated regression test confirms delta-dollar convention\

</div>

</div>

# Testing and Coverage

## Test Integration by Layer

- Formula modules are validated by analytical and course-derived tests.

- Service orchestration is validated by backend integration tests.

- UI panels are validated with Streamlit panel tests.

- Market-data wrappers are validated independently from the risk engine.

- Integration scripts exercise end-to-end behavior with live market data.

## Test Execution Summary

No-network unit test command:

<div class="shellcode">

python -m pytest tests/  –ignore=tests/integration_test.py  –ignore=tests/integration_test_formula_sheet.py -v

</div>

Recorded result (2026-05-13, commit `a4aa9e9b`): **644 passed, 0 failed, 1 intentionally skipped**.

Coverage command:

<div class="shellcode">

python -m pytest tests/ –cov=src –cov-report=term-missing  –cov-report=html:submission/coverage_report  –ignore=tests/integration_test.py  –ignore=tests/integration_test_formula_sheet.py

</div>

Recorded result: **95% statement coverage**.

## Coverage and Remediation Plan

<div class="center">

<div class="tabular">

L3.5cmL5.5cmL4.5cm **Area** & **Remaining gap** & **Planned remediation**\
Streamlit panel branches & Some UI display paths require live browser state & Add headless browser tests as a future enhancement\
Credit-service helpers & Some orchestration branches are lower value & Add service-level synthetic workflows\
Defensive validation & Some invalid states are hard to trigger from the UI & Add direct unit tests against validators\
Historical edge paths & Insufficient-history and absolute-shock branches & Add synthetic price matrices with controlled missing data\

</div>

</div>

# Deployment and Reproducibility

## Core Commands

<div class="shellcode">

pip install -r requirements.txt streamlit run app.py python -m pytest tests/  –ignore=tests/integration_test.py  –ignore=tests/integration_test_formula_sheet.py

</div>

## Recorded Test Environment

- Date/time: 2026-05-13 00:00:00 EDT

- Git commit: `a4aa9e9b0ef8ba069a0331fe22c3cbe6a8c5dc0d`

- Python: 3.12.2 OS: Darwin 24.5.0 arm64

- Key packages: `streamlit 1.37.1`, `numpy 1.26.4`, `pandas 3.0.2`, `scipy 1.17.1`, `yfinance 1.2.0`, `pytest 7.4.4`, `pytest-cov 7.1.0`

## Reproducibility Assessment

Reproducibility is strong for deterministic and no-network paths: analytical modules are deterministic, Monte Carlo defaults to a fixed seed for regression-stable tests, and the application depends on explicit settings objects. Reproducibility is weaker for live-data integration because Yahoo Finance data may update.

# Known Software Limitations

<div class="center">

<div class="tabular">

L4cmL4.5cmL5cm **Limitation** & **Consequence** & **Mitigation**\
Not production software & No enterprise auth, audit, or access controls & Academic use only\
Yahoo Finance data can be imperfect & Stale prices or connection failures & Retry logic and parquet cache; Bloomberg CSV as fallback\
Black-Scholes only & No stochastic volatility, no smile modeling & Appropriate for course scope; documented\
Parametric uses delta-normal & Nonlinear option payoffs underestimated & Historical and MC use full repricing; disclosed\
EWMA not ML-calibrated & May not match market-observed clustering & Acceptable for academic purpose\
Regulatory is illustrative & Does not claim official Basel or DFAST compliance & Course formula implementations only\
MC convergence is seed-dependent & Unseeded runs introduce variance & Fixed seeds for regression tests\
95% coverage rather than 100% & Some UI branches untested & Non-formula paths; documented in Section 11.3\
Historical simulation clusters exceptions & Christoffersen $`p < 10^{-14}`$ & EWMA/FHS path recommended for production\

</div>

</div>

# Design Compliance with Lecture 5

1.  **Requirements.** Section 2: purpose and scope.

2.  **Model choice justification.** Section 2.4: each method tied to a primary reference.

3.  **Mathematical specification.** Section 2.5: full equations for all six models including numerical techniques.

4.  **Assumptions, merits, limitations.** Section 2.6: per-model assumption list, risk coverage, and documented gaps.

5.  **Comparison with alternatives.** Section 2.7: FHS, EVT, age-weighting, binomial trees, and stochastic volatility all addressed.

6.  **Subjective components.** Section 2.8: every default parameter documented with justification and robustness evidence.

7.  **Architecture justification.** Section 2.9: pure functions, separation of concerns, and service boundary justified on MRM grounds.

8.  **Design documentation.** Sections 3–6: layered diagram, separation-of-concerns table, module inventory, interfaces, and schemas.

9.  **Data analysis.** Section 9: validation design and error-handling table.

10. **Testing.** Section 11: 644 tests, 95% coverage, integration scripts, and remediation plan.

11. **System analysis.** Section 4 and 8: end-to-end data flow and service-layer orchestration.
