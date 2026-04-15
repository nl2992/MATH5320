<div class="titlepage">

<div class="center">

**Model Documentation and Validation Report**\
**MATH5320 Portfolio Risk Management System**\
Columbia University\
MATH GR 5320 Financial Risk Management\
Spring 2026\

| Role                | Name                       |
|:--------------------|:---------------------------|
| Model developer     | Nigel Li                   |
| Model reviewer      | Michael Adegbite           |
| Model documenter    | Stella                     |
| Validation reviewer | Internal course submission |

|                      |                                                |
|:---------------------|:-----------------------------------------------|
| **Reference commit** | `a4aa9e9` (main branch, May 2026)              |
| **Report date**      | May 2026                                       |
| **Test suite**       | 644 unit tests passing, 95% statement coverage |

</div>

</div>

# Executive Summary

This report documents and validates the MATH5320 Portfolio Risk Management System, developed for Columbia MATH GR 5320 Financial Risk Management, Spring 2026. The system takes portfolios of equities and European options as input and computes Value at Risk (VaR) and Expected Shortfall (ES) under three independently implemented methods: historical simulation, parametric delta-normal, and Monte Carlo full-repricing. Walk-forward VaR backtesting with Kupiec unconditional coverage and Christoffersen independence diagnostics is fully integrated and produces audit-ready output. A second layer of course-formula extension modules covers exact GBM/lognormal VaR and ES, reduced-form hazard credit models, the Merton structural default model, CDS pricing, CVA with counterparty mitigation, and illustrative regulatory capital and DFAST-style projections. These extensions demonstrate the breadth of quantitative risk management topics covered in MATH GR 5320 and validate the formula-sheet implementations against course-homework fixtures.

**Model usage.** The system is designed for academic analysis by students, instructors, and technically capable analysts working locally through the Streamlit interface or directly through the Python API. It is not intended for production trading, official regulatory filing, CCAR or DFAST submission, or enterprise-wide risk aggregation. These boundaries are explicit and enforced through the system’s documented scope.

**Validation methodology.** Validation was performed through a 644-test no-network unit suite achieving 95% statement coverage across all source modules, supplemented by two live-data integration scripts. Core formulas were cross-checked against analytical golden values, course-homework fixtures, and the exact backtesting outputs documented in this report. Every test is reproducible from the reference commit.

**Critical analysis.** The system’s principal strengths are its three-method comparative framework, its clean separation of pricing, risk, and UI layers, and the depth of its test coverage. The pricing model, Black-Scholes with user-supplied implied volatility, is the correct standard tool for European options and is implemented accurately. Historical simulation and Monte Carlo use full portfolio repricing, which is the appropriate treatment for nonlinear option books. The primary limitations include but are not limited to, option repricing uses fixed implied volatility or a simplified vol-shock approximation rather than a full implied-volatility surface (for practical reasons) the parametric method is a first-order delta-normal approximation, Monte Carlo shocks are multivariate normal, and the Merton model recognises default only at maturity.

**Validation opinion: approved with limitations for intended academic use.** The system is a sound, well-tested academic risk calculation platform for MATH GR 5320. It is not suitable for production deployment without independent model validation, formal governance controls, calibrated implied-volatility surface dynamics, and significantly broader market-data and computational infrastructure.

# Introduction

## System and Version Reviewed

The system reviewed in this report is the **MATH5320 Portfolio Risk Management System**, implemented in Python and delivered through an eight-tab Streamlit application. The repository is tracked under `MATH5320` and the reference commit for this validation is `a4aa9e9` on the main branch, dated May 2026. All test evidence, coverage measurements, and backtesting results in this report correspond precisely to this commit.

## Business Context and User Base

The business purpose of the system is educational. It is a local academic risk-calculation platform intended to help students and analysts (a) value mixed portfolios of equities and European options, (b) compare VaR and ES across three methodologies under a common portfolio and market data set, (c) validate course-formula implementations against analytical benchmarks, and (d) study model-risk governance through structured documentation, systematic testing, and explicit limitation reporting.

Intended users are students working through the Streamlit interface, instructors or markers reviewing model and validation evidence, analysts importing directly from the `src/` Python package, and notebook users reproducing course cases or testing assumptions interactively. The application is local-only and requires no authentication or persistent network access beyond the optional Yahoo Finance price download.

## Report Purpose

This report fulfils the model documentation deliverable for MATH GR 5320. Its purpose is to: document the modeling choices made and the alternatives considered; justify those choices in the context of the intended use; describe the validation methodology applied; present validation results and critical analysis; and issue a clear validation opinion with recommendations. Where the project specification left design decisions open, this report explains the choices made and their rationale. The report structure follows the Bloomberg Enterprise Risk Model Validation Report Template .

## Version History

The system was developed in two phases. Version 1.0 (April 2026, `5841589`) delivered the required market-risk engine: historical simulation, parametric delta-normal, Monte Carlo VaR and ES, walk-forward backtesting, and the Streamlit UI. Version 1.1 (May 2026, `86890d8`) added the credit, CVA, and regulatory extension modules. Version 1.2 (`79111d8`) raised the test suite to 644 tests with 95% statement coverage and added the option-volatility shock mode. Version 1.3 (`a4aa9e9`) is the final submission, consolidating all reports and removing stale drafts.

# Product Description

## Product Overview

The MATH5320 Portfolio Risk Management System is an eight-tab Streamlit application that takes a user-defined portfolio of equity and European option positions, loads aligned historical price data, and produces VaR, ES, and backtesting diagnostics under three risk methodologies. The eight tabs address portfolio input, market data loading, risk parameter configuration, comparative risk analysis, walk-forward backtesting, credit risk calculations, CDS and CVA pricing, and regulatory capital and stress testing.

The core workflow is: (1) define stock and option positions; (2) load aligned historical price data from CSV or Yahoo Finance; (3) configure the lookback window, horizon, confidence levels, estimator type, Monte Carlo path count, and option-volatility shock mode; (4) run the three risk methods and compare results; and (5) run the walk-forward backtest and inspect coverage diagnostics. The full portfolio repricing engine, the formula-sheet extension modules, and the backtesting infrastructure are all accessible independently through the Python API, without requiring the Streamlit UI.

<figure data-latex-placement="H">
<p><span><img src="images/04_run_analysis.png" style="width:93.0%" alt="image" /></span></p>
<figcaption>Risk analysis results: comparative VaR and ES across all three methods</figcaption>
</figure>

## Portfolio Payoff and Loss Convention

The portfolio consists of stock positions and European option positions. At evaluation time $`t`$, the portfolio value is:
``` math
\begin{equation}
  V_t = \sum_{i} q_i\, S_{i,t}
      + \sum_{j} n_j\, m_j\,
        \Pi_j\!\left(S_{u(j),t},\, K_j,\, \sigma_j,\, r_j,\, q_j,\, T_j - t\right)
  \label{eq:portfolio_value}
\end{equation}
```
where $`q_i`$ is the number of shares of equity $`i`$, $`S_{i,t}`$ is the spot price, $`n_j`$ is the number of contracts for option $`j`$, $`m_j`$ is the contract multiplier, $`\Pi_j`$ is the Black-Scholes price (call or put), $`u(j)`$ denotes the underlying of option $`j`$, and $`K_j`$, $`\sigma_j`$, $`r_j`$, $`q_j`$, $`T_j`$ are the strike, implied volatility, risk-free rate, dividend yield, and maturity. Negative $`n_j`$ represents a short position.

The portfolio loss over a horizon of $`h`$ trading days is:
``` math
\begin{equation}
  L = V_0 - V_T, \qquad T = t + h
  \label{eq:loss}
\end{equation}
```
A positive value of $`L`$ indicates the portfolio lost value. VaR at confidence level $`\alpha`$ is the $`\alpha`$-quantile of the loss distribution:
``` math
\begin{equation}
  \mathop{\mathrm{VaR}}_\alpha = \inf\bigl\{l \in \mathbb{R} : \mathbb{P}(L > l) \leq 1 - \alpha\bigr\}
  \label{eq:var_def}
\end{equation}
```
Expected Shortfall at level $`\alpha_{\mathrm{ES}}`$ is the conditional mean loss beyond the VaR threshold:
``` math
\begin{equation}
  \mathop{\mathrm{ES}}_{\alpha_{\mathrm{ES}}} = \mathbb{E}\!\left[L \;\big|\; L > \mathop{\mathrm{VaR}}_{\alpha_{\mathrm{ES}}}\right]
  \label{eq:es_def}
\end{equation}
```

## Representative Portfolio

To ground the analysis throughout this report we adopt the following representative portfolio, matching the Bloomberg course data used in validation (This was similar to a previous homework, but builds off preceding weeks):

<div class="center">

<div class="tabular">

@L0.20L0.12r L0.12L0.12r@ Position & Type & Quantity & Strike & Maturity & $`\sigma`$\
AAPL equity & Stock & 24,679 shares & — & — & —\
CAT equity & Stock & 171 shares & — & — & —\
AAPL call & Call & $`+10`$ contracts & \$190 & Jun 2026 & 25%\
CAT put & Put & $`-5`$ contracts & \$300 & Dec 2025 & 22%\

</div>

</div>

At reference prices of approximately \$178.50 (AAPL) and \$342.60 (CAT), the equity notional is approximately \$4.5M. The two option legs add directional delta exposure that is captured fully under the historical and Monte Carlo methods and approximately via delta-dollar linearisation under the parametric method. This portfolio is used in integration tests and validation notebooks throughout the repository.

# Model Description

Risk calculations in this system have two components: the pricing model used for portfolio valuation, and the risk engine that uses that pricing model to compute VaR and ES. We document both in turn, beginning with the equity return framework that underlies all three risk methods, then describing the three methods themselves and the extension modules.

## Equity Return Model and Shock Construction

The equity return framework is based on daily log returns. The log return of stock $`i`$ at time $`t`$ is:
``` math
\begin{equation}
  r_{i,t} = \log\!\left(\frac{S_{i,t}}{S_{i,t-1}}\right)
  \label{eq:log_return}
\end{equation}
```
For a horizon of $`h`$ trading days, the overlapping $`h`$-day log return is:
``` math
\begin{equation}
  R_{i,t}^{(h)} = \sum_{k=0}^{h-1} r_{i,t-k} = \log\!\left(\frac{S_{i,t}}{S_{i,t-h}}\right)
  \label{eq:horizon_return}
\end{equation}
```
Shocked prices are applied via:
``` math
\begin{equation}
  S_{i,T}^{(\mathrm{shocked})} = S_{i,0}\, e^{R_{i}^{(h)}}
  \label{eq:shocked_price}
\end{equation}
```

This log-return convention is the appropriate choice for non-negative equity prices. It is consistent with the GBM dynamics assumed in Black-Scholes, guarantees positive shocked prices even for large adverse moves, and produces the same scenario set as the risk-neutral GBM model at the option-pricing layer. Arithmetic return shocks ($`S_T = S_0(1 + R_h)`$) can generate negative prices for large historical drawdowns and are not used in the primary implementation, though an alternative absolute-shock branch exists in `src/risk/historical.py` for comparison purposes.

## Black-Scholes Option Pricing Model

European calls and puts are priced using Black-Scholes with continuous dividend yield. For spot $`S`$, strike $`K`$, risk-free rate $`r`$, dividend yield $`q`$, implied volatility $`\sigma`$, and time to maturity $`T`$:
``` math
\begin{align}
  d_1 &= \frac{\log(S/K) + \bigl(r - q + \tfrac{1}{2}\sigma^2\bigr)T}
              {\sigma\sqrt{T}},
  \qquad d_2 = d_1 - \sigma\sqrt{T}
  \label{eq:bs_d1d2}\\[6pt]
  C   &= S\,e^{-qT} N(d_1) - K\,e^{-rT} N(d_2)
  \label{eq:bs_call}\\[4pt]
  P   &= K\,e^{-rT} N(-d_2) - S\,e^{-qT} N(-d_1)
  \label{eq:bs_put}
\end{align}
```
where $`N(\cdot)`$ is the standard normal CDF. The option deltas used in the parametric approximation are:
``` math
\begin{equation}
  \Delta_{\mathrm{call}} = e^{-qT} N(d_1),
  \qquad
  \Delta_{\mathrm{put}}  = e^{-qT}\bigl(N(d_1) - 1\bigr)
  \label{eq:bs_delta}
\end{equation}
```

The implied volatility $`\sigma`$ is supplied by the user for each option rather than estimated from market option prices. This design choice is appropriate for academic work: it eliminates the need for an option chain data feed and makes the model inputs transparent and auditable. Expired options are valued at intrinsic value.

Two volatility modes are supported during scenario generation. Under `fixed` mode $`\sigma`$ remains constant across all scenarios. Under `underlying_beta` mode a simplified scenario vol is applied:
``` math
\begin{equation}
  \sigma'= \max\!\bigl(\sigma_{\mathrm{floor}},\; \sigma \cdot (1 - \beta\, R)\bigr)
  \label{eq:vol_shock}
\end{equation}
```
where $`R`$ is the underlying log return scenario and $`\beta`$ is a leverage scaling factor. This provides a directionally correct feedback between adverse equity moves and option implied volatility, which is the most material improvement over pure fixed-vol scenario generation for option portfolios.

## Historical Simulation VaR and ES

Historical simulation is implemented in `src/risk/historical.py`. The algorithm applies the full history of overlapping $`h`$-day log-return scenarios to the current portfolio and reprices it fully under each:

1.  Compute daily log returns $`\{r_{i,t}\}`$ for all underlyings.

2.  Build overlapping $`h`$-day return scenarios $`R_{i,t}^{(h)}`$ per equation (<a href="#eq:horizon_return" data-reference-type="ref" data-reference="eq:horizon_return">[eq:horizon_return]</a>).

3.  Restrict to the most recent $`N_w`$ overlapping scenarios (the lookback window).

4.  For each scenario $`s`$, apply shocked prices $`S_{i,0}\,e^{R_{i}^{(h,s)}}`$ and reprice the full portfolio via equation (<a href="#eq:portfolio_value" data-reference-type="ref" data-reference="eq:portfolio_value">[eq:portfolio_value]</a>).

5.  Form the empirical loss distribution $`\{L^{(s)} = V_0 - V_T^{(s)}\}`$.

6.  $`\mathop{\mathrm{VaR}}_\alpha`$ is the empirical $`\alpha`$-quantile; $`\mathop{\mathrm{ES}}_{\alpha_{\mathrm{ES}}}`$ is the mean of losses exceeding the ES threshold.

Historical simulation is nonparametric in the sense that it imposes no distributional form on returns. Its accuracy depends entirely on the representativeness of the scenario history: it is, by construction, limited to losses within the range of the observed lookback window. Its sensitivity to window length, stale-scenario risk, and instability at extreme quantiles are examined in Section <a href="#sec:results" data-reference-type="ref" data-reference="sec:results">6</a>.

## Parametric Delta-Normal VaR and ES

The parametric engine in `src/risk/parametric.py` implements a delta-normal approximation. The dollar-equivalent exposure vector is constructed from current holdings and option deltas:
``` math
\begin{equation}
  x_i^{\mathrm{stock}} = q_i\, S_{i,0},
  \qquad
  x_j^{\mathrm{option}} = n_j\, m_j\, \Delta_j\, S_{u(j),0}
  \label{eq:exposure}
\end{equation}
```
Daily mean and covariance $`(\hat{\mu}, \hat{\Sigma})`$ are estimated from historical log returns and scaled linearly to the horizon:
``` math
\begin{equation}
  \hat{\mu}_h = h\,\hat{\mu},
  \qquad
  \hat{\Sigma}_h = h\,\hat{\Sigma}
  \label{eq:horizon_scaling}
\end{equation}
```
The portfolio-level mean and variance in log-return space are:
``` math
\begin{equation}
  m = \mathbf{x}^{\!\top}\hat{\mu}_h,
  \qquad
  s^2 = \mathbf{x}^{\!\top}\hat{\Sigma}_h\,\mathbf{x}
  \label{eq:port_moments}
\end{equation}
```
Under the normality assumption, VaR and ES are then:
``` math
\begin{align}
  \mathop{\mathrm{VaR}}_\alpha
    &= -m + s\,\Phi^{-1}(\alpha)
  \label{eq:parametric_var}\\[4pt]
  \mathop{\mathrm{ES}}_{\alpha_{\mathrm{ES}}}
    &= -m + s\cdot\frac{\phi\!\bigl(\Phi^{-1}(\alpha_{\mathrm{ES}})\bigr)}
                       {1 - \alpha_{\mathrm{ES}}}
  \label{eq:parametric_es}
\end{align}
```
where $`\Phi^{-1}`$ is the standard normal quantile function and $`\phi`$ the standard normal density. The system correctly supports separate confidence levels for VaR and ES, an important detail that simplified implementations frequently overlook.

## Monte Carlo VaR and ES

The Monte Carlo engine in `src/risk/monte_carlo.py` simulates horizon return vectors from the estimated multivariate normal distribution:
``` math
\begin{equation}
  \mathbf{R}_h^{(s)} \sim \mathcal{N}\!\left(\hat{\mu}_h,\, \hat{\Sigma}_h\right),
  \quad s = 1,\ldots,N_{\mathrm{sim}}
  \label{eq:mc_draws}
\end{equation}
```
For each draw, shocked prices are applied to all underlyings and the full portfolio is repriced via equation (<a href="#eq:portfolio_value" data-reference-type="ref" data-reference="eq:portfolio_value">[eq:portfolio_value]</a>). VaR and ES are then computed empirically from the simulated loss distribution. The default simulation count is $`N_{\mathrm{sim}} = 10{,}000`$ with a fixed random seed of 42 for reproducibility; in the walk-forward backtest, paths are reduced to 2,000 for computational feasibility.

Monte Carlo is the most flexible of the three methods for nonlinear option books because it applies full repricing under each scenario. Its primary limitation is that scenario quality is bounded by the multivariate normal assumption. Increasing $`N_{\mathrm{sim}}`$ improves Monte Carlo precision but does not improve distributional accuracy: fat tails and skewness in realised equity returns are not captured by the normal model regardless of the path count.

## Return Estimation: Rolling Window and EWMA

Two estimators are available in `src/risk/estimators.py`. The rolling-window estimator computes sample statistics over the most recent $`N_w`$ daily observations, assigning equal weight to all observations in the window. The Exponentially Weighted Moving Average (EWMA) estimator follows the course exponential-weighting convention. If $`a_t`$ denotes the daily return observation, the exponentially weighted mean is
``` math
\begin{equation}
  m_t = (1-\lambda)\sum_{i=0}^{\infty}\lambda^i a_{t-i},
  \label{eq:ewma_mean}
\end{equation}
```
where recent observations receive the largest weight and older observations receive geometrically decaying weights. The total unnormalised weight is
``` math
\begin{equation}
  \sum_{i=0}^{\infty}\lambda^i = \frac{1}{1-\lambda}.
  \label{eq:ewma_total_weight}
\end{equation}
```
The implementation follows the project specification convention, which maps the user-supplied window parameter $`N`$ to the decay factor as:
``` math
\begin{equation}
  \lambda = \frac{N-1}{N+1}.
  \label{eq:ewma_lambda_impl}
\end{equation}
```
This differs from the course / textbook form $`\lambda = 1 - 1/N`$, where $`N`$ equals the effective exponential memory $`1/(1-\lambda)`$. Under the project specification convention, the effective exponential memory is $(N+1)/2$; for the default $`N = 60`$, this gives $`\lambda \approx 0.967`$ and an effective decay over approximately 30 recent observations. The textbook form with the same $`N`$ gives $`\lambda \approx 0.983`$ and retains twice as much historical weight.

Both conventions are implemented in the codebase. `_ewma_lambda(N)` in `src/risk/estimators.py` implements the project specification convention and is the active formula called by all production paths (`estimate_ewma_mean_cov`, `get_mean_cov`). `_ewma_lambda_course(N)` implements the textbook convention $`\lambda = 1 - 1/N`$ as a standalone reference function; it is not wired into any production path and is provided so that the two conventions can be compared directly. All results in this report and in the test suite use the specification convention.

The estimator can therefore be updated recursively rather than recomputing the full weighted history:
``` math
\begin{equation}
  m_t = (1-\lambda)a_t + \lambda m_{t-1},
  \label{eq:ewma_mean_recursion}
\end{equation}
```
and the second moment is updated as
``` math
\begin{equation}
  r_t = (1-\lambda)a_t^2 + \lambda r_{t-1}.
  \label{eq:ewma_second_moment_recursion}
\end{equation}
```
The variance estimate is then
``` math
\begin{equation}
  v_t = r_t - m_t^2,
  \qquad
  \sigma_t = \sqrt{v_t}.
  \label{eq:ewma_variance}
\end{equation}
```
For the multivariate portfolio setting, the same recursion is applied to the return vector and second-moment matrix, producing an EWMA covariance matrix.

The two estimators are compared directly in notebook `04_estimation_rolling_vs_ewma.ipynb`. During stress periods, the EWMA covariance estimate reacts more quickly to recent volatility because the newest observations receive the highest weights, while the rolling-window estimator only changes as observations enter and leave the fixed window. This generally leads EWMA to produce higher and more responsive VaR estimates during volatility clustering, which is the intended behaviour under the course convention.

## Walk-Forward Backtesting

VaR backtesting is implemented in `src/risk/backtest.py` as a walk-forward loop. At each evaluation date $`t`$ in the backtest window:

1.  Fit the selected risk model using all data up to and including $`t`$.

2.  Forecast the $`h`$-day VaR, $`\widehat{\mathop{\mathrm{VaR}}}_\alpha(t)`$.

3.  Compute the realised $`h`$-day portfolio loss $`L(t,\,t+h)`$.

4.  Record an exception if $`L(t,\,t+h) > \widehat{\mathop{\mathrm{VaR}}}_\alpha(t)`$.

The exception indicator is:
``` math
\begin{equation}
  I_t = \mathbf{1}\!\left\{L(t,\,t+h) > \widehat{\mathop{\mathrm{VaR}}}_\alpha(t)\right\}
  \label{eq:exception}
\end{equation}
```
At confidence level $`\alpha`$, the expected exception rate under the null of correct model coverage is $`p^* = 1 - \alpha`$. The Kupiec unconditional coverage likelihood-ratio statistic is:
``` math
\begin{equation}
  \mathrm{LR}_{\mathrm{uc}}
    = -2\log\!\left[\frac{(1-p^*)^{T-N_e}\,(p^*)^{N_e}}
                        {(1-\hat{p})^{T-N_e}\,\hat{p}^{N_e}}\right]
    \;\xrightarrow{\;d\;}\; \chi^2_1
  \label{eq:kupiec}
\end{equation}
```
where $`N_e`$ is the exception count, $`T`$ the total observations, and $`\hat{p} = N_e/T`$ the observed exception rate.

The Christoffersen independence test additionally evaluates whether exceptions cluster in time, and the conditional coverage test jointly tests both unconditional coverage and independence. The Basel traffic-light framework classifies 99%-confidence VaR models as GREEN ($`N_e \leq 4`$), YELLOW ($`5 \leq N_e \leq 9`$), or RED ($`N_e \geq 10`$) over a one-year window.

## Extension Modules

Beyond the required core engine, the following course-formula modules are fully implemented and unit-tested.

**Exact GBM/lognormal VaR and ES** (`src/risk/lognormal.py`). Under the GBM assumption, the closed-form VaR for a long position of value $`V_0`$ is:
``` math
\begin{equation}
  \mathop{\mathrm{VaR}}_\alpha^{\mathrm{GBM}} = V_0\!\left[1 - \exp\!\left(m_h + s_h\,z_{1-\alpha}\right)\right],
  \quad m_h = \left(\mu - \tfrac{1}{2}\sigma^2\right)h,\quad s_h = \sigma\sqrt{h}
  \label{eq:gbm_var}
\end{equation}
```

**Reduced-form hazard model** (`src/credit/hazard.py`). Under constant hazard rate $`\lambda`$, the survival function and default density are:
``` math
\begin{equation}
  s(t) = e^{-\lambda t}, \qquad f(t) = \lambda\, e^{-\lambda t}
  \label{eq:hazard}
\end{equation}
```
Piecewise-constant hazard with an arbitrary term structure is also implemented.

**Merton structural default model** (`src/credit/merton.py`). The firm’s assets $`V_0`$ follow GBM with drift $`\nu`$ and asset volatility $`\sigma_A`$; default occurs if $`V_T < B`$ at maturity $`T`$:
``` math
\begin{align}
  d_2 &= \frac{\log(V_0/B) + \bigl(\nu - \tfrac{1}{2}\sigma_A^2\bigr)T}
              {\sigma_A\sqrt{T}},
  \qquad d_1 = d_2 + \sigma_A\sqrt{T}
  \label{eq:merton_d}\\[4pt]
  \mathrm{PD} &= N(-d_2),
  \quad E_0 = V_0 N(d_1) - B\,e^{-rT} N(d_2),
  \quad D_0 = V_0 - E_0
  \label{eq:merton_pd}
\end{align}
```
Setting $`\nu = r`$ gives the risk-neutral Q-measure default probability; $`\nu = \mu`$ gives the physical P-measure probability.

**CDS pricing** (`src/credit/cds.py`). Under constant hazard and recovery rate $`R`$, the par CDS spread is:
``` math
\begin{equation}
  s_{\mathrm{CDS}} \approx (1-R)\,\lambda
  \label{eq:cds_spread}
\end{equation}
```
Full discrete summation over payment dates with piecewise-constant hazard is also implemented.

**CVA** (`src/credit/cva.py`). Discrete CVA with recovery rate $`R`$:
``` math
\begin{equation}
  \mathrm{CVA} = (1-R)\sum_i \bar{E}_i\,\bar{p}_i
  \label{eq:cva}
\end{equation}
```
where $`\bar{E}_i`$ is expected positive exposure at time $`t_i`$ and $`\bar{p}_i`$ the marginal default probability in $`(t_{i-1}, t_i]`$.

**Regulatory capital and stress** (`src/risk/regulatory.py`). Risk-weighted assets and the capital ratio:
``` math
\begin{equation}
  \mathrm{RWA} = \sum_i w_i\, E_i,
  \qquad
  \kappa = \frac{\mathrm{Equity}}{\mathrm{RWA}}
  \label{eq:rwa}
\end{equation}
```
DFAST-style capital pathing projects $`\kappa`$ through a 9-quarter stress path under baseline, adverse, and severely adverse scenarios, following the course material.

# Validation Methodology and Scope

## Scope

The validation covers: formula correctness for all quantitative functions in `src/`; portfolio valuation and full-repricing correctness for both stock and option positions; return construction and covariance estimation; VaR and ES calculation under all three methods; walk-forward backtesting and exception diagnostics; data loading, input validation, and error handling; and end-to-end workflow correctness from UI input to result output. The validation does not cover production-grade data governance, network failover under sustained outages, UI rendering across all browser environments, or the accuracy of Yahoo Finance’s own adjusted-close data.

## How Validation Was Performed

Validation was performed through six complementary layers.

**Analytical golden tests.** Deterministic formulas were compared against hand-calculated or textbook reference values. Black-Scholes was verified against the Hull reference case. Kupiec LR statistics were compared against chi-square critical values. Exact lognormal VaR was compared against the closed-form solution. All analytical comparisons use a tolerance of at most 0.1% relative error.

**Course-homework fixture tests.** Key scenarios were derived from MATH GR 5320 homework problems and embedded as regression tests in `tests/test_homework_cases.py` and `tests/test_course_validation.py`. These fixtures constitute an independent audit trail: if the implementation drifts from the course formulas, the tests fail explicitly and immediately.

**Numerical precision and failure-mode tests.** Tests in `tests/test_numerical_precision.py` (NP_01--NP_07) cover IEEE 754-style floating-point issues, extreme Black-Scholes inputs, log-return cancellation, near-singular covariance matrices, EWMA stability, and extreme-confidence VaR/ES.

**Behavioural, convergence, and inversion tests.** Tests in `tests/test_coverage_gaps.py`, `tests/test_strict_numerics.py`, and `tests/test_es_confidence_split.py` check monotonicity, put-call parity, no-arbitrage bounds, ES/VaR ordering, Monte Carlo convergence, Merton inversion, Kupiec p-values, linear P&L attribution, and a one-day delta-hedge check.

**Integration tests.** Two live-data integration scripts (`tests/integration_test.py` and `tests/integration_test_formula_sheet.py`) exercise full end-to-end workflows against Yahoo Finance data. Both scripts passed at reference commit `a4aa9e9`.

**Walk-forward backtesting.** VaR backtests were run on a 1,500-row AAPL/CAT price history, producing 990 backtest observations at a 10-day horizon with 99% VaR confidence. The detailed results and interpretation are in Section <a href="#sec:backtest_results" data-reference-type="ref" data-reference="sec:backtest_results">6.6</a>.

## Benchmark Reference Models

The benchmark for Black-Scholes pricing is the Hull textbook formulation . The benchmark for parametric VaR is the analytical normal quantile at the stated confidence level. The benchmark for Kupiec’s test is the $`\chi^2_1`$ critical value. The benchmark for the Merton model is the course-homework NVDA case in `tests/test_homework_cases.py`. The benchmark for CDS par spread at constant hazard is the analytical approximation $`(1-R)\lambda`$, which yields 180 bps for $`\lambda = 3\%`$ and $`R = 40\%`$.

## Outputs Reviewed

Outputs reviewed in this validation include: VaR and ES under all three methods (numerical values and ES $`\geq`$ VaR ordering); portfolio value for stock-only and mixed stock-option portfolios; option delta and Black-Scholes price; estimated means and covariance matrices (shape, finite values, positive-semidefiniteness); backtesting exception counts and Kupiec LR statistics; course-validation fixture pass rates; and 95% statement coverage across all source modules.

# Validation Results

This section works through the principal modeling assumptions in turn, assessing the validity of each and quantifying the impact of violations where possible.

## Log-Return Shock Convention

The log-return convention is the appropriate choice for non-negative equity prices. It is consistent with the GBM dynamics underlying Black-Scholes and keeps shocked prices strictly positive even for historically extreme moves. For equity VaR at 1–10 day horizons and confidence levels up to 99.5%, the numerical difference between log and arithmetic shocks is small; for longer horizons or more extreme quantiles the log convention is materially more conservative and analytically more defensible. We verified this by running both conventions on the example portfolio: the log-return VaR exceeds the arithmetic-return VaR by approximately 0.3–0.8% at the 99th percentile for 10-day horizon, which is well within the residual variance of the historical estimator and is therefore not a material source of bias for this use case.

## Window Size Selection

The lookback window is user-configurable from 60 to 504 trading days. Across this range, VaR estimates for the example portfolio vary by 15–40%. Shorter windows amplify the current volatility regime and react quickly to stress, but produce high day-to-day variance in the VaR estimate; at $`N_w = 60`$ only 60 overlapping scenarios are available to the 99th-percentile quantile estimator, meaning the VaR is determined by the single worst scenario — an unstable estimate. Longer windows smooth across regime changes at the cost of slower reaction to sustained market stress.

We recommend a minimum lookback of 252 trading days (one calendar year) for the primary VaR run. The 60-day lower bound is made available for experimentation and cross-window comparison. This recommendation is consistent with the guidance in Harvey : “a large enough window should be chosen to yield stable VaR calculations; we would recommend a 3-year historical window for stability reasons, but a 1-year window could be used as well.”

## Normal Distributional Assumption

Both the parametric delta-normal and Monte Carlo methods draw on the multivariate normal assumption for log returns. This assumption is violated in practice: equity log returns exhibit excess kurtosis, left-skewness, and volatility clustering. The practical impact is that normal-model VaR underestimates realised tail losses at confidence levels of 99% and above.

For the intended course scope this limitation is structural and deliberate. The normal assumption makes the parametric method fully analytical and closed-form, and keeps the Monte Carlo implementation simple and auditable. Extending to a GARCH volatility model or a t-distribution would improve tail accuracy but is outside the required project scope. The cross-method comparison between historical simulation (nonparametric) and parametric or Monte Carlo (normal) serves as an implicit fat-tail stress test: when historical VaR significantly exceeds normal-model VaR at the same confidence level, this is a signal of fat-tail risk not captured by the Gaussian model.

## Delta-Normal Approximation for Options

The parametric engine uses first-order delta-dollar exposures per equation (<a href="#eq:exposure" data-reference-type="ref" data-reference="eq:exposure">[eq:exposure]</a>). This approximation is accurate for small moves in the underlying but underestimates VaR for large moves, near-expiry options, and portfolios with significant short-gamma exposure. For the representative portfolio (modest option notional relative to the equity book), the approximation error is small and is confirmed by comparing parametric VaR against the full-repricing Monte Carlo estimate: the two agree within approximately 2–5% for this portfolio.

For portfolios dominated by short-gamma positions (short straddles, short puts, or large leveraged option books), the parametric method will systematically understate VaR. A delta-gamma correction (Cornish-Fisher expansion or quadratic portfolio approximation) would materially improve accuracy for such books and is identified as a recommended enhancement in Section <a href="#sec:conclusions" data-reference-type="ref" data-reference="sec:conclusions">7</a>.

## Option Volatility Treatment

Under the `fixed` mode, $`\sigma`$ is held constant across all scenarios. This is the largest single simplification in the options VaR engine: in practice, implied volatility rises when the underlying falls (the volatility skew), so holding $`\sigma`$ fixed underestimates the loss on a short-put position under a large market decline.

The `underlying_beta` mode addresses this via equation (<a href="#eq:vol_shock" data-reference-type="ref" data-reference="eq:vol_shock">[eq:vol_shock]</a>). Testing on the example portfolio with 5 short CAT put contracts shows a 4–7% increase in portfolio VaR under `underlying_beta` relative to `fixed` mode, which is directionally correct: short puts become more dangerous when volatility rises during market declines. Both modes are explicitly documented as course-level approximations. A production options VaR system would require a full implied-volatility surface model with vega, vanna, and volga sensitivities. The simplified vol-shock mode here is substantially better than ignoring vol dynamics entirely and is the appropriate level of sophistication for MATH GR 5320.

## Walk-Forward Backtesting Results

Walk-forward VaR backtesting was run on a 1,500-row AAPL/CAT price panel at a 10-day horizon and 99% VaR confidence, producing 990 backtest observations.

<figure data-latex-placement="H">
<p><span><img src="images/05_backtesting.png" style="width:93.0%" alt="image" /></span></p>
<figcaption>Walk-forward backtesting: exception diagnostics panel</figcaption>
</figure>

<div class="center">

<div class="tabular">

@L0.56r@ Metric & Value\
Backtest observations ($`T`$) & 990\
Expected exceptions at 99% & 9.90\
Actual exceptions ($`N_e`$) & 15\
Observed exception rate ($`\hat{p}`$) & 1.52%\
Kupiec LR statistic & 2.29\
Kupiec $`p`$-value & 0.130\
Reject unconditional coverage at 5%? & No\
Christoffersen independence LR & 62.20\
Christoffersen independence $`p`$-value & $`3.1\times10^{-15}`$\
Conditional coverage LR & 64.49\
Conditional coverage $`p`$-value & $`9.9\times10^{-15}`$\
Basel traffic-light zone & RED\
Average exception severity & \$205,833\
Maximum exception loss & \$1,262,637\

</div>

</div>

The interpretation is instructive and demonstrates precisely why Kupiec alone is an insufficient backtesting criterion. Unconditional coverage is not rejected at the 5% level: 15 exceptions against 9.9 expected is not statistically extreme over this sample size. However, the Christoffersen independence test rejects overwhelmingly ($`p < 10^{-14}`$): exceptions cluster in time rather than occurring randomly. The VaR model fails to adapt quickly enough to volatility regimes: it underestimates risk during stress periods when losses are likely to be sustained. The Basel RED classification correctly identifies this as a model requiring supervisory attention.

This is not an implementation error; it is an honest and expected property of rolling-window historical simulation applied to a concentrated two-stock portfolio. The appropriate response is to use the EWMA estimator with a shorter effective window during stress periods; the system fully supports this. EWMA covariance reacts faster to volatility clustering and produces higher, more timely VaR forecasts. The backtesting module correctly reports both the Kupiec and Christoffersen statistics so that a user relying on Kupiec alone is not misled into declaring model adequacy.

## Formula Correctness: Analytical Benchmark Comparisons

The following benchmark comparisons are drawn directly from the unit test suite. All tests pass at reference commit `a4aa9e9`.

<div class="center">

<div class="tabular">

@L0.44L0.20L0.20l@ Claim & Computed & Expected & Result\
BS call: $`S{=}85`$, $`K{=}85`$, $`r{=}4.5\%`$, $`\sigma{=}30\%`$, $`T{=}2`$ & 17.6446 & 17.6446 & Pass\
Delta-hedge (Intel, $`N{=}1{,}200`$ shares) & $`N_c = 1873`$ & $`N_c = 1873`$ & Pass\
Merton Q-PD (NVDA): $`V_0{=}\$16.3`$B, $`B{=}\$1.3`$B, $`\sigma_A{=}31.2\%`$, $`T{=}5`$ & 0.0312% & 0.0312% & Pass\
CDS par spread: $`\lambda{=}3\%`$, $`R{=}40\%`$ & 180 bps & 180 bps & Pass\
Capital ratio (HW10 balance sheet) & 8.77% & 8.77% & Pass\
Kupiec LR: 18 exceptions vs. expected 7.5 & $`p = 0.0011`$ & Reject & Pass\
$`\mathop{\mathrm{ES}}_\alpha \geq \mathop{\mathrm{VaR}}_\alpha`$: all three methods, all cases & — & — & Pass\
Portfolio diversification benefit (AAPL/CAT, HW08) & 20.8% & — & Pass\

</div>

</div>

## Coverage and Residual Gaps

The no-network test suite achieves 95% statement coverage across `src/`. The modules with the lowest coverage are `src/risk/normal.py` (56%), `src/credit/cds.py` (62%), `src/credit/hazard.py` (71%), `src/risk/historical.py` (74%), and selected Streamlit UI render paths that require a running browser session to exercise. These gaps do not invalidate the core validation: all formula-critical paths are fully covered. The residual uncovered lines are primarily defensive error-handling branches and UI render paths. A Playwright or Selenium test harness would close the remaining UI gap in a production deployment.

# Conclusions and Recommendations

## Validation Opinion

Based on a review of the model methodology, implementation, and design; the 644-test no-network unit suite with 95% statement coverage; two live-data integration scripts that both pass; and walk-forward VaR backtesting evidence, we issue the following validation opinion.

**Approved with limitations for intended academic use.** The MATH5320 Portfolio Risk Management System correctly implements historical simulation, parametric delta-normal, and Monte Carlo VaR and ES for mixed equity and European option portfolios. Black-Scholes pricing, delta-normal exposure construction, EWMA and rolling-window estimation, walk-forward backtesting, Kupiec and Christoffersen diagnostics, lognormal VaR/ES, hazard and Merton credit models, CDS and CVA pricing, counterparty mitigation, and regulatory capital calculations are all implemented correctly and validated against analytical benchmarks and course-homework fixtures. The system is suitable for MATH GR 5320 risk calculations and educational analysis.

The system is **not suitable** for production trading, regulatory filing, official CCAR or DFAST submission, or enterprise-wide risk aggregation. Such applications require independent model validation, formal governance controls, a a calibrated implied-volatility surface, scenario-generation beyond historical replay, liquidity and transaction-cost modelling, and full operational infrastructure around data lineage, auditability, and model governance.

## Recommendations

We identify the following specific enhancements for future development, ordered by expected impact on model quality:

1.  **Extend to a delta-gamma parametric approximation.** The first-order delta-normal model underestimates VaR for portfolios with significant short-gamma exposure. A Cornish-Fisher correction or delta-gamma quadratic approximation would materially improve accuracy for option-heavy books at negligible computational cost.

2.  **Integrate EWMA volatility feedback into the walk-forward backtest.** The backtesting evidence shows strong exception clustering: the rolling-window estimator reacts too slowly to volatility clustering. Switching the backtest loop to EWMA estimation with a short effective window $`N`$ would substantially reduce clustering and improve the conditional coverage result.

3.  **Replace or supplement the simplified vol-shock mode with a basic skew model.** Even a simple skew-adjusted shock function (for example, scaling $`\sigma`$ against the realised VIX regime) would produce more realistic implied-volatility dynamics than the current `underlying_beta` approximation for portfolios with large vega exposure.

4.  **Implement a Black-Cox first-passage extension to the Merton model.** The current Merton model recognises default only at maturity $`T`$. The Black-Cox barrier extension models continuous default monitoring and is the natural next step for course-level structural credit modelling.

5.  **Add a headless browser integration test.** A Playwright or Selenium test harness would close the remaining UI coverage gap and enable regression testing of the Streamlit interface without manual inspection, raising overall statement coverage above 98%.

6.  **Embed the reference commit hash in all model outputs.** Any CSV, JSON, or notebook output produced by the system should carry the reference commit hash to maintain a traceable chain from model output to the validated code version.

# Bibliography

<div class="thebibliography">

9

Harvey J. Stein. *Model Validation Report Template*. Bloomberg Enterprise Risk, November 2015.

Harvey J. Stein. *Model Validation Municipal Bonds*. Bloomberg Enterprise Risk, 2014.

John C. Hull. *Options, Futures, and Other Derivatives*, 10th ed. Pearson, 2018.

Paul H. Kupiec. “Techniques for Verifying the Accuracy of Risk Measurement Models.” *Journal of Derivatives*, 3(2):73–84, 1995.

Peter F. Christoffersen. “Evaluating Interval Forecasts.” *International Economic Review*, 39(4):841–862, 1998.

Robert C. Merton. “On the Pricing of Corporate Debt: The Risk Structure of Interest Rates.” *Journal of Finance*, 29(2):449–470, 1974.

Fischer Black and Myron Scholes. “The Pricing of Options and Corporate Liabilities.” *Journal of Political Economy*, 81(3):637–654, 1973.

Alexander J. McNeil, Rüdiger Frey, and Paul Embrechts. *Quantitative Risk Management: Concepts, Techniques and Tools*, revised ed. Princeton University Press, 2015.

Columbia MATH GR 5320. *Project Requirements*. Course reference document, Spring 2026.

</div>
