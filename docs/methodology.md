# VoxFrontier Methodology

This document specifies, with equations, every estimator the pipeline runs.
It is intentionally concise but complete enough for peer review or a
methods section of a paper. All estimators are implemented in `src/voxfrontier/`
and covered by tests.

---

## 0. Notation

We observe $n$ livestream sessions (decision-making units, DMUs). For session $i$:

- inputs (resources consumed): $x_i \in \mathbb{R}^m_{++}$
- outputs (results produced): $y_i \in \mathbb{R}^s_{++}$
- voice features: acoustic measurements of the streamer's speech
- basic features: duration, price, traffic

---

## 1. Synthetic data generation (`data/synth.py`)

Let $U \sim \mathcal{LG}(\mu, \sigma)$ denote a log-normal, $\Gamma(k,\theta)$
a gamma draw. Marginals (calibrated to the study's descriptive statistics):

| variable | law |
|---|---|
| followers | $U(3.9, 1.05)$, clip $[0.5, 320]$ |
| duration | $\Gamma(2, 3.9) + 0.4$, clip $[0.4, 24]$ |
| viewers | $U(6.9, 1.5)$, clip $[10, 65000]$ |
| mean F0 | $\mathcal{N}(238, 52^2)$, clip $[128, 425]$ |
| HNR | $\mathcal{N}(9.2, 2.6^2)$, clip $[0.8, 16.5]$ |
| jitter | $3.55 - 0.14\,\text{HNR} + \varepsilon$, $\varepsilon\sim\mathcal{N}(0,0.35^2)$ |
| shimmer | $19.5 - 0.62\,\text{HNR} + \varepsilon$, $\varepsilon\sim\mathcal{N}(0,1.6^2)$ |
| speech rate | $3.4 + 0.006\,\text{F0} + \varepsilon$, $\varepsilon\sim\mathcal{N}(0,0.72^2)$ |

The latent outcome mechanism embeds **testable structure**:

$$
\text{conv} = f(\text{viewers}, \text{price}, \text{followers})
+ 0.055\,\Big|\tfrac{\text{F0}-295}{295}\Big|
+ 0.030\,\Big|\tfrac{\text{rate}-5.5}{5.5}\Big|
+ 0.045\,\tfrac{(\text{HNR}-11)_+}{5}
+ 0.035\,\tfrac{(2.45-\text{jitter})_+}{1.4}
+ \varepsilon
$$

i.e. U-shaped valleys in F0 and speech rate, threshold gains in HNR (above)
and jitter (below) — the exact phenomena the analysis stages must recover.
`sales = viewers · conv · lognormal noise`, all DEA variables $> 0$.

---

## 2. DEA efficiency (`dea/models.py`)

Input-oriented BCC/CCR. For DMU $o$:

$$
\min_{\theta,\lambda}\ \theta \quad
\text{s.t.}\quad \sum_j \lambda_j x_j \le \theta x_o,\quad
\sum_j \lambda_j y_j \ge y_o,\quad
\underbrace{\textstyle\sum_j \lambda_j = 1}_{\text{BCC (VRS) only}},\quad
\lambda \ge 0
$$

- **Model A** inputs: duration, price, viewers (no voice).
- **Model B** inputs: A + mean F0, HNR, jitter, speech rate (core model).
- **Model C** inputs: A + followers + full acoustic set.
- Outputs (all models): sales index, conversion %.
- Scale efficiency $= \theta_{CCR}/\theta_{BCC}$.
- **Super-efficiency** (Andersen–Petersen): re-solve with DMU $o$ removed
  from the reference set; scores $>1$ rank frontier units.

Non-finite / non-positive data are replaced by half the column's smallest
positive value (DEA requires strict positivity).

Backends: `scipy.optimize.linprog` (HiGHS) primary; `pulp` (CBC) optional;
`auto` falls back automatically. Equality of backends is unit-tested.

## 3. Exact Shapley attribution (`attribution/shapley.py`)

Partition inputs into $k$ groups $N=\{1..k\}$ (3 basic + 4 voice). The
characteristic function is mean BCC efficiency:

$$
v(S) = \frac{1}{|D|}\sum_i \theta^{BCC}_i\big(\text{inputs restricted to } S\big)
$$

$$
\phi_i = \sum_{S \subseteq N\setminus\{i\}}
\frac{|S|!\,(k-|S|-1)!}{k!}\,\big[v(S\cup\{i\}) - v(S)\big]
$$

Computed exactly ($2^k-1$ DEA solves, $k=7 \Rightarrow 127$). Reported
shares are $|\phi_i| / \sum_j |\phi_j|$, which sum to 100% by construction
(tested). Values are **always computed from data** — a regression test
fails if two different datasets ever produce identical values.

## 4. Tobit / censored regression (`attribution/tobit.py`)

**OLS-style (primary, as in the study).** Standardized regressors,
classical $t/p$ inference, $R^2$ and adjusted $R^2$.

**Two-limit Tobit MLE (secondary).** With limits $(L,U)=(0,1)$:

$$
\ell(\beta,\sigma) = \sum_{L<y_i<U} \log \phi\!\Big(\frac{y_i-x_i'\beta}{\sigma}\Big)
+ \sum_{y_i\le L} \log \Phi\!\Big(\frac{L-x_i'\beta}{\sigma}\Big)
+ \sum_{y_i\ge U} \log \Phi\!\Big(\frac{x_i'\beta-U}{\sigma}\Big)
$$

maximized with Nelder–Mead over $(\beta, \log\sigma)$.

## 5. Mediation (`attribution/tobit.py:mediation_analysis`)

Paths (all variables standardized): $c$: $X\to Y$; $a$: $X\to M$;
$b, c'$: $\{X,M\}\to Y$. Indirect effect $a\cdot b$ with Sobel $z$ and
percentile bootstrap CI (2000 resamples by default).

## 6. Non-linearity (`nonlinear/analysis.py`)

**Quadratic (U-shape).** $y = \beta_0 + \beta_1 x_s + \beta_2 x_s^2 + e$
($x_s$ standardized). $\beta_2$: t-test; nested F-test vs linear.
Extremum on the raw scale: $x^* = -\beta_1 x_{sd}/(2\beta_2) + x_{mean}$.
$\beta_2>0$ ⇒ U (valley), $\beta_2<0$ ⇒ ∩ (peak).

**Threshold (Hansen grid).** For each candidate $\gamma$ (10th–90th pct):

$$
y = \alpha + \beta_1 x\mathbb{1}(x\le\gamma) + \beta_2 x\mathbb{1}(x>\gamma) + \delta' Z + e
$$

$\hat\gamma = \arg\min SSE$; LR statistic $n\log(SSE_0/SSE_1)$ against the
no-threshold model; **fixed-X bootstrap p-value** (resample unrestricted
residuals under the null; `causality/inference.py:threshold_bootstrap_pvalue`).

**Quantile bands & subgroup effects.** Slopes within efficiency-quantile
bands and within category/gender strata, with classical t inference.

## 7. Double Machine Learning (`causality/dml.py`)

Partially linear model, one treatment at a time:
$Y = \theta D + g_0(X) + U$, $D = m_0(X) + V$.
Cross-fitted FWL (Chernozhukov et al., 2018): with $K$ folds, out-of-fold
residuals $\hat v_i = D_i - \hat m^{(-k(i))}(X_i)$,
$\hat u_i = Y_i - \hat g^{(-k(i))}(X_i)$,

$$
\hat\theta = \frac{\sum_i \hat v_i \hat u_i}{\sum_i \hat v_i^2},\qquad
\widehat{se} = \sqrt{\tfrac{1}{n}\,\widehat{\mathbb{V}}(\hat\psi)},\quad
\psi_i = \hat v_i(\hat u_i - \hat\theta \hat v_i)
$$

Nuisance learners: random forest (default), GBM, ridge, OLS. Heterogeneity:
$\hat\theta$ re-estimated within efficiency-quantile bands. OLS/FWL
benchmark reported alongside for transparency.

## 8. Uncertainty tooling (`causality/inference.py`)

- **Shapley bootstrap**: percentile CIs over $B$ resamples of sessions.
- **Conformal intervals**: split-conformal half-width $q$ = the
  $(\lceil (1-\alpha)(1+1/n_{cal})\rceil)$-quantile of calibration absolute
  residuals; interval $\hat y \pm q$ has $\approx 1-\alpha$ coverage
  (coverage verified by test on fresh draws).
- **Super-efficiency ranking**: see §2.

## 9. Counterfactual scenarios (`simulation/scenarios.py`)

GBR predictor $\hat f$ trained on $\{$basic + voice$\}\to$ efficiency
(5-fold CV $R^2$ reported). Six scenarios perturb the four voice variables:

| scenario | intervention |
|---|---|
| baseline | none |
| voice_to_efficient | all units → high-efficiency-group means |
| voice_to_valley | all units → U-valleys $\hat x^*$ from §6 |
| voice_plus10 | each unit pushed 20% further from its valley |
| full_optimal | per-category efficient-group means |
| conservative | efficient-group targets for $eff<0.8$ only |

Per-unit diagnostics: within ±10% of a valley → "escape valley"
(direction by where the reference lies); >20% from reference → up/down.

## 10. Reproducibility

- Global seed (`config/default.yaml`), deterministic generator.
- `output/manifest.json`: tool/dependency versions, Python/platform,
  SHA-256 of the input dataset and every result table, key scalars.
  `verify_manifest()` re-hashes locally to detect drift or tampering.
- CI: GitHub Actions on Ubuntu × Python 3.9/3.11/3.12 runs the full test
  suite **and** an end-to-end `vxf run-all`.

## References

- Charnes, Cooper, Rhodes (1978). *Measuring the efficiency of decision making units.* EJOR.
- Banker, Charnes, Cooper (1984). *Some models for estimating technical and scale inefficiencies.* Mgmt Sci.
- Shapley (1953). *A value for n-person games.*
- Hansen (1996, 2000). *Inference when a nuisance parameter is not identified under the null* / *Sample splitting and threshold estimation.*
- Chernozhukov, Chetverikov, Demirer, Duflo, Hansen, Newey, Robins (2018). *Double/debiased machine learning for treatment and structural parameters.* Econometrics Journal.
- Andersen, Petersen (1993). *A procedure for ranking efficient units in DEA.* Mgmt Sci.
- Lundberg, Lee (2017). *A unified approach to interpreting model predictions* (SHAP).
- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World* (conformal prediction).
