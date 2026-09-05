# User Guide

**English** | [中文](#中文)

This guide walks through VoxFrontier's workflow, configuration options, and
the library API. For the mathematics behind each estimator see
[`methodology.md`](methodology.md).

## 1. Installation

```bash
git clone https://github.com/GOOD-123-CPU/voxFrontier.git
cd voxFrontier
pip install -e .            # core (scipy HiGHS backend, no extras)
pip install -e ".[pulp]"    # optional exact MILP backend (CBC)
pip install -e ".[shap]"    # optional SHAP plots
pip install -e ".[dev]"     # dev tools (pytest, ruff, pre-commit)
```

Python 3.9–3.12 on Windows / macOS / Linux. No solver binaries beyond the
bundled wheels are required; the default backend is pure `scipy`.

## 2. The five-minute tour

```bash
vxf synth       # regenerate data/synthetic/synthetic_sessions.csv (seeded)
vxf run-all     # full pipeline, ~1 min
```

`run-all` executes, in order:

1. **Synthesis** — deterministic synthetic dataset (unless you point
   `paths.data_dir` at your own licensed CSV; see §3).
2. **DEA** — input-oriented CCR/BCC across three input specs + scale
   efficiency + super-efficiency ranking (`dea_model_summary.csv`,
   `dea_efficiency_by_unit.csv`, `super_efficiency_ranking.csv`).
3. **Attribution** — exact Shapley over input groups (with bootstrap CIs),
   random-forest importance, Tobit OLS + MLE, mediation.
4. **Non-linearity** — quadratic U-tests, threshold regression with
   Hansen-style bootstrap p-values, quantile bands, subgroup effects.
5. **Causality** — cross-fitted DML per voice variable with 95% CIs,
   FWL/OLS benchmark, efficiency-band heterogeneity (`dml_effects.csv`).
6. **Inference** — split-conformal prediction intervals for scenario output
   (`conformal_intervals.csv`).
7. **Simulation** — six counterfactual scenarios + per-unit diagnostics.
8. **Figures** — analytical figures + the 2×2 executive dashboard.

Every run writes `output/manifest.json` — SHA-256 fingerprints of the input
data and every table, plus the tool versions. Verify afterwards:

```bash
make audit      # or: python -c "from voxfrontier.utils.manifest import verify_manifest; ..."
```

## 3. Using your own data (licensing reminder)

VoxFrontier ships **synthetic data only**. To analyse your own dataset you
must hold the rights to it. Point the config at a local CSV that mirrors the
synthetic schema (`voxfrontier/data/schema.py` is the contract):

```yaml
# my-config.yaml
paths:
  data_dir: "data/local"      # NOT committed; .gitignore protects it
data:
  n_units: 207
```

```bash
vxf run-all --config my-config.yaml
```

**Never commit real personal data** — the CI privacy-audit job blocks
suspicious patterns, but the primary responsibility is yours.

## 4. Configuration reference (abridged)

| key | default | meaning |
|---|---|---|
| `seed` | 42 | global RNG seed (numpy / random / sklearn) |
| `data.n_units` | 207 | synthetic sample size |
| `dea.backend` | `auto` | `scipy` (HiGHS) \| `pulp` (CBC) |
| `attribution.shapley_max_vars` | 10 | guard rail on exact Shapley cost |
| `attribution.rf_trees` | 300 | random-forest size |
| `attribution.use_shap` | true | optional SHAP (skips if not installed) |
| `plot.language` | `en` | `en` portable \| `zh` CJK with font fallback |
| `plot.dpi` | 150 | figure resolution |

## 5. Library API sketch

```python
from voxfrontier.config import Config, set_seed
from voxfrontier.data.synth import generate_synthetic
from voxfrontier.dea.models import run_three_models
from voxfrontier.attribution.shapley import shapley_group_decomposition
from voxfrontier.causality.dml import dml_partially_linear

cfg = Config.load()
set_seed(cfg.seed)

df = generate_synthetic(cfg.n_units, cfg.seed)
res = run_three_models(df, specs, outputs, cfg.dea_backend)
df["bcc_efficiency"] = res["B"]["bcc"]

dml = dml_partially_linear(
    df, treatments=["mean_f0"], controls=[...],
    outcome="bcc_efficiency", n_folds=5, learner="gbm", seed=cfg.seed,
)
```

Each module is importable and independently tested — see
`docs/methodology.md` for what each function estimates and its assumptions.

## 6. Troubleshooting

| symptom | fix |
|---|---|
| `scipy` LP fails / backends disagree | `pip install ".[pulp]"`, set `dea.backend: "pulp"` |
| Chinese labels show as boxes | install a CJK font (e.g. `fonts-noto-cjk` on Debian/Ubuntu) or keep `plot.language: en` |
| `use_shap` warns | optional; `pip install ".[shap]"` or set `false` |
| manifest verification fails | outputs were modified after the run; re-run `vxf run-all` |
| Py3.9 wheel resolution slow | fine — constraints are wide (`numpy<3` etc.) |

---

# 中文（用户指南）

## 1. 安装

```bash
pip install -e .            # 核心功能（scipy HiGHS 后端）
pip install -e ".[pulp]"    # 可选：精确 MILP 后端（CBC）
pip install -e ".[shap]"    # 可选：SHAP 图
pip install -e ".[dev]"     # 开发工具链
```

支持 Python 3.9–3.12，Windows / macOS / Linux。默认后端纯 `scipy`，无需额外求解器。

## 2. 五分钟上手

```bash
vxf synth       # 重新生成合成数据（固定种子）
vxf run-all     # 完整流水线，约 1 分钟
```

依次执行：合成数据 → DEA（CCR/BCC 三方案 + 超效率）→ 归因（精确 Shapley +
Bootstrap 置信区间、随机森林、Tobit、中介）→ 非线性（U 型、门槛 + Hansen
Bootstrap p 值、分位数带、亚组）→ 因果（交叉拟合 DML + FWL 基准）→ 共形预测
区间 → 六情景模拟 → 图表与仪表盘。

每次运行写出 `output/manifest.json`（数据与全部结果表的 SHA-256 指纹），
可用 `make audit` 校验结果未被篡改。

## 3. 接入自有数据（许可提醒）

本仓库**只提供合成数据**。分析自有数据集前请确认您拥有相应权利：

```yaml
# my-config.yaml
paths:
  data_dir: "data/local"   # 不会被提交；.gitignore 已保护
```

```bash
vxf run-all --config my-config.yaml
```

**切勿提交真实个人数据**——CI 隐私扫描会拦截可疑模式，但首要责任在使用者。

## 4. 配置速查

| 键 | 默认 | 含义 |
|---|---|---|
| `seed` | 42 | 全局随机种子 |
| `data.n_units` | 207 | 合成样本量 |
| `dea.backend` | `auto` | `scipy`(HiGHS) 或 `pulp`(CBC) |
| `attribution.shapley_max_vars` | 10 | 精确 Shapley 的成本护栏 |
| `plot.language` | `en` | `en` 跨平台 / `zh` 中文（自动字体回退） |

## 5. 库 API 速览

各模块均可独立导入、独立测试（DEA / Shapley / Tobit / DML / 门槛 / 共形区间），
数学定义与假设见 `docs/methodology.md`。

## 6. 常见问题

| 现象 | 处理 |
|---|---|
| LP 求解失败 / 后端不一致 | `pip install ".[pulp]"` 并设 `dea.backend: "pulp"` |
| 中文标签显示为方块 | 安装 CJK 字体（如 `fonts-noto-cjk`）或保持 `en` |
| manifest 校验失败 | 输出在运行后被改动过，重跑 `vxf run-all` |
