# Changelog

All notable changes to VoxFrontier are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versioning follows [SemVer](https://semver.org/).

## [1.0.0] - 2026-09-04

First stable release — production-grade engineering hardening.

### Added
- Release engineering: dynamic single-source versioning (`_version.py` via
  setuptools `attr:`), PEP 561 `py.typed` marker, Makefile, Dockerfile
  (non-root, OCI-labelled) and a VS Code dev container.
- Community health: `SECURITY.md` (private vulnerability reporting, privacy
  incident channel), `CODE_OF_CONDUCT.md` (Contributor Covenant 2.1),
  `SUPPORT.md`, `FUNDING.yml`, issue templates (bug/feature + config) and a
  PR template with methodology & privacy checklists.
- CI/CD: split lint / privacy-audit / test / CodeQL jobs; privacy audit
  scans **git history** (every blob) for personal-data patterns and rejects
  any tracked Excel file; multi-OS test matrix (Ubuntu + macOS + Windows
  smoke); `release.yml` tag-driven build with twine check, clean-venv wheel
  smoke test, GitHub Release artefacts and optional OIDC PyPI publishing;
  Dependabot (pip + actions).
- Docs: `docs/user_guide.md` (EN/中文), `docs/faq.md`, `docs/README.md`
  index, `examples/custom_analysis.py` library-usage example (executed and
  verified).
- Tooling config: ruff (E/W/F/I/B/UP/SIM/C4), coverage (branch, per-line
  excludes), pre-commit (formatting + **local-only** privacy hook — no code
  leaves the machine).

### Changed
- Version now single-sourced from `voxfrontier._version` (`1.0.0`).
- Classifiers: Development Status → Production/Stable, added
  `Typing :: Typed`, `Operating System :: OS Independent`.
- Project URLs expanded (Documentation / Changelog / Issue Tracker).
- CI lint is now blocking (previously `--exit-zero`).

## [0.2.0] - 2026-09-04

VoxFrontier is the new name of the project formerly released as
`livestream-voice-efficiency`. Python package: `voxfrontier`; CLI: `vxf`.

### Added
- **Causality stage**: cross-fitted Double Machine Learning (Neyman-orthogonal,
  RF/GBM nuisances) for voice effects on efficiency, with FWL/OLS benchmark
  and per-band heterogeneity.
- **Inference upgrades**: Shapley bootstrap confidence intervals,
  Hansen-style fixed-X bootstrap p-values for threshold regressions,
  split-conformal prediction intervals, super-efficiency frontier ranking.
- **Reproducibility manifest**: `output/manifest.json` records tool and
  dependency versions, dataset SHA-256, and SHA-256 of every result table;
  `verify_manifest()` detects drift/tampering.
- Executive 2×2 dashboard figure (DML forest plot, Shapley, U-shape, scenarios).
- `docs/methodology.md`: full mathematical specification of every estimator.
- Governance: `CITATION.cff`, this changelog, contributing guide.
- CI: coverage reporting and lint (ruff) in addition to tests + smoke run.
- 5 new test modules sections (31 tests total): DML recovery under
  confounding, conformal coverage, manifest round-trip/tamper detection,
  super-efficiency ranking, DML-vs-OLS comparison.

### Fixed
- Super-efficiency LP indexing bug (peer set vs evaluated unit mix-up).
- Threshold-regression sorting with NaN scores.
- `subgroup_effects` silently dropping all rows when the grouping column
  was non-numeric.

### Changed
- Package/CLI renamed (`voxfrontier` / `vxf`); version bumped to 0.2.0.

## [0.1.0] - 2026-09-04

Initial open-source release (as `livestream-voice-efficiency`).

- Synthetic data generator calibrated to the research study's structure.
- DEA (CCR/BCC, three input specifications, HiGHS primary + PuLP fallback).
- Exact Shapley decomposition (data-driven, anti-caching regression test).
- Random-forest importance, optional SHAP, Tobit OLS + MLE, mediation.
- Quadratic U-shape tests, grid-searched threshold regression, quantile
  bands, subgroup interactions.
- Six-scenario counterfactual simulation + per-unit diagnostics.
- CLI (`voxfrontier synth` / `voxfrontier run-all`), bilingual README, MIT license,
  GitHub Actions CI, 26 tests.
