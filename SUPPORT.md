# Support

Thanks for using VoxFrontier! Here is how to get help efficiently.

## Where to ask

| Need | Channel |
| ---- | ------- |
| Bug report (reproducible) | [GitHub Issues — Bug report](.github/ISSUE_TEMPLATE/bug_report.md) |
| Feature proposal | [GitHub Issues — Feature request](.github/ISSUE_TEMPLATE/feature_request.md) |
| Usage question / discussion | GitHub Discussions (once enabled) or Issues with the `question` label |
| Security / privacy issue | **Never in public** — see [SECURITY.md](SECURITY.md) |

## Before opening an issue

1. **Search existing issues** — your question may already be answered.
2. **Reproduce with synthetic data**: run `vxf synth && vxf run-all` with the
   default config. Issues reported against the default synthetic pipeline are
   far easier to diagnose.
3. **Include the run manifest** (`output/manifest.json`) — it contains the
   exact data fingerprint and dependency versions, no personal information.
4. State your OS, Python version (`python --version`), and how you installed
   the package.

## What we do NOT support

- Analyses involving **real, non-public, or personally identifiable data**.
  The project ships synthetic data only; we cannot help debug pipelines run
  on private datasets.
- Forks or modified copies of the pipeline (please reproduce first on the
  unmodified code).

## Response expectations

This is a research project maintained on a best-effort basis. Typical
response time is a few days; there are no SLA guarantees.
