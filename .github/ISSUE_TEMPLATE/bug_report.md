---
name: Bug report
about: Report a reproducible problem in VoxFrontier
title: "[bug] "
labels: bug
assignees: ""
---

**Before you start**

- [ ] I searched existing issues (open and closed) for a duplicate.
- [ ] I can reproduce this with the **default synthetic pipeline**
      (`vxf synth && vxf run-all`), or the bug is about documentation/packaging.
- [ ] This report contains **no real personal data** (names, IDs, links to
      private datasets). The project is synthetic-data-only.

**Describe the bug**
A clear and concise description of what went wrong.

**To reproduce**
Steps / minimal commands, e.g.:

```bash
vxf synth
vxf run-all --config my-config.yaml
```

**Expected behavior**
What you expected to happen instead.

**Environment (please complete)**
- OS: [e.g. Ubuntu 22.04 / Windows 11 / macOS 14]
- Python version: [output of `python --version`]
- VoxFrontier version: [output of `vxf --version`]
- Install method: [pip / editable / docker]

**Run manifest**
Paste `output/manifest.json` (it contains data fingerprints and dependency
versions only — no personal information).

**Additional context**
Logs, screenshots of figures, or anything else relevant.
