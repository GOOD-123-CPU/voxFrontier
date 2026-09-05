"""Run manifest: cryptographic fingerprint of every pipeline execution.

Writes ``output/manifest.json`` containing:

* voxfrontier + dependency versions and Python build;
* SHA-256 of the input dataset and of every result table produced;
* random seed and wall-clock timestamps;
* key scalar results so runs can be compared across machines.

This is what makes "reproducible" auditable rather than aspirational: any
consumer can re-hash their local outputs and check they match the manifest.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import sys
from datetime import datetime, timezone
from os import PathLike
from pathlib import Path

logger = logging.getLogger("voxfrontier.manifest")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            block = fh.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def dependency_versions() -> dict[str, str]:
    versions = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in (
        "numpy", "pandas", "scipy", "sklearn", "matplotlib",
        "seaborn", "yaml", "pulp", "shap",
    ):
        try:
            mod = __import__(name)
            versions[name] = getattr(mod, "__version__", "unknown")
        except Exception:
            versions[name] = "not-installed"
    return versions


def write_manifest(
    output_dir: str | PathLike[str],
    data_path: str | PathLike[str],
    seed: int,
    summary: dict,
    started_at: datetime | None = None,
) -> Path:
    """Write ``manifest.json`` into ``output_dir`` and return its path."""
    output_dir = Path(output_dir)
    data_path = Path(data_path)
    tables = {}
    for csv in sorted(output_dir.glob("*.csv")):
        tables[csv.name] = sha256_file(csv)

    manifest = {
        "tool": "voxfrontier",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": (started_at or datetime.now(timezone.utc)).isoformat(),
        "seed": seed,
        "environment": dependency_versions(),
        "input": {
            "path": data_path.name,
            "sha256": sha256_file(data_path),
        },
        "result_tables_sha256": tables,
        "summary": summary,
    }
    out = output_dir / "manifest.json"
    out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False),
                   encoding="utf-8")
    logger.info("manifest written: %s (%d tables hashed)", out, len(tables))
    return out


def verify_manifest(output_dir: str | PathLike[str]) -> dict[str, bool]:
    """Re-hash local tables and compare against the stored manifest."""
    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest at {manifest_path}")
    stored = json.loads(manifest_path.read_text(encoding="utf-8"))
    results = {}
    for name, digest in stored.get("result_tables_sha256", {}).items():
        local = output_dir / name
        results[name] = local.exists() and sha256_file(local) == digest
    return results
