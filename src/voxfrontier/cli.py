"""Command-line interface: ``voxfrontier <stage>``."""

from __future__ import annotations

import argparse
import logging
import sys

from voxfrontier import __version__
from voxfrontier.config import Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="voxfrontier",
        description=(
            "VoxFrontier: DEA + attribution + nonlinearity + DML causality + "
            "counterfactual simulation on synthetic livestream data. "
            "(alias: vxf)"
        ),
    )
    p.add_argument("--version", action="version", version=f"voxfrontier {__version__}")
    p.add_argument(
        "-c", "--config", default=None,
        help="path to a YAML config (default: bundled config/default.yaml)",
    )
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("synth", help="generate the synthetic dataset (data/synthetic)")

    sub.add_parser("run-all", help="run the full pipeline end-to-end")

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = Config.load(args.config)

    if args.command == "synth":
        from voxfrontier.pipeline import stage_synth

        stage_synth(cfg)
        return 0

    if args.command == "run-all":
        from voxfrontier.pipeline import run_all

        run_all(cfg)
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
