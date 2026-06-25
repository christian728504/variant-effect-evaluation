"""Console-script entry point: `variant-effect-evaluation <subcommand>`.

A thin argparse layer over the orchestration + plotting logic. Nearly all configuration
lives in `config/eval.yaml`, so the only shared flag is `-c/--config` pointing at that
file; the four subcommands carry no other options.

    variant-effect-evaluation dry-run     # enumerate the matrix; submit nothing
    variant-effect-evaluation submit      # submit the full SLURM array (gpuh200)
    variant-effect-evaluation collect     # aggregate sidecars → all_benchmarks.parquet
    variant-effect-evaluation plot        # render the bar-chart PNGs into results/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import DEFAULT_CONFIG, load_config
from .orchestrate import cmd_collect, cmd_dry_run, cmd_submit
from .plots import render_all
from .utils import configure_logging


def main(argv: list[str] | None = None) -> int:
    """Parse args, load the config from `-c/--config`, and dispatch the subcommand."""
    configure_logging()

    # Shared `-c/--config` flag, inherited by every subparser.
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument(
        "-c", "--config", type=Path, default=DEFAULT_CONFIG,
        help="path to the eval YAML (default: %(default)s)",
    )

    p = argparse.ArgumentParser(
        prog="variant-effect-evaluation",
        description="Benchmark eval harness for the variant-effect-prediction models.",
    )
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("dry-run", parents=[parent],
                   help="enumerate the job matrix + check inputs; submit nothing")
    sub.add_parser("submit", parents=[parent],
                   help="submit the full SLURM job array (gpuh200)")
    sub.add_parser("collect", parents=[parent],
                   help="aggregate result sidecars → all_benchmarks.parquet")
    sub.add_parser("plot", parents=[parent],
                   help="render the static bar-chart PNGs into results/")

    args = p.parse_args(argv)
    config_path = args.config.resolve()
    cfg = load_config(config_path)

    if args.command == "dry-run":
        return cmd_dry_run(cfg)
    if args.command == "submit":
        return cmd_submit(cfg, str(config_path))
    if args.command == "collect":
        return cmd_collect(cfg)
    if args.command == "plot":
        return render_all(cfg)
    return 1  # unreachable: subparsers are required
