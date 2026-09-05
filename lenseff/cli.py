"""Command-line interface: ``lenseff <command> <config.yaml>``.

Phase 0 implements the configuration-facing commands (``validate`` and
``show``) so a configuration can be checked before any compute is spent.
``run`` is wired up in Phase 6.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from lenseff import __version__
from lenseff.config import Config, ConfigError
from lenseff.provenance import collect_provenance

__all__ = ["build_parser", "main"]


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser."""
    parser = argparse.ArgumentParser(
        prog="lenseff",
        description="Detection efficiency for planetary microlensing surveys.",
    )
    parser.add_argument("--version", action="version", version=f"lenseff {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_validate = sub.add_parser("validate", help="validate a configuration file")
    p_validate.add_argument("config", help="path to a YAML configuration file")

    p_show = sub.add_parser("show", help="print the resolved configuration and its hash")
    p_show.add_argument("config", help="path to a YAML configuration file")
    p_show.add_argument(
        "--provenance",
        action="store_true",
        help="print the full provenance block instead of just the config",
    )

    p_run = sub.add_parser("run", help="run an injection-recovery sweep")
    p_run.add_argument("config", help="path to a YAML configuration file")
    return parser


def _load(path: str) -> Config:
    try:
        return Config.from_yaml(path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI.

    Args:
        argv: Argument list, defaulting to ``sys.argv[1:]``.

    Returns:
        A process exit code.
    """
    args = build_parser().parse_args(argv)

    if args.command == "validate":
        config = _load(args.config)
        n = config.n_injections()
        print(f"{args.config}: OK")
        print(f"  config hash : {config.config_hash()}")
        print(f"  seed        : {config.run.seed}")
        print(f"  survey      : {config.survey.preset}")
        print(f"  grid        : {config.injection.grid.n_cells} cells")
        print(f"  injections  : {n:,}")
        return 0

    if args.command == "show":
        config = _load(args.config)
        block = (
            collect_provenance(config)
            if args.provenance
            else {
                "config_hash": config.config_hash(),
                "config": config.to_dict(),
            }
        )
        print(json.dumps(block, indent=2, sort_keys=True))
        return 0

    if args.command == "run":
        config = _load(args.config)
        print(
            f"error: 'lenseff run' is not implemented yet (Phase 6). "
            f"The configuration is valid: {config.short_hash()}",
            file=sys.stderr,
        )
        return 1

    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
