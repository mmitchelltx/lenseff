"""Command-line interface: ``lenseff <command> <config.yaml>``.

Phase 0 implements the configuration-facing commands (``validate`` and
``show``) so a configuration can be checked before any compute is spent.
``run`` is wired up in Phase 6.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from lenseff import __version__
from lenseff.config import Config, ConfigError
from lenseff.events import pspl_magnification
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

    p_lc = sub.add_parser("lightcurve", help="simulate one baseline PSPL light curve and plot it")
    p_lc.add_argument("config", help="path to a YAML configuration file")
    p_lc.add_argument("--event", type=int, default=0, help="index into the sampled event list")
    p_lc.add_argument("--output", "-o", default="lightcurve.png", help="output image path")
    p_lc.add_argument(
        "--window",
        type=float,
        default=3.0,
        help="zoom to this many Einstein times either side of the peak (0 = whole survey)",
    )

    p_run = sub.add_parser("run", help="run an injection-recovery sweep")
    p_run.add_argument("config", help="path to a YAML configuration file")
    p_run.add_argument("--quiet", action="store_true", help="suppress progress reporting")
    p_run.add_argument("--workers", type=int, default=None, help="override compute.n_workers")
    return parser


def _load(path: str) -> Config:
    try:
        return Config.from_yaml(path)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


def _lightcurve(args: argparse.Namespace) -> int:
    """Simulate one baseline event and write a plot of it."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lenseff.events import sample_events, simulate_light_curve
    from lenseff.plotting import plot_light_curve
    from lenseff.survey import Survey

    config = _load(args.config)
    survey = Survey.from_config(config.survey, config.run.seed)
    events = sample_events(config, survey)
    if not 0 <= args.event < len(events):
        print(
            f"error: --event must be in [0, {len(events) - 1}]; the sample holds "
            f"{len(events)} events",
            file=sys.stderr,
        )
        return 2
    event = events[args.event]
    light_curve = simulate_light_curve(event, survey, config)
    model_times = np.linspace(event.t_0 - 4.0 * event.t_E, event.t_0 + 4.0 * event.t_E, 4000)
    magnification = pspl_magnification(
        event, model_times, finite_source=config.injection.finite_source
    )
    figure = plot_light_curve(
        light_curve,
        survey,
        window_t_E=args.window if args.window > 0 else None,
        model_curves={
            "PSPL truth": (model_times, light_curve.f_source * magnification + light_curve.f_blend)
        },
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=150)
    plt.close(figure)
    print(f"event {event.index}: t_0={event.t_0:.3f} u_0={event.u_0:.4f} t_E={event.t_E:.2f} d")
    print(
        f"  source {survey.photometry.band}={event.source_mag:.2f}, f_b/f_s={event.blend_ratio:.2f}"
    )
    print(f"  {light_curve.n_points:,} measurements, peak A={light_curve.magnification.max():.2f}")
    print(f"  wrote {output}")
    return 0


def _run(args: argparse.Namespace) -> int:
    """Execute the sweep, aggregate it, and write the figures."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lenseff.efficiency import aggregate, control_summary, load_injections, write_efficiency
    from lenseff.grid import run_grid
    from lenseff.plotting import plot_efficiency_map, plot_efficiency_slices

    config = _load(args.config)
    if args.workers is not None:
        config = dataclasses.replace(
            config, compute=dataclasses.replace(config.compute, n_workers=args.workers)
        )
    output_dir = run_grid(config, progress=not args.quiet)

    records = load_injections(output_dir)
    surface = aggregate(records, config)
    if config.output.write_efficiency:
        print(f"  wrote {write_efficiency(surface, config)}")

    if config.output.write_plots:
        figures = Path(output_dir) / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        for name, builder in (
            ("efficiency_map", plot_efficiency_map),
            ("efficiency_slices", plot_efficiency_slices),
        ):
            figure = builder(surface, config)
            path = figures / f"{name}.png"
            figure.savefig(path, dpi=150)
            plt.close(figure)
            print(f"  wrote {path}")

    controls = control_summary(records, config)
    print(f"\n{config.run.name} [{config.short_hash()}]")
    print(f"  injections      : {len(records):,}")
    print(f"  grid cells      : {len(surface)}")
    if len(surface):
        print(
            f"  efficiency      : {surface['efficiency'].min():.3f} - "
            f"{surface['efficiency'].max():.3f} "
            f"(mean {surface['efficiency'].mean():.3f})"
        )
    if controls.get("n_controls"):
        print(
            f"  controls        : {controls['n_false_positives']}/{controls['n_controls']} "
            f"false positives, rate {controls['false_positive_rate']:.4f} "
            f"[{controls['false_positive_low']:.4f}, {controls['false_positive_high']:.4f}]"
        )
        print(
            f"  max control dchi2: {controls['max_delta_chi2']:+.4g} "
            f"({controls['max_delta_chi2_relative']:+.1e} relative; must be <= 0 "
            f"up to float64 rounding)"
        )
    return 0


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

    if args.command == "lightcurve":
        return _lightcurve(args)

    if args.command == "run":
        return _run(args)

    raise AssertionError(f"unhandled command {args.command!r}")  # pragma: no cover


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
