"""Render a gallery of planetary anomaly morphologies for visual validation.

The four panels are the textbook cases:

* a **major-image** perturbation (``s > 1``): a single positive bump produced
  by the four-cusp planetary caustic outside the Einstein ring;
* a **minor-image** perturbation (``s < 1``): the characteristic *dip*, where
  the pair of triangular caustics destroy the minor image and the source is
  demagnified relative to the single-lens curve;
* a **central-caustic** perturbation (``s ~ 1``) on a high-magnification
  event, sitting on the peak itself;
* a **low mass ratio** case (``q = 1e-5``), short and weak, near the detection
  limit -- the regime where an under-fitted PSPL refit would most badly
  inflate the efficiency.

Run with ``python examples/anomaly_gallery.py`` from the repository root.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from lenseff.config import Config
from lenseff.events import Event
from lenseff.inject import EventSetup, Planet, inject_planet
from lenseff.survey import Survey

CASES: list[tuple[str, float, float, float, float]] = [
    # label, q, s, u_0, t_E
    ("major image (s > 1)", 1.0e-3, 1.30, 0.30, 25.0),
    ("minor image (s < 1)", 1.0e-3, 0.77, 0.30, 25.0),
    ("central caustic (s ~ 1)", 1.0e-3, 1.05, 0.02, 25.0),
    ("low mass ratio", 1.0e-5, 1.10, 0.10, 25.0),
]


def best_alpha(
    setup: EventSetup, q: float, s: float, survey: Survey, config: Config, n: int = 180
) -> float:
    """Return the trajectory angle giving the strongest anomaly for this case.

    Scanning rather than deriving the geometry keeps the gallery honest: the
    angle is chosen by the same magnification code that the injection uses.
    """
    best, best_dev = 0.0, -1.0
    for alpha in np.linspace(0.0, 360.0, n, endpoint=False):
        window = inject_planet(setup, Planet(q, s, float(alpha)), survey, config).anomaly
        if window.peak_deviation > best_dev:
            best, best_dev = float(alpha), window.peak_deviation
    return best


def main() -> None:
    """Build the gallery and write it to ``docs/figures/anomaly_gallery.png``."""
    from lenseff.plotting import plot_injection

    config = Config.from_yaml("configs/roman_gbtds_demo.yaml")
    survey = Survey.from_config(config.survey, config.run.seed)
    figure, axes = plt.subplots(4, 2, figsize=(13.0, 13.0), height_ratios=[2.4, 1.0] * 2)
    t_0 = float(survey.times[survey.season_index == 2][4320])

    for panel, (label, q, s, u_0, t_E) in enumerate(CASES):
        event = Event(
            index=panel, t_0=t_0, u_0=u_0, t_E=t_E, source_mag=21.0, blend_ratio=0.2, rho=1.0e-3
        )
        setup = EventSetup.build(event, survey, config)
        alpha = best_alpha(setup, q, s, survey, config)
        injection = inject_planet(setup, Planet(q, s, alpha), survey, config)
        row, column = divmod(panel, 2)
        pair = (axes[2 * row][column], axes[2 * row + 1][column])
        plot_injection(
            injection,
            survey,
            axes=pair,
            title=(
                f"{label}: $q$={q:.0e}, $s$={s:.2f}, $\\alpha$={alpha:.0f}$^\\circ$, $u_0$={u_0:g}"
            ),
        )
        print(
            f"{label:26s} q={q:.0e} s={s:.2f} alpha={alpha:6.1f} "
            f"peak={injection.anomaly.peak_deviation:8.1f} sigma  "
            f"duration={injection.anomaly.duration_days:7.3f} d  "
            f"points={injection.points_in_anomaly()}"
        )

    figure.tight_layout()
    output = Path("docs/figures/anomaly_gallery.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=130)
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
