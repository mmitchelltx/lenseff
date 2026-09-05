"""Injection of a binary-lens perturbation with given ``(q, s, alpha, rho)``.

Phase 2.  All magnification evaluation is delegated to ``MulensModel``; this
module never implements a lens equation, magnification or finite-source
integral of its own.  Planned public interface::

    def inject_planet(light_curve, event, q, s, alpha, rho, config) -> LightCurve
    def anomaly_window(event, q, s, alpha) -> tuple[float, float]

Not implemented yet.
"""

from __future__ import annotations
