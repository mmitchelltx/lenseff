"""Free PSPL refit, delta-chi-square statistic and detection criteria.

Phase 3, the scientific core.  The statistic is
``chi2(best free PSPL refit) - chi2(binary-lens model)``, where the refit
lets ``t_0``, ``u_0``, ``t_E``, ``f_s`` and ``f_b`` all float.  Comparing
against the injected PSPL parameters instead would overestimate efficiency,
because a free refit partially reabsorbs the anomaly.  Planned public
interface::

    def refit_pspl(light_curve, config) -> PSPLFit          # multi-start
    def detection_statistic(light_curve, binary_model, config) -> Statistic
    def is_detected(statistic, config) -> bool

Not implemented yet.
"""

from __future__ import annotations
