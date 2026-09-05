"""Detection efficiency for planetary microlensing surveys via injection-recovery.

``lenseff`` injects binary-lens (planetary) perturbations into simulated or
observed single-lens light curves, runs a detection pipeline over them, and
aggregates the recoveries into an efficiency surface in the ``(log q, log s)``
plane with binomial uncertainties.

The detection statistic is the chi-square difference between the binary-lens
model and a *freely refitted* point-source point-lens model.  Refitting is
essential: a free PSPL fit partially reabsorbs a planetary anomaly, and
comparing against the original PSPL parameters instead substantially
overestimates efficiency.
"""

from __future__ import annotations

from importlib import metadata

from lenseff.config import Config, ConfigError, load_config

try:
    __version__ = metadata.version("lenseff")
except metadata.PackageNotFoundError:  # pragma: no cover - source checkout
    __version__ = "0.0.0+unknown"

__all__ = ["Config", "ConfigError", "__version__", "load_config"]
