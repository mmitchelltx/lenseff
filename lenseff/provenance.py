"""Run provenance: what produced this file, from what, with what versions.

Every parquet file written by ``lenseff`` carries a provenance block in its
schema metadata under the key ``lenseff``.  The block is enough to identify
the exact inputs of a run: the canonical configuration and its hash, the
master seed, the package version and git commit, the versions of the
scientific dependencies, and the platform.

Determinism note: the *data* in a parquet file is a pure function of the
config and the seed, so two runs of the same config compare equal column by
column.  The provenance block deliberately is not -- it records a wall-clock
timestamp -- so equality checks should compare table data, or pass
``include_timestamp=False``.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
import shutil
import subprocess
from importlib import metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final

import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

    from lenseff.config import Config

__all__ = [
    "METADATA_KEY",
    "collect_provenance",
    "git_describe",
    "package_versions",
    "read_provenance",
    "write_parquet",
]

#: Parquet schema-metadata key holding the JSON provenance block.
METADATA_KEY: Final[bytes] = b"lenseff"

#: Dependencies whose versions can change numerical results.
TRACKED_PACKAGES: Final[tuple[str, ...]] = (
    "lenseff",
    "numpy",
    "scipy",
    "pandas",
    "astropy",
    "matplotlib",
    "pyarrow",
    "PyYAML",
    "MulensModel",
)


def package_versions(packages: tuple[str, ...] = TRACKED_PACKAGES) -> dict[str, str]:
    """Return installed versions of the tracked packages.

    Args:
        packages: Distribution names to look up.

    Returns:
        Mapping of distribution name to version, with ``"not installed"`` for
        anything missing.
    """
    versions: dict[str, str] = {}
    for name in packages:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def git_describe(repo_dir: str | Path | None = None) -> dict[str, str]:
    """Return the git commit of the working tree, if there is one.

    Args:
        repo_dir: Directory inside the repository.  Defaults to the directory
            containing the installed package.

    Returns:
        Mapping with ``commit`` and ``dirty`` keys.  ``commit`` is
        ``"unknown"`` when git is unavailable or the directory is not a
        repository, which is the normal case for a pip-installed release.
    """
    unknown = {"commit": "unknown", "dirty": "unknown"}
    git = shutil.which("git")
    if git is None:
        return unknown
    cwd = Path(repo_dir) if repo_dir is not None else Path(__file__).resolve().parent
    try:
        commit = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
        status = subprocess.run(
            [git, "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (subprocess.SubprocessError, OSError):
        return unknown
    return {"commit": commit, "dirty": "true" if status else "false"}


def collect_provenance(
    config: Config,
    *,
    extra: dict[str, Any] | None = None,
    include_timestamp: bool = True,
) -> dict[str, Any]:
    """Assemble the provenance block for a run.

    Args:
        config: The resolved configuration for the run.
        extra: Additional key/value pairs to embed, such as timings or the
            number of injections actually completed.
        include_timestamp: Set false for byte-reproducible output.

    Returns:
        A JSON-serialisable provenance mapping.
    """
    block: dict[str, Any] = {
        "schema_version": 1,
        "run_name": config.run.name,
        "seed": config.run.seed,
        "config_hash": config.config_hash(),
        "config": config.to_dict(),
        "config_source": str(config.source_path) if config.source_path else None,
        "packages": package_versions(),
        "git": git_describe(),
        "platform": {
            "python": platform.python_version(),
            "implementation": platform.python_implementation(),
            "system": platform.system(),
            "machine": platform.machine(),
        },
    }
    if include_timestamp:
        block["created_utc"] = dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds")
    if extra:
        block["extra"] = extra
    return block


def write_parquet(
    frame: pd.DataFrame,
    path: str | Path,
    provenance: dict[str, Any],
    *,
    compression: str = "snappy",
) -> Path:
    """Write a dataframe to parquet with an embedded provenance block.

    Args:
        frame: The table to write.
        path: Destination file.  Parent directories are created.
        provenance: Block from :func:`collect_provenance`.
        compression: Parquet codec; ``"none"`` disables compression.

    Returns:
        The path written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(frame, preserve_index=False)
    existing = table.schema.metadata or {}
    payload = json.dumps(provenance, sort_keys=True, separators=(",", ":")).encode("utf-8")
    table = table.replace_schema_metadata({**existing, METADATA_KEY: payload})
    pq.write_table(table, path, compression=None if compression == "none" else compression)
    return path


def read_provenance(path: str | Path) -> dict[str, Any]:
    """Read the provenance block back out of a parquet file.

    Args:
        path: The parquet file.

    Returns:
        The provenance mapping.

    Raises:
        KeyError: If the file carries no ``lenseff`` provenance block.
    """
    schema = pq.read_schema(Path(path))
    meta = schema.metadata or {}
    if METADATA_KEY not in meta:
        raise KeyError(f"{path} has no lenseff provenance metadata")
    return json.loads(meta[METADATA_KEY].decode("utf-8"))
