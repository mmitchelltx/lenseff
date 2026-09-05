# Reproducing a published efficiency map

Everything a `lenseff` run produces is a pure function of one YAML file and one
integer seed. This page is what a stranger needs.

## The claim

Given the same configuration file and the same `run.seed`, `lenseff` produces
**bit-identical data columns**, on any machine, with any number of worker
processes, whether the run completed in one go or was interrupted and resumed.

Three design choices make that true rather than aspirational:

1. **No random draw comes from a running sequence.** Every one is addressed:
   trajectory angles come from `generator(seed, "alpha", cell, event)`,
   photometric noise from `generator(seed, "photometric_noise", event,
   realisation)`. Reordering the work cannot reorder the randomness. Asserted
   by `test_draws_are_independent_of_call_order` and
   `test_parallel_matches_serial_exactly`.
2. **Unknown configuration keys are errors.** A mistyped key cannot silently
   leave a threshold at its default.
3. **The config hash covers what can change a number** — survey, events, grid,
   criteria, output — and deliberately excludes the `compute` section, so a
   checkpointed run can be resumed on a machine with a different core count.

What is *not* bit-identical: the provenance timestamp, and the order of rows
across parquet shards when a run is resumed. The combined `injections.parquet`
is sorted, and the efficiency surface is order-independent by construction.

## Running it

```bash
git clone https://github.com/mmitchelltx/lenseff
cd lenseff
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

lenseff validate configs/roman_gbtds_demo.yaml   # confirm the hash matches the paper
lenseff run configs/roman_gbtds_demo.yaml
```

That writes, under `run.output_dir`:

| file | contents |
| --- | --- |
| `injections/shard_*.parquet` | one row per injection, written as the run proceeds |
| `injections.parquet` | the same rows, combined and sorted (small runs only) |
| `efficiency.parquet` | one row per grid cell, with Wilson intervals |
| `figures/efficiency_map.png` | the contour map |
| `figures/efficiency_slices.png` | efficiency against `log q` at fixed `s` |
| `manifest.json` | config hash and task counts, used by resume |
| `provenance.json` | the full run record |

## Checking that you got the same thing

The config hash is printed by `lenseff validate` and embedded in every parquet
file. Compare it first — if it differs, the configurations differ, and the
resolved configuration is inside the provenance block to show you where.

```python
from lenseff.provenance import read_provenance

block = read_provenance("results/roman_gbtds_demo/efficiency.parquet")
print(block["config_hash"], block["seed"])
print(block["packages"])  # every dependency version
print(block["git"])  # commit, and whether the tree was dirty
```

Then compare the surface itself:

```python
import pandas as pd

mine = pd.read_parquet("results/roman_gbtds_demo/efficiency.parquet")
theirs = pd.read_parquet("published/efficiency.parquet")
pd.testing.assert_frame_equal(mine, theirs)
```

## If the numbers differ

Work down this list:

1. **Config hash differs** → the configurations differ. Diff the `config`
   blocks from the two provenance records.
2. **Hash matches, numbers differ** → compare `packages` in the two provenance
   blocks. MulensModel, SciPy and NumPy versions can all move a chi-square in
   the last digits; a *large* difference points at MulensModel, since it owns
   every magnification.
3. **Only the marginal cells differ** → check `n_trials` per cell. Cells near
   the efficiency contour have the largest binomial variance, and a run with
   fewer angles or fewer events will disagree there first.

## Resuming an interrupted run

Just run the same command again. Completed tasks are read out of the existing
shards and skipped. Resuming into a directory written by a *different*
configuration is refused, and names both hashes.

## Re-thresholding without recomputing

Every criterion is stored per injection, along with both the global and the
anomaly-windowed Delta chi-square. Changing `delta_chi2_min` to see how the
surface moves does **not** require rerunning the grid:

```python
import pandas as pd
from lenseff.config import Config
from lenseff.efficiency import aggregate

config = Config.from_yaml("configs/roman_gbtds_demo.yaml")
records = pd.read_parquet("results/roman_gbtds_demo/injections.parquet")
records["detected"] = (
    (records["delta_chi2"] >= 300.0)
    & records["criterion_consecutive_points"]
    & records["criterion_in_season"]
)
aggregate(records, config)
```

Reporting how the surface responds to the threshold is a good idea; it is the
choice a referee is most likely to question.
