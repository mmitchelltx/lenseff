# Contributing to lenseff

Contributions are welcome — bug reports, questions about the method, and pull
requests alike.

## Getting help or reporting a problem

Open an issue at
<https://github.com/mmitchelltx/lenseff/issues>. For a bug, the most useful
report includes:

* the output of `lenseff show --provenance <your config>`, which contains the
  resolved configuration, its hash, and every dependency version;
* what you expected and what happened.

For a question about the *method* rather than the code — why a threshold is
what it is, whether an approximation is safe for your survey — open an issue
too. Those questions are the reason `docs/detection-criteria.md` exists, and
answering them usually improves it.

## Development setup

```bash
git clone https://github.com/mmitchelltx/lenseff
cd lenseff
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Before opening a pull request:

```bash
ruff check .           # lint
ruff format --check .  # formatting
mypy                   # type checking
pytest -q              # the full suite, ~90 s
```

CI runs all four on Python 3.11 and 3.12, plus the validation suite and a
reduced end-to-end demo, and all of them must pass.

## What a good change looks like

**Nothing scientific is hard-coded.** If your change introduces a number that a
user might reasonably want to vary, it belongs in the configuration schema in
`lenseff/config.py` with a default, a docstring, and a row in
`docs/configuration.md`. Unknown configuration keys are errors on purpose;
please keep it that way.

**Tests make real numerical assertions.** The suite compares against closed
forms, analytic moments, and independent implementations rather than checking
that code runs. If you add a sampler, assert its moments. If you add a
statistic, assert it against something computed a different way.

**Determinism is a contract, not a preference.** Every random draw comes from
`lenseff.rng.generator(seed, stream, *indices)`, addressed by a stable key
path. Never draw from a shared running sequence: it would make results depend
on worker count and task order, and break checkpoint resumption. If you add a
new source of randomness, add a stream name to `STREAM_IDS` — and never
renumber the existing ones, which would change the output of every existing
seed.

**Magnification comes from MulensModel.** `lenseff` does not implement lens
equations, magnification formulae, or finite-source integrals, and should not
start.

**Changes to the detection criteria need a rationale.** `detect.py` is the
scientific core. A change to what counts as a detection should come with an
update to `docs/detection-criteria.md` explaining what it buys and what it
costs, and ideally with a validation test.

## Code style

Ruff enforces formatting, imports, and docstrings (Google convention); type
hints are required on public functions and checked by mypy. Scientific
parameter names (`t_0`, `u_0`, `t_E`) are exempt from the naming rules — that
is deliberate, so that code reads like the literature it implements.
