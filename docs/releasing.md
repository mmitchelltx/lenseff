# Releasing, archiving, and the JOSS submission

Three things happen at a release: a version tag, a Zenodo archive with a DOI,
and (once) a JOSS submission. This page is the order to do them in.

## Before anything else: the placeholders

Three files carry `TODO` markers that must be resolved before a public release.
Grep for them:

```bash
grep -rn "TODO before" paper/paper.md CITATION.cff
```

| file | what to fix |
| --- | --- |
| `paper/paper.md` | ORCID (`0000-0000-0000-0000` is a placeholder and **JOSS's editorial bot will reject it**), affiliation |
| `CITATION.cff` | ORCID, affiliation, `date-released`, and the Zenodo DOI once it exists |
| `.zenodo.json` | affiliation, and an `orcid` field on the creator |

## 1. Check the paper builds

The `draft PDF` workflow compiles `paper/paper.md` on every push that touches
`paper/`, and uploads the PDF as a build artifact. Trigger it manually from the
Actions tab, or push a change to the paper, then download the artifact and read
it. Do this before submitting; the JOSS build is the same one.

Verify the bibliography separately. Every DOI in `paper/paper.bib` was checked
against a primary source on 2026-09-06, but re-check before submission:

```bash
grep -o 'doi *= *{[^}]*}' paper/paper.bib | sed 's/.*{\(.*\)}/https:\/\/doi.org\/\1/' | while read -r url; do
  printf '%s  ' "$url"
  curl -s -o /dev/null -w '%{http_code}\n' -L "$url"
done
```

Anything that is not `200` needs attention — except `000`, which means curl
could not reach `doi.org` at all (a proxy or an offline machine), not that the
DOI is bad. JOSS's bot runs an equivalent check and will comment on failures.

Note on provenance: each DOI above was verified against a publisher or ADS page
on 2026-09-06, but **not** by resolving `doi.org`, which was unreachable from
the machine the bibliography was written on. Run the loop once from a
networked machine before submitting.

## 2. Validate the metadata files

```bash
pip install cffconvert
cffconvert --validate                       # CITATION.cff against the CFF 1.2.0 schema
python -c "import json; json.load(open('.zenodo.json'))"
```

`cffconvert` pins an old `jsonschema`; install it in a throwaway environment
rather than the development one.

## 3. Connect the repository to Zenodo (once)

1. Sign in at <https://zenodo.org> with GitHub.
2. Go to <https://zenodo.org/account/settings/github/> and flip the toggle for
   `mmitchelltx/lenseff` **on**.

Only releases created *after* the toggle is switched on are archived, so do
this before making the release, not after.

Zenodo reads `.zenodo.json` from the tagged tree, so the title, description,
authors, licence and keywords in the archive come from that file rather than
from Zenodo's web form. Editing it is the way to change the archive's metadata.

## 4. Make the release

```bash
# bump the version in pyproject.toml and CITATION.cff first, and set
# date-released in CITATION.cff to today
git commit -am "Release v0.1.0"
git tag -a v0.1.0 -m "lenseff v0.1.0"
git push origin main --tags
```

Then create a GitHub Release from the tag (`gh release create v0.1.0
--generate-notes`, or the web UI). Zenodo picks it up within a minute or two
and mints two DOIs:

* a **version DOI**, pointing at v0.1.0 specifically;
* a **concept DOI**, which always resolves to the newest version. This is the
  one to put in the README badge and in `CITATION.cff`, because it does not go
  stale.

## 5. Record the DOI

Add the concept DOI to `CITATION.cff` as a top-level `doi:` field, and add the
badge to the README:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.XXXXXXX.svg)](https://doi.org/10.5281/zenodo.XXXXXXX)
```

Commit that on `main`. It will not be in the v0.1.0 archive — that is expected
and harmless, and is why the concept DOI is the one that is cited.

## 6. Submit to JOSS

Submit at <https://joss.theoj.org/papers/new>. You will need:

* the repository URL;
* the **version tag** of the release under review (`v0.1.0`);
* the **archive DOI** from step 4 (the version DOI is what JOSS asks for here).

JOSS's review checklist maps onto the repository as follows:

| checklist item | where it lives |
| --- | --- |
| open source, OSI licence | `LICENSE` (MIT) |
| installation instructions | `README.md` |
| example usage | `README.md`, `docs/tutorial.ipynb` |
| functionality documentation | `docs/configuration.md`, module docstrings |
| automated tests | `tests/`, run by `.github/workflows/ci.yml` |
| community guidelines | `CONTRIBUTING.md` |
| statement of need | `paper/paper.md` |
| references with DOIs | `paper/paper.bib` |

Reviewers will interrogate the scientific choices before the code. Expect
questions about the Δχ² threshold, the consecutive-points requirement, and the
decision to evaluate χ²(binary) at the injected parameters rather than fitting.
Those are answered in `docs/detection-criteria.md`, which is the page to point
a reviewer at first.

During review, JOSS will ask for changes in the repository. Keep making
releases as needed; only the archive DOI recorded in the review thread has to
be updated at the end.

## Subsequent releases

Steps 4 and 5 only. Zenodo archives every release automatically once the
toggle is on, the concept DOI keeps resolving to the newest one, and the badge
never needs changing.
