# Configuration reference

A `lenseff` configuration is a single YAML document with seven sections.
`run`, `survey`, `events` and `injection` are required; `detection`, `compute`
and `output` fall back to defaults. Unknown keys are rejected with the dotted
path of the offender, so a typo can never silently change a run.

Validate before spending compute:

```bash
lenseff validate configs/roman_gbtds_demo.yaml
```

## `run`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `name` | str | *required* | run label, used in output filenames |
| `seed` | int ≥ 0 | *required* | master seed; every draw derives from it |
| `output_dir` | path | `results` | parquet, checkpoints and figures |
| `overwrite` | bool | `false` | allow writing into an existing output dir |

## `survey`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `preset` | str | `custom` | `roman_gbtds`, or `custom` to specify everything |
| `cadence_minutes` | float > 0 | *from preset* | interval between visits in a season |
| `precision_scale` | float > 0 | `1.0` | multiplies every σ; `0.5` = twice as precise |
| `dropout_fraction` | float ∈ [0,1) | `0.0` | scheduled visits randomly discarded |

A preset supplies a complete `survey` block; anything written under `survey:`
overrides it key by key. The **resolved** values are what get validated,
hashed and stamped into the output, so a run record is never a bare preset
name that could later drift.

### `survey.seasons`

| key | type | meaning |
| --- | --- | --- |
| `t_start` | float | HJD of the first season start |
| `length_days` | float > 0 | duration of each season |
| `start_offsets_days` | list of float | season starts in days after `t_start`, strictly increasing and non-overlapping |

Seasons matter more than any other structural choice: an anomaly lasting hours
to days is simply invisible in a gap.

### `survey.photometry`

A photon-noise-plus-systematics model, so the precision-versus-magnitude curve
has the right shape without hard-coding one survey's numbers.

| key | type | meaning |
| --- | --- | --- |
| `band` | str | band name (informational) |
| `zero_point` | float | magnitude giving 1 detected e⁻/s |
| `exposure_time_s` | float > 0 | effective exposure per visit |
| `n_exposures` | int ≥ 1 | exposures combined per visit |
| `sky_e_per_s_per_pixel` | float ≥ 0 | sky plus unresolved-star background |
| `read_noise_e` | float ≥ 0 | read noise per pixel per exposure |
| `n_pixels` | float > 0 | effective aperture area |
| `systematic_floor_mmag` | float ≥ 0 | floor added in quadrature |
| `saturation_mag` | float | brighter than this is unusable |
| `faint_limit_mag` | float | fainter than this is dropped |

## `events`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `source` | `population` \| `catalog` | `population` | draw events or read them |
| `n_events` | int ≥ 1 | *required* | baseline PSPL events in the sample |
| `catalog_path` | path | `null` | required when `source: catalog` |
| `require_peak_in_season` | bool | `true` | reject events peaking in a gap |
| `peak_window_t_E` | float > 0 | `2.0` | half-width of the coverage window, in t_E |
| `min_points_near_peak` | int ≥ 0 | `100` | reject events with fewer usable points in that window |

### `events.distributions`

One mapping per parameter: `u_0`, `t_E`, `source_mag`, `blend_ratio`, `rho`
are required in population mode; `t_0` is optional and defaults to uniform
over the union of the season windows.

```yaml
t_E: { dist: lognormal, median: 25.0, sigma_ln: 0.6, truncate_min: 1.0, truncate_max: 300.0 }
```

| `dist` | parameters |
| --- | --- |
| `fixed` | `value` |
| `uniform` | `min`, `max` |
| `loguniform` | `min` > 0, `max` |
| `normal` | `mean`, `std` > 0 |
| `lognormal` | `median` > 0, `sigma_ln` > 0 |
| `powerlaw` | `slope`, `min` > 0, `max` |

`truncate_min` / `truncate_max` apply hard bounds to any family.

## `injection`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `grid.log_q` | `{min, max, n}` | *required* | log₁₀ mass ratio axis |
| `grid.log_s` | `{min, max, n}` | *required* | log₁₀ separation axis |
| `grid.n_alpha` | int ≥ 1 | *required* | trajectory angles per (cell, event) |
| `grid.alpha_mode` | `stratified` \| `random` \| `uniform` | `stratified` | one angle per equal bin of [0°, 360°) is unbiased with lower variance |
| `grid.events_per_cell` | int ≥ 0 | `0` | `0` uses the whole event sample |
| `finite_source` | bool | `true` | include ρ |
| `magnification_methods` | str | `VBBL` | MulensModel method near the anomaly |
| `default_method` | str | `point_source_point_lens` | method away from it |
| `method_window_t_E` | float > 0 | `1.0` | half-width of the accurate-method window, in t_E |
| `include_zero_q_control` | bool | `true` | run planet-free controls |
| `n_zero_q_trials` | int ≥ 0 | `0` | how many; measures the false-positive rate |

## `detection`

See [`detection-criteria.md`](detection-criteria.md).

| key | type | default |
| --- | --- | --- |
| `delta_chi2_min` | float > 0 | `160.0` |
| `consecutive_points` | int ≥ 1 | `3` |
| `point_sigma` | float > 0 | `3.0` |
| `require_same_sign` | bool | `true` |
| `require_in_season` | bool | `true` |
| `max_gap_within_run_days` | float > 0 | `0.25` |
| `min_points_in_anomaly` | int ≥ 0 | `1` |

### `detection.refit`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `n_starts` | int ≥ 1 | `12` | multi-start initialisations |
| `method` | `Nelder-Mead` \| `Powell` \| `L-BFGS-B` | `Nelder-Mead` | optimiser for the non-linear parameters |
| `max_iterations` | int ≥ 1 | `5000` | iteration cap per start |
| `tolerance` | float > 0 | `1e-6` | convergence tolerance |
| `free_blending` | bool | `true` | fit `f_s`, `f_b` linearly at fixed non-linear parameters |
| `t_0_jitter_t_E` | float ≥ 0 | `0.3` | spread of `t_0` starts, in t_E |
| `t_E_factors` | list > 0 | `[0.7, 1.0, 1.4]` | multipliers on truth `t_E` for starts |
| `u_0_factors` | list > 0 | `[0.6, 1.0, 1.6]` | multipliers on truth `u_0` for starts |
| `mask_anomaly_start` | bool | `true` | seed one start from an anomaly-masked fit |

Under-fitting here inflates efficiency, which is exactly the failure mode the
package exists to avoid, so the refit is deliberately over-provisioned.

## `compute`

| key | type | default | meaning |
| --- | --- | --- | --- |
| `n_workers` | int ≥ 0 | `0` | `0` = one per CPU |
| `chunk_size` | int ≥ 1 | `64` | injections dispatched per task |
| `checkpoint_every` | int ≥ 1 | `5000` | injections between checkpoint flushes |
| `resume` | bool | `true` | resume an interrupted run |
| `max_records_in_memory` | int ≥ 1 | `200000` | bounds memory on 10⁶-injection grids |

## `output`

| key | type | default |
| --- | --- | --- |
| `compression` | `snappy` \| `zstd` \| `gzip` \| `none` | `snappy` |
| `write_injections` | bool | `true` |
| `write_efficiency` | bool | `true` |
| `write_plots` | bool | `true` |
| `confidence_level` | float ∈ (0,1) | `0.6827` |
