# Hyperspectral-and-Thermal

Python package: `urbanheat`.

**Find what makes parts of a city hot, and estimate how much greening, cool roofs or less paving would cool them.**

Hyperspectral images (EMIT, EnMAP, Hyperion) show what the surface is made of; Landsat measures how hot it is.
A deep-learning model learns the link, explains which factors drive the heat, and predicts the cooling of
what-if scenarios. Later, the hyperspectral model teaches a Sentinel-2 / Landsat model that works in every city.

## Data

422 hyperspectral + Landsat pairs in 281 locations (Member 1's handover, QC v2.3):

| Sensor | Pairs | Locations | Pixel | Bands | Years |
|---|---|---|---|---|---|
| EMIT | 258 | 162 | 60 m | 232 | 2023–2026 |
| Hyperion | 133 | 106 | 30 m | 137 | 2013–2016 |
| EnMAP | 31 | 23 | 30 m | 177 | 2023–2026 |

Split by location (train / val / test = 321 / 45 / 56 pairs), so test cities are never seen in training.

## Quick start

```bash
pip install -e ".[ml,dev]"
urbanheat download --limit 2      # 2 pairs per sensor + the index, to try things out
urbanheat download                # everything (several GB)
urbanheat catalog                 # pairs per sensor and split, and what is on disk
urbanheat thermal-context         # median LST of other dates per pair (Earth Engine, needs ee.project)
urbanheat harmonize               # one dataset: 122 common bands, 60 m, all inputs (~2.2 GB)
pytest
```

Set your data folder and Earth Engine project in `configs/local.yaml` (copy `configs/local.example.yaml`).
If `urbanheat` is not on your PATH, use `python -m urbanheat.cli ...`.

Run commands from the repository root. Data goes to `data/` and run outputs to `runs/`; neither is in git.

## Who owns what

| Package | Owner | Delivers |
|---|---|---|
| `data`, `harmonize` | Members 1 + 2 | download, common wavelength grid + 60 m, the shared dataset |
| `models` | Member 2 | baselines, spectral / context / fusion CNNs |
| `drivers` | Member 2 | factor ranking, effect sizes, "what hyperspectral adds" |
| `efficient` | Member 3 | small fast model: band reduction, distillation, ONNX, benchmarks |
| `scenarios` | Member 3 | what-if engine → cooling maps |
| `explain` | Member 4 | per-pixel explanations |
| `app` | Member 4 | API, map dashboard, reports |

Members connect only through [docs/CONTRACTS.md](docs/CONTRACTS.md): one dataset format, one model interface,
one metrics schema, one scenario format. Until the real model exists, Members 3 and 4 can build against any
object with `predict(batch)` (for example a trivial model that returns the scene mean).

## Working together

- One branch per task, merged through a pull request reviewed by another member.
- `pytest` must pass before merging.
- Every step has a notebook in `notebooks/<member>/`, committed **with its outputs** so results can be read on
  GitHub without running anything. Notebooks call the package; reusable code lives in `src/urbanheat/`.

| Notebook | Shows |
|---|---|
| [member1/01_thermal_context_qc](notebooks/member1/01_thermal_context_qc.ipynb) | thermal context layer: coverage, observations, match with the target |
| [member2/01_dataset_overview](notebooks/member2/01_dataset_overview.ipynb) | harmonised dataset: splits, band grid, sensor agreement, variance split, a training patch |

## Honest use

LST is *surface* temperature, not air temperature or health risk. Factor effects are associations learned from
data, and scenario results are model estimates with uncertainty, not guarantees.
