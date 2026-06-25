# variant-effect-evaluation

Benchmark eval harness for the
[`variant-effect-prediction`](https://github.com/christian728504/variant-effect-prediction)
model library. It scores QTL/benchmark datasets with each model (ChromBPNet, Cherimoya,
AlphaGenome, Enformer, Borzoi) and reports signed/unsigned Spearman + Pearson against the
published effect sizes.

The only tie to the library is `import variant_effect_prediction` (a pinned git
dependency). The input data the matrix needs is copied — laid out cleanly under `data/`.

## Layout

```
scripts/      bench_config.py  benchmark_job.py  orchestrate.py
              eval_static_plots.py        # static matplotlib bar charts
data/         references/  qtl/  weights/  metadata/   # eval inputs (gitignored)
results/  logs/                            # outputs (gitignored)
```

## Setup

```bash
uv sync          # resolves the pinned library + model deps into .venv
```

## Run

```bash
PY=.venv/bin/python
$PY scripts/orchestrate.py --list      # print the job matrix
$PY scripts/orchestrate.py --submit    # submit the full SLURM array (gpuh200)
$PY scripts/orchestrate.py --collect   # aggregate result sidecars → results/all_benchmarks.parquet
$PY scripts/eval_static_plots.py       # render bar-chart PNGs into results/

# one job, standalone (no SLURM):
$PY scripts/benchmark_job.py caqtls_microglia cherimoya microglia NA
```
