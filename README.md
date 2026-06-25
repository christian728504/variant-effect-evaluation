# variant-effect-evaluation

A declarative, SLURM-native benchmark harness for the
[`variant-effect-prediction`](https://github.com/christian728504/variant-effect-prediction)
model library. It scores QTL / benchmark datasets with each sequence-to-function model and
reports how well predicted effects track published effect sizes.

Point it at a config, run one command per stage, and get a tidy table of correlations plus
publication-ready bar charts — one SLURM job per (dataset, model, accession, assay) cell of
the matrix, fully parallel on GPU.

## Overview

Given a set of variant datasets (caQTLs, dsQTLs, bQTLs, allele-specific binding) and a set
of trained models (ChromBPNet, Cherimoya, AlphaGenome, Enformer, Borzoi), the harness:

1. builds a `VariantSet` per dataset — significance, SNV-only, and `isused` filtering;
2. scores it with each model to get a per-variant predicted log-fold-change;
3. computes **signed and unsigned Spearman + Pearson** of prediction vs. published effect size;
4. aggregates everything into a single parquet and renders bar charts.

The entire matrix — datasets, per-dataset model plans, weight/cell-type conventions, and the
SLURM/executor parameters — is defined in one human-editable file, [`config/eval.yaml`](config/eval.yaml),
parsed and validated by a pydantic v2 loader. Nothing about the run is hardcoded in Python.

## Features

- **Single declarative config** — the 54-job matrix, paths, model conventions, and cluster
  parameters all live in `config/eval.yaml`; the code is pure logic.
- **One CLI, four verbs** — `dry-run`, `submit`, `collect`, `plot`. Almost no flags.
- **Validated on load** — pydantic models (frozen, `extra="forbid"`) catch typos, resolve
  repo-relative paths, and cross-check that every dataset's genome build has a reference FASTA.
- **Embarrassingly parallel** — `submit` fans the matrix out as a SLURM array (one GPU each);
  each job is a picklable, standalone unit that writes a durable result sidecar.
- **Crash-tolerant aggregation** — `collect` reads the sidecars, so it runs in a separate
  process from the submitter and tolerates partial / errored runs.
- **Five models, two scoring families** — BPNet-like (ChromBPNet, Cherimoya) and many-tracks
  (AlphaGenome, Enformer, Borzoi), dispatched by a per-spec `kind`.

## How it works

```
config/eval.yaml ──load_config()──► EvalConfig (validated, paths resolved)
        │
   dry-run ────► enumerate 54 jobs + pre-flight inputs            (submits nothing)
        │
   submit ─────► SLURM array, 1 job / (dataset, model, accession, assay)
        │              each job: build VariantSet → build scorer → score
        │                        → metrics → results/<stem>.parquet
        │                        + results/<stem>.result.json  (durable sidecar)
        │                        + logs/<stem>.state.log        (full state report)
        │
   collect ────► aggregate *.result.json → results/all_benchmarks.parquet
        │
   plot ───────► results/bench_*_pearson_signed.png
```

Each job re-loads the config on its compute node from the path threaded in at submit time, so
the YAML on the shared filesystem is the single source of truth end-to-end.

## Prerequisites

- Python **≥ 3.12** and [`uv`](https://docs.astral.sh/uv/)
- A SLURM cluster with GPUs for `submit` (defaults target a `gpuh200` partition); `dry-run`,
  `collect`, and `plot` run anywhere, no GPU required
- The eval inputs present under `data/` (see [Data layout](#data-layout))

> [!NOTE]
> The model library and its dependencies (cherimoya, alphagenome-pytorch, borzoi-pytorch,
> enformer-pytorch, bpnet-lite) are pinned git sources, re-declared in `[tool.uv.sources]`.

## Getting started

```bash
uv sync          # resolve the pinned library + model deps, install the CLI (editable)
```

`uv sync` builds this project as an editable package, so the `variant-effect-evaluation`
console script and the `variant_effect_evaluation` import are available in `.venv` — and,
because the venv lives on the shared filesystem, on every compute node too.

## Usage

One console command drives everything. All configuration lives in `config/eval.yaml`; point
at a different file with `-c/--config`.

```bash
variant-effect-evaluation dry-run    # enumerate the job matrix + check inputs; submit nothing
variant-effect-evaluation submit     # submit the full SLURM array (gpuh200)
variant-effect-evaluation collect    # aggregate result sidecars → results/all_benchmarks.parquet
variant-effect-evaluation plot       # render bar-chart PNGs into results/
```

Typical flow:

```bash
variant-effect-evaluation dry-run            # sanity-check the matrix + inputs
variant-effect-evaluation submit             # fire the array, returns once queued
squeue -p gpuh200 -u $USER                   # monitor
variant-effect-evaluation collect            # once jobs finish
variant-effect-evaluation plot
```

> [!TIP]
> `dry-run` is the snakemake-`-n` analogue: it validates the config, lists every job a
> real submit would create, and runs a non-fatal inputs-present pre-flight — but submits
> nothing. Run it before every `submit`.

> [!NOTE]
> `submit` is fire-and-forget: it returns as soon as the array is queued and writes a
> manifest to `logs/slurm/`. Run `collect` afterwards — it reads the durable
> `<stem>.result.json` sidecars, so it works in a separate process and captures errored
> jobs too.

## Configuration

[`config/eval.yaml`](config/eval.yaml) has four top-level sections:

| Section | What it defines |
| --- | --- |
| `paths` | repo-relative data/results/logs dirs + the genome-build → reference-FASTA map |
| `models` | per-model construction conventions: cell-type dirs, weight subdirs, batch caps |
| `datasets` | each dataset's file, column map, significance rule, genome build, and **job plan** |
| `cluster` | SLURM directives (`partition`, `gres`, `mem`, …) and the executor's venv python |

Each dataset's `plan` is **fully enumerated** — one row per `(model, accession, assay)` job,
no templating. To add a model run to a dataset, add a plan row; to add a dataset, add a block.

### Datasets

| Key | Description | Build | Positives | Jobs |
| --- | --- | --- | --- | --- |
| `caqtls_eu` | European LCL ATAC caQTLs | hg38 | −log10(p) > 6 | 8 |
| `caqtls_african` | African LCL ATAC caQTLs (+ 6 AFGR populations) | hg38 | −log10(p) > 5 | 20 |
| `asb_african` | African LCL allele-specific binding | hg38 | pre-filtered | 8 |
| `caqtls_microglia` | Microglia scATAC caQTLs (BPNet-like only) | hg38 | pre-filtered | 2 |
| `dsqtls_yoruba` | Yoruban LCL dsQTLs | hg19 | `obs.label == 1` | 8 |
| `bqtls_pu1` | PU1/SPI1 bQTLs | hg19 | −log10(p) > 4 | 8 |

### Models

| Model | Scoring family | Weights source |
| --- | --- | --- |
| ChromBPNet | BPNet-like | `.torch` folds (converted from h5/tar) |
| Cherimoya | BPNet-like | `.torch` folds |
| AlphaGenome | many-tracks | safetensors (assay-selected head) |
| Enformer | many-tracks | DNase-track proxy |
| Borzoi | many-tracks | DNase-track proxy |

## Project structure

```
src/variant_effect_evaluation/
  config.py         # pydantic models + load_config() — the one data module
  utils.py          # logic: VariantSet build, scorer dispatch, metrics, job matrix
  benchmark_job.py  # run_single_benchmark() — the picklable unit of work
  orchestrate.py    # SLURM submit / collect; cfg-parametrized command functions
  plots.py          # static matplotlib bar charts
  cli.py            # argparse entry point (dry-run / submit / collect / plot)
config/eval.yaml    # the declarative matrix + cluster params
data/               # eval inputs (gitignored)
results/  logs/     # outputs (gitignored)
```

## Data layout

The harness resolves every input against `data/` (repo-relative; overridable in `paths`):

```
data/
  references/   hg38/  hg19/                     # bgzip + faidx genome FASTAs
  qtl/          *.benchmarking.all.tsv           # the benchmark datasets
  weights/      chrombpnet/  cherimoya/          # BPNet-like: <celltype>/fold_{i}.torch
                enformer-pytorch/  borzoi-pytorch/  alphagenome-pytorch/
```

> [!IMPORTANT]
> `data/`, `results/`, and `logs/` are gitignored. The inputs (~7 GB) are not tracked —
> they must be present on disk before `submit`. `dry-run`'s pre-flight reports what's missing.

## Outputs

| Path | Produced by | Contents |
| --- | --- | --- |
| `results/<stem>.parquet` | each job | full per-variant scores |
| `results/<stem>.result.json` | each job | one result row (metrics, status, timing, host) |
| `logs/<stem>.state.log` | each job | the §7 state report (VariantSet / scorer / weights) |
| `results/all_benchmarks.parquet` | `collect` | every job's result row, aggregated |
| `results/bench_*_pearson_signed.png` | `plot` | per-subset bar charts |

`<stem>` is `<dataset>__<model>__<accession>__<assay>`.
