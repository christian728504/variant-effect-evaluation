"""submitit orchestration for the Stage 4 benchmark matrix (gpuh200).

One SLURM job per (dataset, model, accession, assay) across the datasets' plans in
`config/eval.yaml`. All SLURM parameters come from that file's `cluster.slurm` section,
derived from the cluster introspection (the user's proven-working `srun`): gpuh200 has
8 H200s / 256 CPU / 1 TB on one node, so one GPU's fair share is 32 CPU + 125 GB; the
directives match the proven-working `srun` (`--gres=gpu:1 --cpus-per-task=32
--mem=125000MB`). `--gpu-freq` is deprecated on this SLURM (21.08.5) and is not set.

These command functions are cfg-parametrized and driven by the CLI (`cli.py`):
`cmd_dry_run` enumerates the matrix and submits nothing; `cmd_submit` fires the full
array + writes a manifest; `cmd_collect` aggregates the durable `<stem>.result.json`
sidecars each job writes (so it runs in a separate process from the submitter).
"""

from __future__ import annotations

import json
from pathlib import Path

from loguru import logger

from .benchmark_job import run_single_benchmark
from .config import EvalConfig
from .utils import iter_jobs, job_stem

# Columns of the aggregated all_benchmarks.parquet (§8), in order.
RESULT_COLUMNS = [
    "dataset", "model", "accession", "assay",
    "n_variants_scored",
    "spearman_signed", "spearman_unsigned",
    "pearson_signed", "pearson_unsigned",
    "wall_time_seconds", "status", "error_message",
    "result_parquet", "state_log", "slurm_job_id", "hostname",
    "gpu_name", "started_utc",
]

JOB_COLUMNS = ("job_id", "dataset", "model", "accession", "assay")


def _slurm_folder(cfg: EvalConfig) -> Path:
    return cfg.paths.logs / "slurm"


def _manifest(cfg: EvalConfig) -> Path:
    return _slurm_folder(cfg) / "submitted_jobs.json"


def build_executor(cfg: EvalConfig):
    """SlurmExecutor for one-GPU gpuh200 jobs; params from `cfg.cluster.slurm`.

    The job array launches via the repo-relative venv python from `cfg.cluster.executor`;
    the package is installed (editable) in that shared-FS venv, so `run_single_benchmark`
    imports natively on the compute node — no PYTHONPATH shim needed.
    """
    import submitit

    slurm = cfg.cluster.slurm
    folder = _slurm_folder(cfg)
    folder.mkdir(parents=True, exist_ok=True)
    python = str(cfg.paths.project_root / cfg.cluster.executor.venv_python)
    ex = submitit.SlurmExecutor(folder=str(folder), python=python)
    ex.update_parameters(
        partition=slurm.partition,
        gres=slurm.gres,
        cpus_per_task=slurm.cpus_per_task,
        mem=slurm.mem,
        time=slurm.time_min,
        array_parallelism=slurm.array_parallelism,
        job_name=slurm.job_name,
        stderr_to_stdout=False,
    )
    return ex


def _submit_with_retry(submit_fn, *, tries: int = 4, delay: float = 10.0):
    """Run a submission callable, retrying transient slurmctld socket timeouts.

    A busy controller occasionally answers `sbatch` with "Socket timed out on
    send/recv operation" (the script is fine; the RPC just timed out). submitit
    surfaces this as FailedJobError, so we retry the whole submission a few times.
    """
    import time

    from submitit.core.utils import FailedJobError

    last_exc = None
    for attempt in range(1, tries + 1):
        try:
            return submit_fn()
        except FailedJobError as e:  # noqa: PERF203
            msg = str(e).lower()
            transient = "timed out" in msg or "socket" in msg or "try again" in msg
            if not transient or attempt == tries:
                raise
            last_exc = e
            logger.warning(
                "submission attempt {}/{} hit a transient slurmctld error; "
                "retrying in {:.0f}s…",
                attempt, tries, delay,
            )
            time.sleep(delay)
    raise last_exc  # unreachable


def _log_table(title, columns, rows) -> None:
    """Log `title`, a header, then one width-aligned line per row at INFO level.

    The plain-text stand-in for a table: same columns and rows, no box-drawing.
    `rows` is a list of equal-length sequences; cells are stringified.
    """
    logger.info(title)
    str_rows = [[str(c) for c in r] for r in rows]
    widths = [
        max(len(str(col)), *(len(r[i]) for r in str_rows)) if str_rows else len(str(col))
        for i, col in enumerate(columns)
    ]
    logger.info("  " + "  ".join(str(c).ljust(w) for c, w in zip(columns, widths)))
    for r in str_rows:
        logger.info("  " + "  ".join(c.ljust(w) for c, w in zip(r, widths)))


def cmd_dry_run(cfg: EvalConfig) -> int:
    """Enumerate the job matrix + SLURM params, run an inputs pre-flight — submit nothing.

    The snakemake-`-n` analogue: loading `cfg` already validated the YAML; this lists
    every job a full submit would create and runs a non-fatal inputs-present check.
    """
    slurm = cfg.cluster.slurm
    jobs = iter_jobs(cfg)
    _log_table(
        f"Stage 4 job matrix — {len(jobs)} jobs",
        JOB_COLUMNS,
        [("(unsubmitted)", *j) for j in jobs],
    )
    logger.info(
        "SLURM: partition={} gres={} cpus_per_task={} mem={} time={}min "
        "array_parallelism={}",
        slurm.partition, slurm.gres, slurm.cpus_per_task, slurm.mem,
        slurm.time_min, slurm.array_parallelism,
    )
    try:
        cfg.assert_inputs_exist()
        logger.success("inputs ✓ — all reference FASTAs + QTL TSVs present")
    except FileNotFoundError as e:
        logger.warning("inputs incomplete (a real submit may fail):\n{}", e)
    return 0


def cmd_submit(cfg: EvalConfig, config_path: str) -> int:
    """Batch-submit the full matrix as a SLURM array (fire-and-forget) + manifest.

    `config_path` is threaded as the job's 5th plain-string arg so each compute node
    re-loads the same YAML this submitter used.
    """
    jobs_spec = iter_jobs(cfg)
    ex = build_executor(cfg)

    def _do_batch():
        out = []
        with ex.batch():
            for dataset, model, accession, assay in jobs_spec:
                job = ex.submit(
                    run_single_benchmark, dataset, model, accession, assay, config_path
                )
                out.append((job, dataset, model, accession, assay))
        return out

    submitted = _submit_with_retry(_do_batch)

    rows = [(j.job_id, d, m, a, asy) for j, d, m, a, asy in submitted]
    _log_table(f"Submitted full matrix to {cfg.cluster.slurm.partition}", JOB_COLUMNS, rows)

    manifest_path = _manifest(cfg)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = [
        {
            "job_id": j.job_id, "dataset": d, "model": m, "accession": a,
            "assay": asy, "stem": job_stem(d, m, a, asy),
        }
        for j, d, m, a, asy in submitted
    ]
    manifest_path.write_text(json.dumps(manifest, indent=2))
    logger.info("manifest → {}", manifest_path)
    logger.info(
        "Jobs are queued (array_parallelism={}). Monitor with squeue -p {} -u $USER. "
        "When finished, aggregate with `variant-effect-evaluation collect`.",
        cfg.cluster.slurm.array_parallelism, cfg.cluster.slurm.partition,
    )
    return 0


def cmd_collect(cfg: EvalConfig) -> int:
    """Aggregate the durable per-job result sidecars → all_benchmarks.parquet."""
    import polars as pl

    results_dir = cfg.paths.results
    all_bench = results_dir / "all_benchmarks.parquet"
    manifest_path = _manifest(cfg)

    rows = []
    for sidecar in sorted(results_dir.glob("*.result.json")):
        try:
            rows.append(json.loads(sidecar.read_text()))
        except Exception as e:  # noqa: BLE001
            logger.warning("skipping unreadable {}: {}", sidecar.name, e)

    if not rows:
        logger.warning("no result sidecars found in {}", results_dir)
        return 0

    df = pl.DataFrame(rows)
    # Stable column order; tolerate any missing/extra keys.
    cols = [c for c in RESULT_COLUMNS if c in df.columns]
    df = df.select(cols + [c for c in df.columns if c not in cols])
    results_dir.mkdir(parents=True, exist_ok=True)
    df.write_parquet(all_bench)

    n_ok = int((df["status"] == "success").sum())
    n_err = int((df["status"] == "error").sum())
    logger.success(
        "aggregated {} jobs → {} ({} success, {} error)",
        df.height, all_bench, n_ok, n_err,
    )

    # Report jobs in the manifest that have no result sidecar yet (pending/lost).
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        have = {f"{r['dataset']}__{r['model']}__{r['accession']}__{r['assay']}"
                for r in rows}
        pending = [m["stem"] for m in manifest if m["stem"] not in have]
        if pending:
            logger.warning(
                "{} submitted job(s) have no result yet: {}{}",
                len(pending),
                ", ".join(pending[:10]),
                " …" if len(pending) > 10 else "",
            )

    # Compact metrics table for the successful jobs.
    ok_df = df.filter(pl.col("status") == "success").sort(["dataset", "model", "accession"])
    if ok_df.height:
        show = ["dataset", "model", "accession", "assay", "n_variants_scored",
                "spearman_signed", "spearman_unsigned", "pearson_signed",
                "pearson_unsigned", "wall_time_seconds"]
        metric_rows = [
            [f"{v:.3f}" if isinstance(v, float) else v for v in r]
            for r in ok_df.select(show).iter_rows()
        ]
        _log_table("benchmark metrics", show, metric_rows)
    if n_err:
        logger.error("errored jobs:")
        for r in df.filter(pl.col("status") == "error").iter_rows(named=True):
            logger.error(
                "  {}__{}__{}__{}: {}",
                r["dataset"], r["model"], r["accession"], r["assay"],
                r["error_message"],
            )
    return 0
