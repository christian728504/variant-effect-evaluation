"""submitit orchestration for the Stage 4 benchmark matrix (gpuh200).

One SLURM job per (dataset, model, accession, assay) across the datasets in
`bench_config.DATASET_MODEL_PLANS`. All SLURM parameters are
*derived* from the cluster introspection (the user's proven-working `srun`),
not hardcoded magic: gpuh200 has 8 H200s / 256 CPU / 1 TB on one node, so one GPU's
fair share is 32 CPU + 125 GB; the directives below match the user's proven-working
`srun` (`--gres=gpu:1 --cpus-per-task=32 --mem=125000MB`). `--gpu-freq` is deprecated
on this SLURM (21.08.5) and is intentionally not set.

Usage:
    python scripts/orchestrate.py --list      # print the job matrix, submit nothing
    python scripts/orchestrate.py --dry-run   # submit ONE job (microglia/cherimoya) e2e
    python scripts/orchestrate.py --submit     # submit the full job array (fire & forget)
    python scripts/orchestrate.py --collect    # aggregate result sidecars → all_benchmarks.parquet

`--submit` returns as soon as the array is queued and writes a manifest; run
`--collect` afterwards (it reads the durable `<stem>.result.json` sidecars each job
writes, so it works in a separate process from the submitter).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Make the sibling modules importable regardless of CWD or launch style.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger

from bench_config import (
    LOGS_DIR,
    PROJECT_ROOT,
    RESULTS_DIR,
    configure_logging,
    iter_jobs,
    job_stem,
)
from benchmark_job import run_single_benchmark

# --- Derived SLURM config (see resources/context/cluster-introspection.md) ---
PARTITION = "gpuh200"  # only GPU/H200 partition; user decision
GRES = "gpu:1"  # GRES is gpu:8 (no type sublabel) → request one GPU
CPUS_PER_TASK = 32  # 256 CPU ÷ 8 GPU
MEM = "125000MB"  # 1 TB ÷ 8 GPU ≈ 125 GB (matches running job's AllocMem)
TIME_MIN = 720  # 12 h; ample for the slowest many-tracks run, << 14-day cap
ARRAY_PARALLELISM = 8  # use all 8 GPUs; user decision
JOB_NAME = "vep_bench"

SCRIPTS_DIR = Path(__file__).resolve().parent  # dir holding bench_config/benchmark_job
PYTHON = str(PROJECT_ROOT / ".venv" / "bin" / "python")  # uv venv launcher
SLURM_FOLDER = LOGS_DIR / "slurm"
MANIFEST = SLURM_FOLDER / "submitted_jobs.json"
ALL_BENCH = RESULTS_DIR / "all_benchmarks.parquet"

# The single job used for the end-to-end dry run (smallest, BPNet-like, no assay).
DRY_RUN_JOB = ("caqtls_microglia", "cherimoya", "microglia", "NA")

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


def build_executor():
    """SlurmExecutor configured for one-GPU gpuh200 jobs (derived params)."""
    import submitit

    SLURM_FOLDER.mkdir(parents=True, exist_ok=True)
    ex = submitit.SlurmExecutor(folder=str(SLURM_FOLDER), python=PYTHON)
    ex.update_parameters(
        partition=PARTITION,
        gres=GRES,
        cpus_per_task=CPUS_PER_TASK,
        mem=MEM,
        time=TIME_MIN,
        array_parallelism=ARRAY_PARALLELISM,
        job_name=JOB_NAME,
        # Make benchmark_job / bench_config importable on the compute node.
        setup=[f"export PYTHONPATH={SCRIPTS_DIR}:$PYTHONPATH"],
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


JOB_COLUMNS = ("job_id", "dataset", "model", "accession", "assay")


def cmd_list() -> None:
    jobs = iter_jobs()
    _log_table(
        f"Stage 4 job matrix — {len(jobs)} jobs",
        JOB_COLUMNS,
        [("(unsubmitted)", *j) for j in jobs],
    )
    logger.info(
        "SLURM: partition={} gres={} cpus_per_task={} mem={} time={}min "
        "array_parallelism={}",
        PARTITION, GRES, CPUS_PER_TASK, MEM, TIME_MIN, ARRAY_PARALLELISM,
    )


def cmd_dry_run() -> int:
    """Submit exactly one job to gpuh200 and block on its result; validate outputs."""
    dataset, model, accession, assay = DRY_RUN_JOB
    logger.info("── DRY RUN · {} ──", job_stem(dataset, model, accession, assay))
    ex = build_executor()
    job = _submit_with_retry(
        lambda: ex.submit(run_single_benchmark, dataset, model, accession, assay)
    )
    logger.info("submitted SLURM job {}; waiting for result…", job.job_id)

    result = job.result()  # blocks until the SLURM job finishes
    logger.info("dry-run result:\n{}", json.dumps(result, indent=2))

    stem = job_stem(dataset, model, accession, assay)
    parquet = RESULTS_DIR / f"{stem}.parquet"
    state_log = LOGS_DIR / f"{stem}.state.log"
    sidecar = RESULTS_DIR / f"{stem}.result.json"
    ok = True
    for label, p, need_nonempty in [
        ("result parquet", parquet, True),
        ("state log", state_log, True),
        ("result sidecar", sidecar, True),
    ]:
        exists = p.exists() and (not need_nonempty or p.stat().st_size > 0)
        logger.info("  {} {}: {}", "✓" if exists else "✗", label, p)
        ok = ok and exists
    ok = ok and result.get("status") == "success"
    (logger.success if ok else logger.error)(
        "{} (status={})",
        "DRY RUN OK" if ok else "DRY RUN FAILED",
        result.get("status"),
    )
    if ok:
        logger.info(
            "To submit the FULL matrix yourself, run:\n"
            "  {} {} --submit\n"
            "then once jobs finish:\n"
            "  {} {} --collect",
            PYTHON, Path(__file__), PYTHON, Path(__file__),
        )
    return 0 if ok else 1


def _submit_and_record(jobs_spec, *, title: str, merge_manifest: bool = False) -> None:
    """Batch-submit `jobs_spec`, print a table, and (re)write the manifest.

    With `merge_manifest=True` the new submissions are merged into any existing
    manifest by stem (so re-submitting a subset keeps prior entries) — used by
    `--resubmit-missing`. Otherwise the manifest is overwritten (`--submit`).
    """
    if not jobs_spec:
        logger.success("nothing to submit — all jobs already have results.")
        return
    ex = build_executor()

    def _do_batch():
        out = []
        with ex.batch():
            for dataset, model, accession, assay in jobs_spec:
                job = ex.submit(run_single_benchmark, dataset, model, accession, assay)
                out.append((job, dataset, model, accession, assay))
        return out

    submitted = _submit_with_retry(_do_batch)

    rows = [(j.job_id, d, m, a, asy) for j, d, m, a, asy in submitted]
    _log_table(title, JOB_COLUMNS, rows)

    SLURM_FOLDER.mkdir(parents=True, exist_ok=True)
    new_entries = {
        job_stem(d, m, a, asy): {
            "job_id": j.job_id, "dataset": d, "model": m, "accession": a,
            "assay": asy, "stem": job_stem(d, m, a, asy),
        }
        for j, d, m, a, asy in submitted
    }
    if merge_manifest and MANIFEST.exists():
        existing = {m["stem"]: m for m in json.loads(MANIFEST.read_text())}
        existing.update(new_entries)
        manifest = list(existing.values())
    else:
        manifest = list(new_entries.values())
    MANIFEST.write_text(json.dumps(manifest, indent=2))
    logger.info("manifest → {}", MANIFEST)
    logger.info(
        "Jobs are queued (array_parallelism={}). Monitor with "
        "squeue -p {} -u $USER. When finished, aggregate with {} {} --collect.",
        ARRAY_PARALLELISM, PARTITION, PYTHON, Path(__file__),
    )


def cmd_submit() -> None:
    """Batch-submit the full matrix as a SLURM array (fire-and-forget) + manifest."""
    _submit_and_record(iter_jobs(), title=f"Submitted full matrix to {PARTITION}")


def cmd_resubmit_missing() -> None:
    """Submit only the jobs whose result sidecar is absent (e.g. after an OOM)."""
    missing = [
        (d, m, a, asy)
        for (d, m, a, asy) in iter_jobs()
        if not (RESULTS_DIR / f"{job_stem(d, m, a, asy)}.result.json").exists()
    ]
    logger.info("{} job(s) missing a result sidecar.", len(missing))
    _submit_and_record(
        missing,
        title=f"Re-submitted {len(missing)} missing job(s) to {PARTITION}",
        merge_manifest=True,
    )


def cmd_collect() -> None:
    """Aggregate the durable per-job result sidecars → all_benchmarks.parquet."""
    import polars as pl

    rows = []
    for sidecar in sorted(RESULTS_DIR.glob("*.result.json")):
        try:
            rows.append(json.loads(sidecar.read_text()))
        except Exception as e:  # noqa: BLE001
            logger.warning("skipping unreadable {}: {}", sidecar.name, e)

    if not rows:
        logger.warning("no result sidecars found in results/")
        return

    df = pl.DataFrame(rows)
    # Stable column order; tolerate any missing/extra keys.
    cols = [c for c in RESULT_COLUMNS if c in df.columns]
    df = df.select(cols + [c for c in df.columns if c not in cols])
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.write_parquet(ALL_BENCH)

    n_ok = int((df["status"] == "success").sum())
    n_err = int((df["status"] == "error").sum())
    logger.success(
        "aggregated {} jobs → {} ({} success, {} error)",
        df.height, ALL_BENCH, n_ok, n_err,
    )

    # Report jobs in the manifest that have no result sidecar yet (pending/lost).
    if MANIFEST.exists():
        manifest = json.loads(MANIFEST.read_text())
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


def main() -> int:
    configure_logging()
    p = argparse.ArgumentParser(description=__doc__)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true", help="print the job matrix only")
    g.add_argument("--dry-run", action="store_true", help="submit 1 job end-to-end")
    g.add_argument("--submit", action="store_true", help="submit the full job array")
    g.add_argument("--resubmit-missing", action="store_true",
                   help="submit only jobs whose result sidecar is missing")
    g.add_argument("--collect", action="store_true", help="aggregate result sidecars")
    a = p.parse_args()

    if a.list:
        cmd_list()
        return 0
    if a.dry_run:
        return cmd_dry_run()
    if a.submit:
        cmd_submit()
        return 0
    if a.resubmit_missing:
        cmd_resubmit_missing()
        return 0
    if a.collect:
        cmd_collect()
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
