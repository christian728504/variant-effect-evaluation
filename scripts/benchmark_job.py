"""The picklable unit of work for one SLURM benchmark job.

`run_single_benchmark(dataset, model, accession, assay)` is a module-level function
whose args are all plain strings, so `submitit`/cloudpickle can ship it to a compute
node with no closure baggage. All heavy imports (torch, the scorers) happen inside the
call. It is fully standalone-runnable for debugging — no SLURM required:

    python scripts/benchmark_job.py caqtls_microglia cherimoya microglia NA

On every run it emits a comprehensive **state report** (Stage 4 §7) to stderr and to
`logs/<stem>.state.log` (the 4 job params + SLURM_JOB_ID, hostname, GPU name,
ISO-8601 timestamp, then `repr(VariantSet)`, `repr(scorer)`, `repr(FoldedModelWeights)`),
then scores, computes signed/unsigned Spearman + Pearson, writes the full per-variant
frame to `results/<stem>.parquet`, and returns a result dict. Any exception is
caught so the job returns `status="error"` (with the message + traceback in the log)
rather than crashing the collector.
"""

from __future__ import annotations

import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Make bench_config importable for standalone runs regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from loguru import logger

from bench_config import (
    DATASETS,
    LOGS_DIR,
    RESULTS_DIR,
    build_scorer,
    build_variant_set,
    compute_metrics,
    configure_logging,
    job_stem,
    ref_genome,
)

_METRIC_KEYS = (
    "spearman_signed",
    "spearman_unsigned",
    "pearson_signed",
    "pearson_unsigned",
)


def _emit_state_report(header: dict, vs, scorer) -> None:
    """Log the §7 state report; goes to stderr and the per-job .state.log file sink."""
    fw = getattr(scorer, "folded_weights", None)
    weights_block = (
        repr(fw)
        if fw is not None
        else "(many-tracks model — see track_indices / weights_path on the scorer above)"
    )
    sep = "-" * 72
    stem = header.get("stem", "job")
    lines = [
        f"state · {stem}",
        *[f"{k:<14}= {v}" for k, v in header.items()],
        sep,
        "[VariantSet]",
        repr(vs),
        sep,
        "[Scorer]",
        repr(scorer),
        sep,
        "[FoldedModelWeights]",
        weights_block,
    ]
    logger.info("\n".join(lines))


def run_single_benchmark(
    dataset: str, model: str, accession: str, assay: str
) -> dict:
    """Score one (dataset, model, accession, assay) config; return a result dict.

    `assay` is a plain string; "NA" (or empty) means the dataset has no assay
    distinction (microglia). Never raises — failures are captured into the returned
    dict's `status`/`error_message`.
    """
    t0 = time.perf_counter()
    configure_logging()
    assay_norm = None if assay in (None, "", "NA") else assay
    stem = job_stem(dataset, model, accession, assay)
    state_log_path = LOGS_DIR / f"{stem}.state.log"
    result_parquet = RESULTS_DIR / f"{stem}.parquet"

    started_utc = datetime.now(timezone.utc).isoformat()
    result: dict = {
        "dataset": dataset,
        "model": model,
        "accession": accession,
        "assay": assay if assay else "NA",
        "n_variants_scored": 0,
        **{k: None for k in _METRIC_KEYS},
        "wall_time_seconds": None,
        "status": "error",
        "error_message": None,
        "result_parquet": str(result_parquet),
        "state_log": str(state_log_path),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "<none>"),
        "hostname": socket.gethostname(),
        "gpu_name": "<unknown>",
        "started_utc": started_utc,
    }

    # Per-job .state.log sink: clean, timestamp-free artifact (the report + any
    # error traceback), removed in `finally`. The default stderr sink mirrors it.
    state_log_path.parent.mkdir(parents=True, exist_ok=True)
    log_sink = logger.add(state_log_path, level="INFO", format="{message}", mode="w")
    try:
        import torch

        result["gpu_name"] = (
            torch.cuda.get_device_name() if torch.cuda.is_available() else "<cpu>"
        )

        ds = DATASETS[dataset]
        rg = ref_genome(ds.genome_build)
        vs = build_variant_set(ds)
        scorer = build_scorer(model, accession, assay_norm, rg)

        header = {
            "dataset": dataset,
            "model": model,
            "accession": accession,
            "assay": assay if assay else "NA",
            "stem": stem,
            "SLURM_JOB_ID": result["slurm_job_id"],
            "hostname": result["hostname"],
            "gpu_name": result["gpu_name"],
            "started_utc": started_utc,
        }
        _emit_state_report(header, vs, scorer)

        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        score_df = scorer.score(vs)
        score_df.write_parquet(result_parquet)

        metrics = compute_metrics(
            score_df["logfc"].to_numpy(), score_df["effect_size"].to_numpy()
        )
        result.update(metrics)
        result["n_variants_scored"] = int(score_df.height)
        result["status"] = "success"
    except Exception as e:  # noqa: BLE001 — every failure must return, not crash
        result["status"] = "error"
        result["error_message"] = f"{type(e).__name__}: {e}"
        # loguru appends the traceback to both the stderr and .state.log sinks.
        logger.opt(exception=True).error("{} FAILED: {}", stem, result["error_message"])
    finally:
        logger.remove(log_sink)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        result["wall_time_seconds"] = round(time.perf_counter() - t0, 2)
        # Durable result sidecar — lets `orchestrate.py --collect` aggregate across
        # process boundaries (no need to hold submitit Job objects), and captures
        # error rows too. Written last so it reflects the final status + wall time.
        try:
            import json

            RESULTS_DIR.mkdir(parents=True, exist_ok=True)
            with open(RESULTS_DIR / f"{stem}.result.json", "w") as fh:
                json.dump(result, fh, indent=2)
        except Exception:
            pass

    return result


if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser(description="Run one benchmark job standalone (no SLURM).")
    p.add_argument("dataset")
    p.add_argument("model")
    p.add_argument("accession")
    p.add_argument("assay", nargs="?", default="NA", help="ATAC | DNASE | NA")
    a = p.parse_args()
    out = run_single_benchmark(a.dataset, a.model, a.accession, a.assay)
    print(json.dumps(out, indent=2))
