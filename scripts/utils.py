"""Benchmark logic, parametrized by the config objects in `config.py`.

Everything here is *behavior* (polars filtering, scorer construction, library metadata
lookups, metric math) as opposed to *data* (which lives in config/eval.yaml). All heavy
torch/model imports are deferred into the scorer factories so this module imports cleanly
without a GPU. Functions that need paths or conventions take an `EvalConfig` (or a
`QTLDataset`/`ModelSpec` from it) as an argument — no module-level state.
"""

from __future__ import annotations

import sys

import polars as pl
from loguru import logger

from config import EvalConfig, ModelSpec, QTLDataset


def configure_logging(level: str = "INFO") -> None:
    """Reset loguru to a single stderr sink at `level` using loguru's default format.

    Idempotent: removes any existing sinks and installs one stderr sink. Call once from
    each entry point (orchestrator / job runner).
    """
    logger.remove()
    logger.add(sys.stderr, level=level)


# --------------------------------------------------------------------------------
# Job matrix helpers
# --------------------------------------------------------------------------------


def iter_jobs(cfg: EvalConfig) -> list[tuple[str, str, str, str]]:
    """Flatten the datasets' plans to a list of (dataset, model, accession, assay) tuples.

    `assay` is stringified ("NA" when null) so every job key is a plain-string 4-tuple —
    picklable and usable directly in result filenames.
    """
    jobs: list[tuple[str, str, str, str]] = []
    for dataset, ds in cfg.datasets.items():
        for spec in ds.plan:
            jobs.append((dataset, spec.model, spec.accession, spec.assay or "NA"))
    return jobs


def job_stem(dataset: str, model: str, accession: str, assay: str | None) -> str:
    """Canonical filesystem-safe stem for a job's result/log files."""
    return f"{dataset}__{model}__{accession}__{assay if assay else 'NA'}"


# --------------------------------------------------------------------------------
# VariantSet construction (significance + SNV + isused filtering)
# --------------------------------------------------------------------------------


def ref_genome(cfg: EvalConfig, build: str = "GRCh38"):
    from variant_effect_prediction import RefGenome

    return RefGenome(cfg.paths.ref_fasta_paths[build])


def build_variant_set(ds: QTLDataset, cfg: EvalConfig, *, significant_only: bool = True):
    """Load a QTL TSV → 0-based start → (optional) significance filter → VariantSet.

    The scored set is SNV ∩ significant ∩ isused: SNV-only is applied by
    `VariantSet.from_dataframe(snvs_only=True)`, the significance cutoff is applied here on
    the raw p-value column, and isused is applied at score-time by `vs.used()`.
    """
    from variant_effect_prediction import VariantSet

    path = cfg.paths.qtl / ds.filename
    df = pl.read_csv(path, separator="\t", infer_schema_length=2**13)
    n_total = df.height
    df = df.with_columns((pl.col(ds.pos_col) - 1).cast(pl.Int64).alias("__start0"))

    filters: list[str] = ["snvs_only"]
    if significant_only and ds.significance_neglog10p is not None:
        cutoff = ds.significance_neglog10p
        # -log10(p) > cutoff ⇔ significant. null p-values yield null ⇒ dropped by filter.
        df = df.filter(-(pl.col(ds.pvalue_col).log10()) > cutoff)
        filters.append(f"-log10({ds.pvalue_col})>{cutoff}")
    if significant_only and ds.significance_label_col is not None:
        df = df.filter(pl.col(ds.significance_label_col) == 1)
        filters.append(f"{ds.significance_label_col}==1")
    if ds.isused_col is not None:
        filters.append(f"isused({ds.isused_col})")

    meta = {
        "name": ds.key,
        "source_path": str(path),
        "filters_applied": filters,
        "n_total": n_total,
    }
    return VariantSet.from_dataframe(
        df,
        chrom_col=ds.chrom_col,
        start_col="__start0",
        allele1_col=ds.allele1_col,
        allele2_col=ds.allele2_col,
        effect_size_col=ds.effect_size_col,
        pvalue_col=ds.pvalue_col,
        isused_col=ds.isused_col,
        apply_isused=ds.isused_col is not None,
        snvs_only=True,
        meta=meta,
    )


# --------------------------------------------------------------------------------
# Scorer dispatch
# --------------------------------------------------------------------------------


def track_indices_for(model: str, accession: str, assay: str | None = None) -> list[int]:
    """Look up a many-tracks model's track index(es) for an ENCODE accession.

    For AlphaGenome `track_index` is per-head, so `assay` (ATAC/DNASE) selects the head
    and must be passed to disambiguate.
    """
    from variant_effect_prediction.metadata import (
        load_alphagenome_accession_map,
        load_borzoi_tracks,
        load_enformer_tracks,
    )

    if model == "enformer":
        df = load_enformer_tracks().filter(pl.col("accession") == accession)
        idx = df["index"].to_list()
    elif model == "borzoi":
        df = load_borzoi_tracks().filter(pl.col("accession") == accession)
        idx = df["track_index"].to_list()
    elif model == "alphagenome":
        df = load_alphagenome_accession_map().filter(pl.col("accession") == accession)
        if assay is not None:
            df = df.filter(pl.col("output_type") == assay.lower())
        idx = df["track_index"].to_list()
    else:
        raise ValueError(f"{model} is not a many-tracks model")

    if not idx:
        raise ValueError(f"no {model} track for accession {accession} (assay={assay})")
    return idx


def _find_chrombpnet_model_tar(cfg: EvalConfig, accession: str, max_members_per_tar: int = 12):
    """Find the tar archive holding the model h5s for an ENCODE ChromBPNet accession.

    Iterates tar members lazily (gzip needs sequential decompression), breaking as soon as
    a `model.bias_scaled` entry appears. The resolved path is cached to JSON.
    """
    import json
    import tarfile
    from pathlib import Path

    cache_path = cfg.paths.metadata / "chrombpnet-model-tars.json"
    cache: dict[str, str] = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
        if accession in cache and Path(cache[accession]).exists():
            return Path(cache[accession])

    acc_dir = cfg.paths.weights / "chrombpnet" / accession
    found: Path | None = None
    for tar in sorted(acc_dir.glob("*.tar.gz")):
        with tarfile.open(tar, "r:*") as tf:
            for i, member in enumerate(tf):
                if "model.bias_scaled" in member.name:
                    found = tar
                    break
                if i >= max_members_per_tar:
                    break
        if found is not None:
            break

    if found is None:
        raise FileNotFoundError(f"no ChromBPNet model tar found under {acc_dir}")

    cache[accession] = str(found)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))
    return found


def _tag(scorer, *, accession: str, assay: str | None, track_indices=None, weights_path=None):
    """Attach repr-only reporting attrs (Stage 4 §7) and return the scorer."""
    scorer.accession = accession
    scorer.assay_type = assay
    if track_indices is not None:
        scorer.track_indices = track_indices
    if weights_path is not None:
        scorer.weights_path = str(weights_path)
    return scorer


def build_scorer(spec: ModelSpec, cfg: EvalConfig, ref_genome, **kw):
    """Construct + tag the scorer for a plan entry. Heavy model imports are deferred here.

    Dispatch is on `spec.kind` (chrombpnet / cherimoya / many_tracks). The returned scorer
    carries repr-only `accession`/`assay_type`/`track_indices`/`weights_path` attrs.
    """
    if spec.kind == "chrombpnet":
        return _build_chrombpnet(spec.accession, spec.assay, cfg, ref_genome, **kw)
    if spec.kind == "cherimoya":
        return _build_cherimoya(spec.accession, spec.assay, cfg, ref_genome, **kw)
    if spec.kind == "many_tracks":
        return _build_many_tracks(spec.model, spec.accession, spec.assay, cfg, ref_genome, **kw)
    raise ValueError(f"unknown scorer kind {spec.kind!r}")


def _build_chrombpnet(accession: str, assay: str | None, cfg: EvalConfig, rg, **kw):
    from variant_effect_prediction import FoldedModelWeights
    from variant_effect_prediction.scorers import ChromBPNetVariantScorer

    subpath = cfg.models.chrombpnet_celltype_dirs.get(accession)
    if subpath is not None:  # primary-cell / AFGR syn h5 folds
        fw = FoldedModelWeights.from_chrombpnet_h5_folds(cfg.paths.weights / "chrombpnet" / subpath)
    else:  # ENCODE tar (bias + nobias h5 blobs)
        tar = _find_chrombpnet_model_tar(cfg, accession)
        fw = FoldedModelWeights.from_chrombpnet_tar(tar, eid=accession)
    scorer = ChromBPNetVariantScorer(folded_weights=fw, ref_genome=rg, **kw)
    return _tag(scorer, accession=accession, assay=assay)


def _build_cherimoya(accession: str, assay: str | None, cfg: EvalConfig, rg, **kw):
    from variant_effect_prediction import FoldedModelWeights
    from variant_effect_prediction.scorers import CherimoyaVariantScorer

    dir_name = cfg.models.cherimoya_celltype_dirs.get(accession, accession)
    weights_dir = cfg.paths.weights / cfg.models.weights_subdirs["cherimoya"] / dir_name
    fw = FoldedModelWeights.from_cherimoya_dir(weights_dir)
    scorer = CherimoyaVariantScorer(folded_weights=fw, ref_genome=rg, **kw)
    return _tag(scorer, accession=accession, assay=assay, weights_path=weights_dir)


def _build_many_tracks(model: str, accession: str, assay: str | None, cfg: EvalConfig, rg, **kw):
    cap = cfg.models.many_tracks_batch[model]
    kw["batch_size"] = min(kw.get("batch_size", cap), cap)
    track_idx = track_indices_for(model, accession, assay=assay)
    wpath = cfg.paths.weights / cfg.models.weights_subdirs[model]

    if model == "enformer":
        from enformer_pytorch import Enformer
        from variant_effect_prediction.scorers import EnformerVariantScorer

        m = Enformer.from_pretrained(str(wpath))
        scorer = EnformerVariantScorer(model=m, track_idx=track_idx, ref_genome=rg, **kw)
    elif model == "borzoi":
        from borzoi_pytorch import Borzoi
        from variant_effect_prediction.scorers import BorzoiVariantScorer

        m = Borzoi.from_pretrained(str(wpath))
        scorer = BorzoiVariantScorer(model=m, track_idx=track_idx, ref_genome=rg, **kw)
    elif model == "alphagenome":
        from alphagenome_pytorch import AlphaGenome
        from variant_effect_prediction.scorers import AlphaGenomeVariantScorer

        device = kw.get("device", "cuda")
        m = AlphaGenome.from_pretrained(str(wpath), device=device)
        head = "atac" if (assay or "atac").lower() == "atac" else "dnase"
        scorer = AlphaGenomeVariantScorer(
            model=m,
            track_idx=track_idx,
            ref_genome=rg,
            wrapper_kwargs={"output_name": head, "bin_size": 128},
            **kw,
        )
    else:
        raise ValueError(model)

    return _tag(scorer, accession=accession, assay=assay, track_indices=track_idx, weights_path=wpath)


# --------------------------------------------------------------------------------
# Metrics — signed AND unsigned Spearman + Pearson
# --------------------------------------------------------------------------------


def _spearman(a, b) -> float:
    import numpy as np
    from scipy.stats import spearmanr

    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan")
    return float(spearmanr(a[mask], b[mask]).statistic)


def _pearson(a, b) -> float:
    import numpy as np
    from scipy.stats import pearsonr

    mask = ~(np.isnan(a) | np.isnan(b))
    if mask.sum() < 3:
        return float("nan")
    return float(pearsonr(a[mask], b[mask])[0])


def compute_metrics(logfc, effect_size) -> dict[str, float]:
    """Signed (raw) and unsigned (|·|) Spearman + Pearson of logFC vs effect size."""
    import numpy as np

    logfc = np.asarray(logfc, dtype=float)
    effect = np.asarray(effect_size, dtype=float)
    abs_logfc = np.abs(logfc)
    abs_effect = np.abs(effect)
    return {
        "spearman_signed": _spearman(logfc, effect),
        "spearman_unsigned": _spearman(abs_logfc, abs_effect),
        "pearson_signed": _pearson(logfc, effect),
        "pearson_unsigned": _pearson(abs_logfc, abs_effect),
    }
