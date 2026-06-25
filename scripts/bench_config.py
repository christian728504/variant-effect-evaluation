"""Stage 4 benchmark configuration — picklable, importable on a SLURM compute node.

Self-contained re-implementation of the reusable benchmark logic (dataset registry
+ column maps, per-dataset model plans, significance rules, scorer dispatch,
AFGR/microglia weight dirs, ChromBPNet model-tar discovery, many-tracks track-index
lookup, metric helpers). Deliberately does NOT import from `evals/old/` — that tree
is archival only.

This module is imported both by the orchestrator (to enumerate the job matrix) and by
`benchmark_job.run_single_benchmark` on the compute node, so it must import cleanly
without a GPU and carry no heavy state at import time (all torch/model imports are
deferred into the scorer factories).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from loguru import logger


def configure_logging(level: str = "INFO") -> None:
    """Reset loguru to a single stderr sink at `level` using loguru's default format.

    Idempotent: removes any existing sinks and installs one stderr sink. Keeps the
    stock loguru look (green time, level-colored level/message, cyan
    name:function:line) so the whole eval suite renders consistently. Call once
    from each entry point (orchestrator / job runner).
    """
    logger.remove()
    logger.add(sys.stderr, level=level)


PROJECT_ROOT = Path("/zata/zippy/ramirezc/Projects/variant-effect-evaluation")
# All eval inputs live under data/ (a clean copy of just the subset the matrix
# needs); outputs go directly under the project root.
DATA_DIR = PROJECT_ROOT / "data"
BENCH_DIR = DATA_DIR / "qtl"
# Reference FASTA per genome build. Both are UCSC `chr1`-named, bgzip + faidx
# indexed; pysam/RefGenome is build-agnostic, so a dataset selects its build via
# `QTLDataset.genome_build`. GRCh37 carries the hg19-only dsQTL/bQTL datasets.
_REF_DIR = DATA_DIR / "references"
REF_FASTAS: dict[str, Path] = {
    "GRCh38": _REF_DIR / "GRCh38" / "GRCh38_no_alt_analysis_set_GCA_000001405.15.fasta.gz",
    "GRCh37": _REF_DIR / "GRCh37" / "GCF_000001405.25_GRCh37.p13_genomic.fa.gz",
}
WEIGHTS = DATA_DIR / "weights"
RESULTS_DIR = PROJECT_ROOT / "results"
LOGS_DIR = PROJECT_ROOT / "logs"


# --------------------------------------------------------------------------------
# Dataset registry + significance rules
# --------------------------------------------------------------------------------


@dataclass(frozen=True)
class QTLDataset:
    """Per-dataset config: file + column map + significance cutoff + assay."""

    key: str
    filename: str
    chrom_col: str
    pos_col: str  # 1-based position column
    allele1_col: str
    allele2_col: str
    effect_size_col: str
    pvalue_col: str | None
    isused_col: str | None
    source_assay: str  # "ATAC" | "DNASE" | "scATAC"
    # Reference build the position column is in; selects REF_FASTAS[genome_build].
    genome_build: str = "GRCh38"
    # Significance: positives are -log10(pvalue) > this. None => the file is
    # already pre-filtered to significant variants (no extra cutoff).
    # From resources/context/defining-significant-qtls.md.
    significance_neglog10p: float | None = None
    # Alternative significance: keep rows whose categorical label column == 1
    # (positives), with the rest being controls. Used by dsQTLs (obs.label).
    significance_label_col: str | None = None
    notes: str = ""

    @property
    def path(self) -> Path:
        return BENCH_DIR / self.filename


DATASETS: dict[str, QTLDataset] = {
    "caqtls_eu": QTLDataset(
        key="caqtls_eu",
        filename="caqtls.eu.lcls.benchmarking.all.tsv",
        chrom_col="var.chr",
        pos_col="var.pos_hg38",
        allele1_col="var.allele1",
        allele2_col="var.allele2",
        effect_size_col="obs.beta",
        pvalue_col="obs.pval",
        isused_col="var.isused",
        source_assay="ATAC",
        significance_neglog10p=6.0,
        notes="European LCL ATAC caQTLs; significant = -log10(p)>6.",
    ),
    "caqtls_african": QTLDataset(
        key="caqtls_african",
        filename="caqtls.african.lcls.benchmarking.all.tsv",
        chrom_col="var.chr",
        pos_col="var.pos_hg38",
        allele1_col="var.allele1",
        allele2_col="var.allele2",
        effect_size_col="obs.beta",
        pvalue_col="obs.pval",
        isused_col="var.isused",
        source_assay="ATAC",
        significance_neglog10p=5.0,
        notes="African LCL ATAC caQTLs; significant = -log10(p)>5.",
    ),
    "asb_african": QTLDataset(
        key="asb_african",
        filename="caqtls.african.lcls.asb.benchmarking.all.tsv",
        chrom_col="var.chr",
        pos_col="var.pos_hg38",
        allele1_col="allele1",  # ASB file drops the var. prefix on alleles
        allele2_col="allele2",
        effect_size_col="obs.meanLog2FC",
        pvalue_col=None,
        isused_col="var.isused",
        source_assay="ATAC",
        significance_neglog10p=None,  # all rows already significant ASB sites
        notes="African LCL allele-specific binding; all rows pre-significant.",
    ),
    "caqtls_microglia": QTLDataset(
        key="caqtls_microglia",
        filename="caqtls.microglia.benchmarking.all.tsv",
        chrom_col="var.chr",
        pos_col="var.pos_hg38",
        # obs.Beta is signed in the var.allele1 direction (opposite of the caQTL
        # LCL obs.beta convention), so allele1/allele2 are swapped — same flip as
        # dsqtls_yoruba. Matches the file's published logFC + the paper's r=0.6
        # (Fig. 6j): signed corr is +0.60, not -0.60.
        allele1_col="var.allele2",
        allele2_col="var.allele1",
        effect_size_col="obs.Beta",
        pvalue_col="obs.Z_score_fixed",
        isused_col="var.isused",
        source_assay="scATAC",
        significance_neglog10p=None,  # all rows already significant caQTLs
        notes="Microglia scATAC caQTLs; all rows pre-significant; BPNet-like only.",
    ),
    "dsqtls_yoruba": QTLDataset(
        key="dsqtls_yoruba",
        filename="dsqtls.yoruba.lcls.benchmarking.all.tsv",
        chrom_col="var.chr",
        pos_col="var.pos_hg19",
        # obs.estimate is signed in the var.allele1 direction (opposite of the
        # caQTL obs.beta convention), so allele1/allele2 are swapped to match the
        # pipeline's logFC=log2(allele2/allele1): with var.allele2 as the ref-like
        # allele, signed corr matches the file's published logFC (+0.75, not -0.75).
        allele1_col="var.allele2",
        allele2_col="var.allele1",
        effect_size_col="obs.estimate",
        pvalue_col=None,
        isused_col="var.isused",
        source_assay="DNASE",
        genome_build="GRCh37",
        significance_label_col="obs.label",  # 1 = significant, -1 = control
        notes="Yoruban LCL dsQTLs (GRCh37); significant = obs.label==1 (~560 positives).",
    ),
    "bqtls_pu1": QTLDataset(
        key="bqtls_pu1",
        filename="bqtls.pu1.lcls.benchmarking.all.tsv",
        chrom_col="var.chr",
        pos_col="var.pos_hg19",
        allele1_col="var.POSTallele",  # POST = reference-like allele
        allele2_col="var.ALTallele",
        effect_size_col="obs.chiplogratio",
        pvalue_col="obs.pval",
        isused_col="var.isused",
        source_assay="ATAC",
        genome_build="GRCh37",
        significance_neglog10p=4.0,
        notes="PU1/SPI1 bQTLs (GRCh37); significant = -log10(obs.pval)>4.",
    ),
}


# AFGR ancestry-specific LCL subpopulations (lowercase keys; capitalized dirs).
AFGR_POPS = ("esan", "gambian", "luhya", "maasai", "mende", "yoruba")


# Per-dataset evaluation plan — every (model, accession, assay) that can score the
# dataset. BPNet-like + AlphaGenome use the assay-matched accession (ATAC); Enformer/
# Borzoi have no ATAC track so they use the biosample's DNase accession as a proxy.
# African caQTLs additionally get the 6 AFGR ChromBPNet + Cherimoya models. Microglia
# is a primary cell: only the cell-type-specific BPNet-like models exist.
DATASET_MODEL_PLANS: dict[str, list[tuple[str, str, str | None]]] = {
    "caqtls_eu": [
        ("chrombpnet", "ENCSR637XSC", "ATAC"),
        ("cherimoya", "ENCSR637XSC", "ATAC"),
        ("alphagenome", "ENCSR637XSC", "ATAC"),
        ("chrombpnet", "ENCSR000EMT", "DNASE"),
        ("cherimoya", "ENCSR000EMT", "DNASE"),
        ("alphagenome", "ENCSR000EMT", "DNASE"),
        ("enformer", "ENCSR000EMT", "DNASE"),
        ("borzoi", "ENCSR000EMT", "DNASE"),
    ],
    "caqtls_african": [
        ("chrombpnet", "ENCSR637XSC", "ATAC"),
        ("cherimoya", "ENCSR637XSC", "ATAC"),
        ("alphagenome", "ENCSR637XSC", "ATAC"),
        ("chrombpnet", "ENCSR000EMT", "DNASE"),
        ("cherimoya", "ENCSR000EMT", "DNASE"),
        ("alphagenome", "ENCSR000EMT", "DNASE"),
        ("enformer", "ENCSR000EMT", "DNASE"),
        ("borzoi", "ENCSR000EMT", "DNASE"),
        *[("chrombpnet", pop, "ATAC") for pop in AFGR_POPS],
        *[("cherimoya", pop, "ATAC") for pop in AFGR_POPS],
    ],
    "asb_african": [
        ("chrombpnet", "ENCSR637XSC", "ATAC"),
        ("cherimoya", "ENCSR637XSC", "ATAC"),
        ("alphagenome", "ENCSR637XSC", "ATAC"),
        ("chrombpnet", "ENCSR000EMT", "DNASE"),
        ("cherimoya", "ENCSR000EMT", "DNASE"),
        ("alphagenome", "ENCSR000EMT", "DNASE"),
        ("enformer", "ENCSR000EMT", "DNASE"),
        ("borzoi", "ENCSR000EMT", "DNASE"),
    ],
    "caqtls_microglia": [
        ("chrombpnet", "microglia", None),
        ("cherimoya", "microglia", None),
    ],
    # dsQTL/bQTL are GM12878 LCL datasets (GRCh37) — same biosample as caqtls_eu, so
    # the same 8-model plan applies (ATAC ENCSR637XSC + DNase ENCSR000EMT).
    "dsqtls_yoruba": [
        ("chrombpnet", "ENCSR637XSC", "ATAC"),
        ("cherimoya", "ENCSR637XSC", "ATAC"),
        ("alphagenome", "ENCSR637XSC", "ATAC"),
        ("chrombpnet", "ENCSR000EMT", "DNASE"),
        ("cherimoya", "ENCSR000EMT", "DNASE"),
        ("alphagenome", "ENCSR000EMT", "DNASE"),
        ("enformer", "ENCSR000EMT", "DNASE"),
        ("borzoi", "ENCSR000EMT", "DNASE"),
    ],
    "bqtls_pu1": [
        ("chrombpnet", "ENCSR637XSC", "ATAC"),
        ("cherimoya", "ENCSR637XSC", "ATAC"),
        ("alphagenome", "ENCSR637XSC", "ATAC"),
        ("chrombpnet", "ENCSR000EMT", "DNASE"),
        ("cherimoya", "ENCSR000EMT", "DNASE"),
        ("alphagenome", "ENCSR000EMT", "DNASE"),
        ("enformer", "ENCSR000EMT", "DNASE"),
        ("borzoi", "ENCSR000EMT", "DNASE"),
    ],
}


def iter_jobs() -> list[tuple[str, str, str, str]]:
    """Flatten the matrix to a list of (dataset, model, accession, assay) tuples.

    `assay` is stringified ("NA" when None) so every job key is a plain-string
    4-tuple — picklable and usable directly in result filenames.
    """
    jobs: list[tuple[str, str, str, str]] = []
    for dataset, plan in DATASET_MODEL_PLANS.items():
        for model, accession, assay in plan:
            jobs.append((dataset, model, accession, assay if assay else "NA"))
    return jobs


def job_stem(dataset: str, model: str, accession: str, assay: str | None) -> str:
    """Canonical filesystem-safe stem for a job's result/log files."""
    return f"{dataset}__{model}__{accession}__{assay if assay else 'NA'}"


# --------------------------------------------------------------------------------
# VariantSet construction (significance + SNV + isused filtering)
# --------------------------------------------------------------------------------


def ref_genome(build: str = "GRCh38"):
    from variant_effect_prediction import RefGenome

    return RefGenome(REF_FASTAS[build])


def build_variant_set(ds: QTLDataset, *, significant_only: bool = True):
    """Load a QTL TSV → 0-based start → (optional) significance filter → VariantSet.

    The scored set is SNV ∩ significant ∩ isused: SNV-only is applied by
    `VariantSet.from_dataframe(snvs_only=True)`, the significance cutoff is applied
    here on the raw p-value column, and isused is applied at score-time by
    `vs.used()`. The pre-filter row count + the applied filters are recorded in
    `meta` for the state report.
    """
    from variant_effect_prediction import VariantSet

    df = pl.read_csv(ds.path, separator="\t", infer_schema_length=2**13)
    n_total = df.height
    df = df.with_columns((pl.col(ds.pos_col) - 1).cast(pl.Int64).alias("__start0"))

    filters: list[str] = ["snvs_only"]
    if significant_only and ds.significance_neglog10p is not None:
        cutoff = ds.significance_neglog10p
        # -log10(p) > cutoff  ⇔  significant. log10(0)→-inf ⇒ inf>cutoff (kept);
        # null p-values yield null ⇒ dropped by filter.
        df = df.filter(-(pl.col(ds.pvalue_col).log10()) > cutoff)
        filters.append(f"-log10({ds.pvalue_col})>{cutoff}")
    if significant_only and ds.significance_label_col is not None:
        # Categorical significance: 1 = positive (significant), -1 = control.
        df = df.filter(pl.col(ds.significance_label_col) == 1)
        filters.append(f"{ds.significance_label_col}==1")
    if ds.isused_col is not None:
        filters.append(f"isused({ds.isused_col})")

    meta = {
        "name": ds.key,
        "source_path": str(ds.path),
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

_AFGR_CHROMBPNET_ROOT = WEIGHTS / "chrombpnet" / "afgr"

# Cherimoya .torch dirs for non-ENCODE models (primary cells + AFGR subpops), keyed
# by the benchmark's pseudo-accession (ENCODE accessions map to themselves).
CHERIMOYA_CELLTYPE_DIRS = {
    "microglia": "Microglia",
    **{pop: pop.capitalize() for pop in AFGR_POPS},
}

# ChromBPNet (TF/h5) model dirs in the syn59449898 archive (models/fold_N/*.h5).
CHROMBPNET_CELLTYPE_DIRS = {
    "microglia": WEIGHTS / "chrombpnet" / "microglia" / "models",
    **{pop: _AFGR_CHROMBPNET_ROOT / pop.capitalize() / "models" for pop in AFGR_POPS},
}

# Many-tracks track metadata now ships inside the library (the loaders read their
# own bundled parquet — no paths to pass).

# Long sequences make many-tracks activations huge; cap batch even on a 140 GB H200.
_MANY_TRACKS_BATCH = {"enformer": 2, "borzoi": 2, "alphagenome": 1}

_MODEL_TAR_CACHE = DATA_DIR / "metadata" / "chrombpnet-model-tars.json"


def track_indices_for(model: str, accession: str, assay: str | None = None) -> list[int]:
    """Look up a many-tracks model's track index(es) for an ENCODE accession.

    For AlphaGenome `track_index` is per-head, so `assay` (ATAC/DNASE) selects the
    head and must be passed to disambiguate.
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


def _find_chrombpnet_model_tar(accession: str, max_members_per_tar: int = 12) -> Path:
    """Find the tar archive holding the model h5s for an ENCODE ChromBPNet accession.

    Iterates tar members lazily (gzip needs sequential decompression), breaking as
    soon as a `model.bias_scaled` entry appears. The resolved path is cached to JSON.
    """
    import json
    import tarfile

    cache: dict[str, str] = {}
    if _MODEL_TAR_CACHE.exists():
        cache = json.loads(_MODEL_TAR_CACHE.read_text())
        if accession in cache and Path(cache[accession]).exists():
            return Path(cache[accession])

    acc_dir = WEIGHTS / "chrombpnet" / accession
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
    _MODEL_TAR_CACHE.parent.mkdir(parents=True, exist_ok=True)
    _MODEL_TAR_CACHE.write_text(json.dumps(cache, indent=2))
    return found


def _tag(scorer, *, accession: str, assay: str | None,
         track_indices=None, weights_path=None):
    """Attach repr-only reporting attrs (Stage 4 §7) and return the scorer."""
    scorer.accession = accession
    scorer.assay_type = assay
    if track_indices is not None:
        scorer.track_indices = track_indices
    if weights_path is not None:
        scorer.weights_path = str(weights_path)
    return scorer


def build_scorer(model: str, accession: str, assay: str | None, ref_genome, **kw):
    """Construct + tag the scorer for a (model, accession, assay) config.

    `accession` is an ENCODE accession (ENCSR...) or a celltype/pop key (microglia,
    esan, ...). All heavy model imports are deferred into here. The returned scorer
    carries repr-only `accession`/`assay_type`/`track_indices`/`weights_path` attrs.
    """
    if model == "chrombpnet":
        return _build_chrombpnet(accession, assay, ref_genome, **kw)
    if model == "cherimoya":
        return _build_cherimoya(accession, assay, ref_genome, **kw)
    if model in ("enformer", "borzoi", "alphagenome"):
        return _build_many_tracks(model, accession, assay, ref_genome, **kw)
    raise ValueError(f"unknown model {model!r}")


def _build_chrombpnet(accession: str, assay: str | None, rg, **kw):
    from variant_effect_prediction import FoldedModelWeights
    from variant_effect_prediction.scorers import ChromBPNetVariantScorer

    if accession in CHROMBPNET_CELLTYPE_DIRS:  # primary-cell syn h5 folds
        fw = FoldedModelWeights.from_chrombpnet_h5_folds(
            CHROMBPNET_CELLTYPE_DIRS[accession]
        )
    else:  # ENCODE tar (bias + nobias h5 blobs)
        tar = _find_chrombpnet_model_tar(accession)
        fw = FoldedModelWeights.from_chrombpnet_tar(tar, eid=accession)
    scorer = ChromBPNetVariantScorer(folded_weights=fw, ref_genome=rg, **kw)
    return _tag(scorer, accession=accession, assay=assay)


def _build_cherimoya(accession: str, assay: str | None, rg, **kw):
    from variant_effect_prediction import FoldedModelWeights
    from variant_effect_prediction.scorers import CherimoyaVariantScorer

    dir_name = CHERIMOYA_CELLTYPE_DIRS.get(accession, accession)
    weights_dir = WEIGHTS / "cherimoya" / "models" / dir_name
    fw = FoldedModelWeights.from_cherimoya_dir(weights_dir)
    scorer = CherimoyaVariantScorer(folded_weights=fw, ref_genome=rg, **kw)
    return _tag(scorer, accession=accession, assay=assay, weights_path=weights_dir)


def _build_many_tracks(model: str, accession: str, assay: str | None, rg, **kw):
    # Force a safe batch size unless the caller asked for a smaller one.
    kw["batch_size"] = min(
        kw.get("batch_size", _MANY_TRACKS_BATCH[model]), _MANY_TRACKS_BATCH[model]
    )
    track_idx = track_indices_for(model, accession, assay=assay)

    if model == "enformer":
        from enformer_pytorch import Enformer
        from variant_effect_prediction.scorers import EnformerVariantScorer

        wpath = WEIGHTS / "enformer-pytorch"
        m = Enformer.from_pretrained(str(wpath))
        scorer = EnformerVariantScorer(
            model=m, track_idx=track_idx, ref_genome=rg, **kw
        )
    elif model == "borzoi":
        from borzoi_pytorch import Borzoi
        from variant_effect_prediction.scorers import BorzoiVariantScorer

        wpath = WEIGHTS / "borzoi-pytorch"
        m = Borzoi.from_pretrained(str(wpath))
        scorer = BorzoiVariantScorer(
            model=m, track_idx=track_idx, ref_genome=rg, **kw
        )
    elif model == "alphagenome":
        from alphagenome_pytorch import AlphaGenome
        from variant_effect_prediction.scorers import AlphaGenomeVariantScorer

        wpath = WEIGHTS / "alphagenome-pytorch" / "model_all_folds.safetensors"
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

    return _tag(
        scorer, accession=accession, assay=assay,
        track_indices=track_idx, weights_path=wpath,
    )


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
