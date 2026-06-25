"""Pydantic models + loader for the declarative benchmark config (config/eval.yaml).

This module is the single source of *data* for the eval suite: it parses and validates
the YAML into frozen pydantic models, resolving all repo-relative paths against the
project root (derived from the config file's location). The *logic* that consumes this
config lives in `utils.py`.

Imported both by the orchestrator (to enumerate the matrix + read cluster params) and by
`benchmark_job.run_single_benchmark` on the compute node, so it must import cleanly with
no GPU and no heavy state. `load_config()` reads the YAML fresh; the config is never
pickled — `run_single_benchmark` calls `load_config()` itself on the node.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, model_validator

# src/variant_effect_evaluation/config.py → parents[2] is the repo root (config/ sits
# directly under it). Valid under the editable install, which keeps the source in place.
DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "config" / "eval.yaml"

_FROZEN = ConfigDict(frozen=True, extra="forbid")


class Paths(BaseModel):
    """Repo-relative path layout. `project_root` is injected by `EvalConfig`'s validator;
    the absolute locations are exposed as computed properties."""

    model_config = _FROZEN

    data_dir: str = "data"
    results_dir: str = "results"
    logs_dir: str = "logs"
    qtl_subdir: str = "qtl"
    weights_subdir: str = "weights"
    references_subdir: str = "references"
    metadata_subdir: str = "metadata"
    ref_fastas: dict[str, str]
    project_root: Path | None = None

    @property
    def data(self) -> Path:
        return self.project_root / self.data_dir

    @property
    def results(self) -> Path:
        return self.project_root / self.results_dir

    @property
    def logs(self) -> Path:
        return self.project_root / self.logs_dir

    @property
    def qtl(self) -> Path:
        return self.data / self.qtl_subdir

    @property
    def weights(self) -> Path:
        return self.data / self.weights_subdir

    @property
    def references(self) -> Path:
        return self.data / self.references_subdir

    @property
    def metadata(self) -> Path:
        return self.data / self.metadata_subdir

    @property
    def ref_fasta_paths(self) -> dict[str, Path]:
        """genome build -> absolute FASTA path."""
        return {b: self.references / rel for b, rel in self.ref_fastas.items()}


class ModelSpec(BaseModel):
    """One (model, accession, assay) job in a dataset's plan. `kind` is the scorer
    dispatch key; if omitted it is derived from `model`."""

    model_config = ConfigDict(frozen=True, extra="forbid", protected_namespaces=())

    model: Literal["chrombpnet", "cherimoya", "enformer", "borzoi", "alphagenome"]
    accession: str
    assay: str | None = None
    kind: Literal["chrombpnet", "cherimoya", "many_tracks"] | None = None

    @model_validator(mode="after")
    def _derive_kind(self) -> ModelSpec:
        if self.kind is None:
            many = self.model in ("enformer", "borzoi", "alphagenome")
            return self.model_copy(update={"kind": "many_tracks" if many else self.model})
        return self


class QTLDataset(BaseModel):
    """Per-dataset config: file + column map + significance rule + assay + job plan."""

    model_config = _FROZEN

    key: str
    filename: str
    chrom_col: str
    pos_col: str  # 1-based position column
    allele1_col: str
    allele2_col: str
    effect_size_col: str
    pvalue_col: str | None = None
    isused_col: str | None = None
    source_assay: Literal["ATAC", "DNASE", "scATAC"]
    genome_build: str = "hg38"
    # Significance: positives are -log10(pvalue) > this. None => pre-filtered file.
    significance_neglog10p: float | None = None
    # Alternative significance: keep rows whose label column == 1 (dsQTLs).
    significance_label_col: str | None = None
    notes: str = ""
    plan: list[ModelSpec] = Field(min_length=1)


class ModelConventions(BaseModel):
    """Per-model construction conventions consumed by utils.build_scorer."""

    model_config = _FROZEN

    cherimoya_celltype_dirs: dict[str, str]
    chrombpnet_celltype_dirs: dict[str, str]
    many_tracks_batch: dict[str, int]
    weights_subdirs: dict[str, str]


class SlurmConfig(BaseModel):
    model_config = _FROZEN

    partition: str
    gres: str
    cpus_per_task: int
    mem: str  # SLURM needs the unit suffix, e.g. "125000MB"
    time_min: int
    array_parallelism: int
    job_name: str


class ExecutorConfig(BaseModel):
    model_config = _FROZEN

    venv_python: str = ".venv/bin/python"  # repo-relative; resolved against project_root
    dry_run_job: tuple[str, str, str, str]


class ClusterConfig(BaseModel):
    model_config = _FROZEN

    slurm: SlurmConfig
    executor: ExecutorConfig


class EvalConfig(BaseModel):
    """Root of config/eval.yaml."""

    model_config = _FROZEN

    paths: Paths
    models: ModelConventions
    datasets: dict[str, QTLDataset]
    cluster: ClusterConfig
    project_root: Path | None = None

    @model_validator(mode="after")
    def _resolve(self, info: ValidationInfo) -> EvalConfig:
        # Cross-field: every dataset's genome_build must have a ref FASTA.
        for key, ds in self.datasets.items():
            if ds.genome_build not in self.paths.ref_fastas:
                raise ValueError(
                    f"dataset {key!r}: genome_build {ds.genome_build!r} has no ref_fastas entry"
                )
        # Resolve repo-relative paths against project_root (config/ sits under root).
        config_dir = (info.context or {}).get("config_dir")
        if config_dir is None:
            return self
        root = Path(config_dir).resolve().parent
        if not root.is_dir():
            raise ValueError(f"project_root {root} is not a directory")
        paths = self.paths.model_copy(update={"project_root": root})
        return self.model_copy(update={"project_root": root, "paths": paths})

    def assert_inputs_exist(self) -> None:
        """Raise FileNotFoundError listing any missing reference FASTA or QTL TSV."""
        missing: list[str] = []
        if not self.paths.project_root or not self.paths.project_root.is_dir():
            raise FileNotFoundError(f"project_root not a directory: {self.paths.project_root}")
        for build, p in self.paths.ref_fasta_paths.items():
            if not p.exists():
                missing.append(f"ref_fasta[{build}]: {p}")
        for key, ds in self.datasets.items():
            p = self.paths.qtl / ds.filename
            if not p.exists():
                missing.append(f"qtl[{key}]: {p}")
        for d in (self.paths.weights, self.paths.references):
            if not d.is_dir():
                missing.append(f"dir: {d}")
        if missing:
            raise FileNotFoundError("missing inputs:\n  " + "\n  ".join(missing))


def load_config(path: str | Path | None = None) -> EvalConfig:
    """Parse + validate config/eval.yaml into a frozen EvalConfig with resolved paths.

    `path` defaults to <repo>/config/eval.yaml (resolved relative to this module).
    project_root is derived from the config file's parent directory.
    """
    cfg_path = Path(path).resolve() if path else DEFAULT_CONFIG
    raw = yaml.safe_load(cfg_path.read_text())
    return EvalConfig.model_validate(raw, context={"config_dir": cfg_path.parent})
