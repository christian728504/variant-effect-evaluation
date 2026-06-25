"""Static matplotlib bar charts of the variant-effect benchmark (`pearson_signed`).

No marimo/altair/quak, just matplotlib. `render_all(cfg)` reads
`<results>/all_benchmarks.parquet` and writes one fixed-scope bar chart PNG per dataset
subset into that same `cfg.paths.results` directory. Driven by the CLI:

    variant-effect-evaluation plot
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import polars as pl
from loguru import logger
from matplotlib.patches import Patch

from .config import EvalConfig

mpl.rcParams["figure.dpi"] = 300
mpl.rcParams["font.family"] = "SF Pro Text"
mpl.rcParams["figure.constrained_layout.use"] = True

MODEL_DOMAIN = ["alphagenome", "borzoi", "chrombpnet", "cherimoya", "enformer"]
MODEL_RANGE = ["#6a93cb", "#e2e872", "#a9a8ab", "#44c6b4", "#5e9c0a"]

# Display names for legend + tick labels (raw bench values are lowercase).
MODEL_DISPLAY = {
    "alphagenome": "AlphaGenome",
    "borzoi": "Borzoi",
    "chrombpnet": "ChromBPNet",
    "cherimoya": "Cherimoya",
    "enformer": "Enformer",
}

# One fixed-scope bar chart per subset. dsqtls_yoruba + bqtls_pu1 are the new
# hg19 GM12878 LCL datasets — same ENCODE ATAC+DNASE accessions as caqtls_eu.
PLOTS = [
    dict(
        filename="bench_caqtls_african_encode_all_models_pearson_signed.png",
        title="caQTL African — ENCODE ATAC + DNASE",
        datasets=["caqtls_african"],
        accessions=["ENCSR000EMT", "ENCSR637XSC"],
        models=None,
    ),
    dict(
        filename="bench_caqtls_african_afgr_cheri_chrombpnet_pearson_signed.png",
        title="caQTL African — AFGR subpopulations",
        datasets=["caqtls_african"],
        accessions=["esan", "gambian", "luhya", "maasai", "mende", "yoruba"],
        models=["cherimoya", "chrombpnet"],
    ),
    dict(
        filename="bench_asb_african_encode_all_models_pearson_signed.png",
        title="ASB African — ENCODE ATAC + DNASE",
        datasets=["asb_african"],
        accessions=["ENCSR000EMT", "ENCSR637XSC"],
        models=None,
    ),
    dict(
        filename="bench_caqtls_eu_encode_all_models_pearson_signed.png",
        title="caQTL European — ENCODE ATAC + DNASE",
        datasets=["caqtls_eu"],
        accessions=["ENCSR000EMT", "ENCSR637XSC"],
        models=None,
    ),
    dict(
        filename="bench_caqtls_microglia_cheri_chrombpnet_pearson_signed.png",
        title="caQTL Microglia",
        datasets=["caqtls_microglia"],
        accessions=["microglia"],
        models=["cherimoya", "chrombpnet"],
    ),
    dict(
        filename="bench_dsqtls_yoruba_encode_all_models_pearson_signed.png",
        title="dsQTL Yoruba LCL (hg19) — ENCODE ATAC + DNASE",
        datasets=["dsqtls_yoruba"],
        accessions=["ENCSR000EMT", "ENCSR637XSC"],
        models=None,
    ),
    dict(
        filename="bench_bqtls_pu1_encode_all_models_pearson_signed.png",
        title="bQTL PU1/SPI1 LCL (hg19) — ENCODE ATAC + DNASE",
        datasets=["bqtls_pu1"],
        accessions=["ENCSR000EMT", "ENCSR637XSC"],
        models=None,
    ),
]


def render(spec: dict, bench: pl.DataFrame, out_dir: Path) -> tuple[str, int]:
    """Render one subset bar chart to a PNG; return (path, bar_count).

    Bars are sorted by descending `pearson_signed` and colored by model. Rows
    with a null `pearson_signed` (errored or not-yet-scored jobs) are dropped.
    Returns a bar count of 0 (and writes nothing) when no benchmark row matches
    the subset — e.g. a dataset that has not been scored/collected yet.
    """
    df = bench.filter(pl.col("dataset").is_in(spec["datasets"]))
    df = df.filter(pl.col("accession").is_in(spec["accessions"]))
    if spec["models"]:
        df = df.filter(pl.col("model").is_in(spec["models"]))
    df = df.select("model", "dataset", "accession", "assay", "pearson_signed").sort(
        "pearson_signed", descending=True
    )

    n = df.height
    if n == 0:
        return spec["filename"], 0

    bar_models = df["model"].to_list()
    bar_accs = df["accession"].to_list()
    bar_assays = df["assay"].to_list()
    # ENCODE accessions stay as-is (e.g. ENCSR000EMT); AFGR subpops and
    # "microglia" get capitalized — "esan" → "Esan", "microglia" → "Microglia".
    _disp_acc = lambda acc: acc if acc.isupper() else acc.capitalize()
    # "Model Accession (Assay)"; drop the parenthetical when assay is "NA".
    labels = [
        (
            f"{MODEL_DISPLAY.get(m, m)} {_disp_acc(acc)} ({a})"
            if a and a != "NA"
            else f"{MODEL_DISPLAY.get(m, m)} {_disp_acc(acc)}"
        )
        for m, acc, a in zip(bar_models, bar_accs, bar_assays)
    ]
    values = df["pearson_signed"].to_list()
    bar_colors = [MODEL_RANGE[MODEL_DOMAIN.index(m)] for m in bar_models]

    # Width scales gently with bar count so dense plots don't collide.
    fig, ax = plt.subplots(figsize=(max(6.0, 0.65 * n + 3.0), 5.5))
    xs = list(range(n))
    ax.bar(xs, values, color=bar_colors, edgecolor="none")
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, rotation=-45, ha="left", fontsize=8)
    ax.set_ylabel("Pearson (signed)")
    ax.set_xlabel("Model Accession (Assay)")
    ax.set_title(spec["title"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Bar-tip value labels: above positive bars, below negative bars (3 d.p.).
    # ε scales with the y-span so labels float a constant visual offset.
    y_lo, y_hi = ax.get_ylim()
    eps = 0.012 * (y_hi - y_lo)
    for x, v in zip(xs, values):
        if v >= 0:
            ax.text(x, v + eps, f"{v:.3f}", ha="center", va="bottom", fontsize=8)
        else:
            ax.text(x, v - eps, f"{v:.3f}", ha="center", va="top", fontsize=8)

    # Legend ordered canonically (MODEL_DOMAIN), only models that appear.
    # `loc="outside center right"` is constrained-layout-native — keeps the
    # legend out of the bar area and out of the way of value labels.
    present = [m for m in MODEL_DOMAIN if m in set(bar_models)]
    handles = [
        Patch(
            facecolor=MODEL_RANGE[MODEL_DOMAIN.index(m)], label=MODEL_DISPLAY.get(m, m)
        )
        for m in present
    ]
    if handles:
        fig.legend(
            handles=handles,
            title="Model",
            loc="outside center right",
            frameon=False,
            fontsize=8,
        )

    out_path = out_dir / spec["filename"]
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return str(out_path), n


def render_all(cfg: EvalConfig) -> int:
    """Render every subset bar chart from all_benchmarks.parquet into cfg.paths.results.

    Returns 1 (and renders nothing) when the aggregated parquet is absent — run
    `collect` first.
    """
    out_dir = cfg.paths.results
    all_bench_path = out_dir / "all_benchmarks.parquet"
    if not all_bench_path.exists():
        logger.warning("missing {} — run `collect` first", all_bench_path)
        return 1

    bench = pl.read_parquet(all_bench_path)

    logger.info("rendering static bar charts from {}", all_bench_path)
    for spec in PLOTS:
        path, n = render(spec, bench, out_dir)
        if n:
            logger.success("wrote {} — {} bars", path, n)
        else:
            logger.warning("skipped {} — no matching benchmarks yet", spec["filename"])
    return 0
