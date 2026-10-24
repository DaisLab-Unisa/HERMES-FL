"""Shared loading, constants, and plot style of the HERMES-FL result figures.

The four result CSVs (star/hierarchical x plaintext/CKKS) are read from
star/results/ and hierarchical/results/; the figures are written to plots/
(PDF, plus a PNG used by the README).

Requirements: matplotlib 3.10.9, seaborn, pandas, numpy, pillow, pymupdf 1.24.14.
"""
import argparse
import io
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pymupdf

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
RESULT_FILES = {("star", "plain"): REPO / "star" / "results" / "star_no_enc.csv",
                ("star", "ckks"): REPO / "star" / "results" / "star_ckks.csv",
                ("hier", "plain"): REPO / "hierarchical" / "results" / "hierarchical_no_enc.csv",
                ("hier", "ckks"): REPO / "hierarchical" / "results" / "hierarchical_ckks.csv"}

# --------------------------------------------------------------------------- #
# Grid definition
# --------------------------------------------------------------------------- #
DATASETS = {  # csv name -> (paper name, task, domain)
    "dataset_slices_sorted": ("MS-MRI-MULTI", "binary", "MS"),
    "dataset_slices_sorted_Flair": ("MS-MRI-FLAIR", "binary", "MS"),
    "dataset_slices_sorted_T1": ("MS-MRI-T1", "binary", "MS"),
    "dataset_slices_sorted_T2": ("MS-MRI-T2", "binary", "MS"),
    "dataset_slices_sorted_augmented": ("MS-MRI-MULTI (Aug)", "binary", "MS"),
    "dataset_slices_sorted_Flair_augmented": ("MS-MRI-FLAIR (Aug)", "binary", "MS"),
    "dataset_slices_sorted_T1_augmented": ("MS-MRI-T1 (Aug)", "binary", "MS"),
    "dataset_slices_sorted_T2_augmented": ("MS-MRI-T2 (Aug)", "binary", "MS"),
    "dataset_HAM1000": ("SLM-HAM10000", "binary", "SLM"),
    "dataset_ISIC": ("SLM-ISIC1", "binary", "SLM"),
    "dataset_ISIC2": ("SLM-ISIC2", "binary", "SLM"),
    "dataset_RD_colored_1": ("DR-COLORED", "binary", "DR"),
    "dataset_RD_kdr299_1": ("DR-KDR299", "binary", "DR"),
    "dataset_multy_sorted": ("MS-MRI-MULTI", "multiclass", "MS"),
    "dataset_multy_sorted_Flair": ("MS-MRI-FLAIR", "multiclass", "MS"),
    "dataset_multy_sorted_T1": ("MS-MRI-T1", "multiclass", "MS"),
    "dataset_multy_sorted_T2": ("MS-MRI-T2", "multiclass", "MS"),
    "dataset_multy_sorted_augmented": ("MS-MRI-MULTI (Aug)", "multiclass", "MS"),
    "dataset_multy_sorted_Flair_augmented": ("MS-MRI-FLAIR (Aug)", "multiclass", "MS"),
    "dataset_multy_sorted_T1_augmented": ("MS-MRI-T1 (Aug)", "multiclass", "MS"),
    "dataset_multy_sorted_T2_augmented": ("MS-MRI-T2 (Aug)", "multiclass", "MS"),
    "brain_tumor": ("BT-MRI", "multiclass", "BT"),
    "dataset_RD_colored": ("DR-COLORED", "multiclass", "DR"),
    "dataset_RD_kdr299": ("DR-KDR299", "multiclass", "DR"),
}
DOMAINS = {"binary": ["MS", "SLM", "DR"], "multiclass": ["MS", "BT", "DR"]}
MODELS = ["ConvNet", "AlexNet", "GoogLeNet", "ResNet18", "ResNet34", "ResNet50", "ResNet101"]
MODEL_LABEL = {"ConvNet": "ConvNet", "AlexNet": "AlexNet", "GoogLeNet": "GoogLeNet",
               "ResNet18": "ResNet-18", "ResNet34": "ResNet-34", "ResNet50": "ResNet-50",
               "ResNet101": "ResNet-101"}
# trainable (= exchanged) parameters for a binary task, head depth L=2 (Table 2);
# ConvNet: average of the two input resolutions
PARAMS = {"ConvNet": (215962 + 682906) / 2, "AlexNet": 2360066, "GoogLeNet": 262914,
          "ResNet18": 131842, "ResNet34": 131842, "ResNet50": 525058, "ResNet101": 525058}
N_VALUES = [2, 4, 8]
ALGS = ["FedAvg", "FedProx"]
TOPOLOGIES, ENCS = ["star", "hier"], ["plain", "ckks"]
SETTINGS = [("star", "plain"), ("star", "ckks"), ("hier", "plain"), ("hier", "ckks")]
CFG = ["dataset_name", "model_name", "global_epoch", "local_epoch", "num_clients",
       "aggregation_algorithm", "image_size", "num_custom_layers"]


def load_results() -> pd.DataFrame:
    """The four result CSVs in one frame, with topology, encryption, task, dataset, and domain."""
    frames = []
    for (topo, enc), path in RESULT_FILES.items():
        d = pd.read_csv(path, low_memory=False)
        d["topology"], d["enc"] = topo, enc
        frames.append(d)
    full = pd.concat(frames, ignore_index=True)
    full["task"] = full.dataset_name.map(lambda x: DATASETS[x][1])
    full["dataset"] = full.dataset_name.map(lambda x: DATASETS[x][0])
    full["domain"] = full.dataset_name.map(lambda x: DATASETS[x][2])
    return full


def output_dir() -> Path:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(REPO / "plots"),
                        help="output directory of the figures (default: plots/)")
    out = Path(parser.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------------------------------------------------------- #
# Plot style
# --------------------------------------------------------------------------- #
# matplotlib defaults (seaborn's original-style reset) plus the paper's sizes
sns.reset_orig()
plt.rcParams.update(matplotlib.rcParamsDefault)
plt.rcParams.update({"font.size": 14, "axes.titlesize": 14, "axes.labelsize": 15,
                     "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 14,
                     "pdf.fonttype": 42, "hatch.linewidth": 1.2, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.05})

GRID_MAJOR = dict(color="0.72", lw=0.8)
GRID_MINOR = dict(color="0.86", lw=0.5, ls=":")

# soft pastel blues (star) and reds (hierarchical): light = FedAvg, dark = FedProx
_BAR_PALETTE = sns.color_palette(["#c3d9ee", "#7fa8d4", "#f5c6c4", "#de8f8b"])
BAR_COLORS = dict(zip([("star", "FedAvg"), ("star", "FedProx"), ("hier", "FedAvg"), ("hier", "FedProx")],
                      _BAR_PALETTE))
TOPO_COLORS = dict(zip(["star", "hier"], sns.color_palette(["#76c1b2", "#a99bd6"])))   # teal / lavender
PAIR_LABEL = {"plain": "PLAINTEXT", "ckks": "CKKS", "star": "STAR", "hier": "HIERARCHICAL"}
HATCH = {"plain": "", "ckks": "//"}               # CKKS bars are hatched
STAR_FACE, STAR_EDGE = "#f2c12e", "#9a7400"      # readable dark-yellow star

# The bar charts and their legends are published as raster images: PNG at RASTER_DPI, wrapped in a
# single-page PDF whose page is the image at the given density (dpi). The run-time figure and its
# legend are vector PDFs.
RASTER_DPI = 90
PAGE_DENSITY = {"binary_enc_domain": 96, "multiclass_enc_domain": 96, "legend_topo": 96,
                "binary_topo_backbone": 102, "multiclass_topo_backbone": 102, "legend_bars": 102}


def _g(x):
    return f"{x:g}"


def raster_pdf(png, pdf_path: Path, density: float):
    """Single-page PDF holding the PNG (RGB + alpha soft mask) scaled to `density` dpi."""
    im = np.asarray(Image.open(png).convert("RGBA"))
    h, w = im.shape[:2]
    W, H = _g(w * 72 / density), _g(h * 72 / density)
    rgb = zlib.compress(np.ascontiguousarray(im[:, :, :3]).tobytes())
    alpha = zlib.compress(np.ascontiguousarray(im[:, :, 3]).tobytes())
    content = f"q\n{W} 0 0 {H} 0 0 cm\n/Im0 Do\nQ\n".encode()
    objs = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W} {H}] "
         f"/Resources << /XObject << /Im0 5 0 R >> >> /Contents 4 0 R >>").encode(),
        b"<< /Length %d >>\nstream\n" % len(content) + content + b"\nendstream",
        (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceRGB "
         f"/BitsPerComponent 8 /Filter /FlateDecode /SMask 6 0 R /Length {len(rgb)} >>\nstream\n").encode()
        + rgb + b"\nendstream",
        (f"<< /Type /XObject /Subtype /Image /Width {w} /Height {h} /ColorSpace /DeviceGray "
         f"/BitsPerComponent 8 /Filter /FlateDecode /Length {len(alpha)} >>\nstream\n").encode()
        + alpha + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for k, o in enumerate(objs, start=1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % k + o + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objs) + 1)
    out += b"".join(b"%010d 00000 n \n" % off for off in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objs) + 1, xref)
    pdf_path.write_bytes(bytes(out))


# Fixed creation timestamp of the published vector figures (reproducible output). Matplotlib writes
# the UTC offset of timezone-aware dates with the wrong sign, so the PDF date string is set directly.
PUBLISHED_DATE = {"runtime_plot": "D:20261013154332+02'00'",
                  "legend_runtime_plot": "D:20261013154333+02'00'"}
PREVIEW_DPI = 200                                # PNG previews used by the README


def _set_pdf_creation_date(pdf: Path, date: str):
    data = pdf.read_bytes()
    start = data.index(b"/CreationDate (") + len(b"/CreationDate (")
    end = data.index(b")", start)
    if end - start != len(date):
        raise ValueError("CreationDate length mismatch: the xref offsets would change")
    pdf.write_bytes(data[:start] + date.encode() + data[end:])


def save(fig, out: Path, name: str):
    """Writes <name>.pdf and the README preview <name>.png (the PDF rendered at PREVIEW_DPI).
    Raster figures: the PDF wraps the PNG rendering at RASTER_DPI; vector figures: vector PDF."""
    pdf = out / f"{name}.pdf"
    if name in PAGE_DENSITY:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=RASTER_DPI)
        buf.seek(0)
        raster_pdf(buf, pdf, PAGE_DENSITY[name])
    else:
        fig.savefig(pdf, metadata={"CreationDate": datetime(2000, 1, 1, tzinfo=timezone(timedelta(hours=2)))})
        if name in PUBLISHED_DATE:
            _set_pdf_creation_date(pdf, PUBLISHED_DATE[name])
    plt.close(fig)
    pymupdf.open(pdf)[0].get_pixmap(dpi=PREVIEW_DPI, alpha=False).save(out / f"{name}.png")
    print(f"saved {pdf} / .png")


def style_grid(ax, n_groups=None):
    """Common grid of all the figures: horizontal major and minor lines; vertical lines at
    the x ticks (line plots) or between the groups of bars (bar charts, `n_groups`)."""
    ax.minorticks_on()
    ax.tick_params(axis="x", which="minor", bottom=False)
    ax.yaxis.grid(True, which="major", **GRID_MAJOR)
    ax.yaxis.grid(True, which="minor", **GRID_MINOR)
    if n_groups is None:
        ax.xaxis.grid(True, which="major", **GRID_MAJOR)
    else:
        for xv in np.arange(n_groups - 1) + 0.5:
            ax.axvline(xv, zorder=0, **GRID_MAJOR)
    ax.set_axisbelow(True)


def label_facets(axes, col_titles, row_titles, bold=False, row_fontsize=15):
    fw = "bold" if bold else "normal"
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontweight=fw)
    for i, t in enumerate(row_titles):
        ax = axes[i, -1]
        ax.annotate(t, xy=(1.03, 0.5), xycoords="axes fraction", rotation=270,
                    ha="left", va="center", fontsize=row_fontsize, fontweight=fw)


def lighten(color, f=0.55):
    """Mix a colour with white (used for the hatch lines, so that they are not too dark)."""
    c = np.array(matplotlib.colors.to_rgb(color))
    return tuple(c + (1 - c) * f)


def hatch_color(color):
    """Hatch lines visible but soft: darker on light bars, lighter on dark bars."""
    c = np.array(matplotlib.colors.to_rgb(color))
    if 0.299 * c[0] + 0.587 * c[1] + 0.114 * c[2] > 0.7:
        return tuple(c * 0.72)
    return lighten(color, 0.5)


def star(ax, xpos, ypos, ms=13):
    """Marker on the best bar."""
    ax.plot(xpos, ypos, marker="*", ms=ms, color=STAR_FACE, markeredgecolor=STAR_EDGE,
            markeredgewidth=0.9, zorder=5, clip_on=False)


def draw_bar(ax, x, h, width, color, enc):
    """Bar with an optional light hatch and a thin black outline."""
    ax.bar(x, h, width=width, color=color, edgecolor=hatch_color(color), linewidth=0,
           hatch=HATCH[enc])
    ax.bar(x, h, width=width, fill=False, edgecolor="black", linewidth=0.5)


def best_per_dataset(full, keys):
    """Best F1 of every dataset over all the grid parameters not in `keys`."""
    return full.groupby(["dataset"] + keys, as_index=False).best_f1.max()


def legend_figure(entries, out, name, figsize, ncol, fontsize, **kw):
    """Legend exported as a separate figure (frame and shadow), shared by the panels."""
    from matplotlib.legend_handler import HandlerTuple
    fig = plt.figure(figsize=figsize)
    fig.gca().set_axis_off()
    fig.legend([h for h, _ in entries], [l for _, l in entries], loc="center", ncol=ncol,
               frameon=True, shadow=True, fontsize=fontsize,
               handler_map={tuple: HandlerTuple(ndivide=1)}, **kw)
    save(fig, out, name)
