"""Star vs. hierarchical topology (Fig. 5 of the paper).

Generates plots/binary_topo_backbone, plots/multiclass_topo_backbone, and plots/legend_topo.

    python scripts/plot_star_vs_hierarchical.py [--out plots]
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import (ALGS, ENCS, HATCH, MODELS, MODEL_LABEL, N_VALUES, PAIR_LABEL, TOPOLOGIES, TOPO_COLORS,
                    best_per_dataset, draw_bar, label_facets, legend_figure, load_results, output_dir,
                    save, star, style_grid)


def topo_backbone_bars(full, task, out, name):
    """Star vs hierarchical: two adjacent bars per backbone (STAR, HIERARCHICAL),
    rows: algorithm x aggregation mode (CKKS hatched), cols: clients. Best F1 per dataset
    (maximum over the remaining grid parameters), averaged over the datasets of the task."""
    d = full[full.task == task]
    keys = ["topology", "enc", "aggregation_algorithm", "num_clients", "model_name"]
    mean = best_per_dataset(d, keys).groupby(keys).best_f1.mean()
    rows = [(alg, enc) for alg in ALGS for enc in ENCS]
    lo = np.floor((mean.min() - 0.02) / 0.05) * 0.05
    hi = min(1.0, np.ceil((mean.max() + 0.025) / 0.05) * 0.05)
    fig, axes = plt.subplots(len(rows), len(N_VALUES), figsize=(14, 2.6 * len(rows)),
                             sharex=True, sharey=True, gridspec_kw={"wspace": 0.05, "hspace": 0.16})
    x = np.arange(len(MODELS))
    w = 0.36
    for i, (alg, enc) in enumerate(rows):
        for j, n in enumerate(N_VALUES):
            ax = axes[i, j]
            style_grid(ax, n_groups=len(MODELS))
            best = (-1, None)
            for k, topo in enumerate(TOPOLOGIES):
                vals = mean.loc[(topo, enc, alg, n)].reindex(MODELS).values
                xs = x + (k - 0.5) * w
                for xi, h in zip(xs, vals):
                    draw_bar(ax, xi, h, w, TOPO_COLORS[topo], enc)
                kk = int(np.nanargmax(vals))
                if vals[kk] > best[0]:
                    best = (vals[kk], xs[kk])
            star(ax, best[1], best[0] + 0.006, ms=13)
            ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
            ax.set_xticks(x, [MODEL_LABEL[m] for m in MODELS], rotation=40, ha="right")
            if j == 0:
                ax.set_ylabel("Best F1-score")
    label_facets(axes, [f"Clients: {n}" for n in N_VALUES],
                 [f"{alg} - {PAIR_LABEL[enc]}" for alg, enc in rows], row_fontsize=12)
    save(fig, out, name)


def legend_topo(out, name):
    """Legend of the topology figure: STAR / HIERARCHICAL colours, PLAINTEXT / CKKS hatch."""
    entries = []
    for topo in TOPOLOGIES:
        entries.append(((Patch(facecolor=TOPO_COLORS[topo], edgecolor="black", linewidth=0.5),),
                        PAIR_LABEL[topo]))
    for enc in ENCS:
        fill = Patch(facecolor="white", edgecolor="0.55", linewidth=0, hatch=HATCH[enc])
        outline = Patch(fill=False, edgecolor="black", linewidth=0.5)
        entries.append(((fill, outline), PAIR_LABEL[enc]))
    legend_figure(entries, out, name, figsize=(9, 0.6), ncol=4, fontsize=16,
                  handlelength=2.2, handleheight=1.1)


def main():
    out = output_dir()
    full = load_results()
    for task in ("binary", "multiclass"):
        topo_backbone_bars(full, task, out, f"{task}_topo_backbone")
    legend_topo(out, "legend_topo")


if __name__ == "__main__":
    main()
