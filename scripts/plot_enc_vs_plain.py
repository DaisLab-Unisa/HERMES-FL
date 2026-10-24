"""Encrypted vs. plaintext aggregation (Fig. 4 of the paper).

Generates plots/binary_enc_domain, plots/multiclass_enc_domain, and plots/legend_bars.

    python scripts/plot_enc_vs_plain.py [--out plots]
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from common import (ALGS, BAR_COLORS, DOMAINS, ENCS, HATCH, MODELS, MODEL_LABEL, PAIR_LABEL, TOPOLOGIES,
                    best_per_dataset, draw_bar, label_facets, legend_figure, load_results, output_dir,
                    save, star, style_grid)


def enc_domain_model_bars(full, task, out, name):
    """Plaintext vs CKKS per domain and backbone: rows = domains, cols = topology,
    x = backbones, four bars per backbone (FedAvg/FedProx x plaintext/CKKS).
    Best F1 per dataset and backbone (maximum over rounds, local epochs, resolution,
    head depth, and clients), averaged over the datasets of the domain."""
    d = full[full.task == task]
    keys = ["domain", "topology", "enc", "aggregation_algorithm", "model_name"]
    mean = best_per_dataset(d, keys).groupby(keys).best_f1.mean()
    doms = DOMAINS[task]
    lo = np.floor((mean.min() - 0.02) / 0.05) * 0.05
    hi = 1.03
    fig, axes = plt.subplots(len(doms), len(TOPOLOGIES), figsize=(14, 2.75 * len(doms)),
                             sharex=True, sharey=True, gridspec_kw={"wspace": 0.05, "hspace": 0.18})
    x = np.arange(len(MODELS))
    w = 0.19
    offs = [-1.6 * w, -0.6 * w, 0.6 * w, 1.6 * w]
    for i, dom in enumerate(doms):
        for j, topo in enumerate(TOPOLOGIES):
            ax = axes[i, j]
            style_grid(ax, n_groups=len(MODELS))
            ax.tick_params(axis="x", which="minor", bottom=False)
            heights = np.zeros((4, len(MODELS)))          # bar heights: (algorithm x mode, backbone)
            k = 0
            for alg in ALGS:
                for enc in ENCS:
                    vals = mean.loc[(dom, topo, enc, alg)].reindex(MODELS).values
                    heights[k] = vals
                    xs = x + offs[k]
                    for xi, h in zip(xs, vals):
                        draw_bar(ax, xi, h, w, BAR_COLORS[(topo, alg)], enc)
                    k += 1
            # one star per backbone, on the winner of its group of four bars
            win = np.nanargmax(heights, axis=0)
            for mi in range(len(MODELS)):
                star(ax, x[mi] + offs[win[mi]], heights[win[mi], mi] + 0.008, ms=11)
            ax.set_ylim(lo, hi)
            ax.yaxis.set_major_locator(matplotlib.ticker.FixedLocator(np.arange(lo, 1.0001, 0.05)))
            ax.set_xticks(x, [MODEL_LABEL[m] for m in MODELS], rotation=40, ha="right")
            if j == 0:
                ax.set_ylabel("Best F1-score")
    label_facets(axes, [PAIR_LABEL[t] for t in TOPOLOGIES], [f"Domain: {dm}" for dm in doms])
    save(fig, out, name)


def legend_bars(out, name):
    """Shared legend of the bar charts: colour = topology, shade = algorithm, hatch = CKKS."""
    entries = []
    for topo in TOPOLOGIES:
        for alg in ALGS:
            c = BAR_COLORS[(topo, alg)]
            entries.append(((Patch(facecolor=c, edgecolor="black", linewidth=0.5),),
                            f"{PAIR_LABEL[topo]} {alg}"))
    for enc in ENCS:
        fill = Patch(facecolor="white", edgecolor="0.55", linewidth=0, hatch=HATCH[enc])
        outline = Patch(fill=False, edgecolor="black", linewidth=0.5)
        entries.append(((fill, outline), PAIR_LABEL[enc]))
    legend_figure(entries, out, name, figsize=(14, 0.6), ncol=len(entries), fontsize=14,
                  handlelength=2.0, handleheight=1.1, columnspacing=1.2)


def main():
    out = output_dir()
    full = load_results()
    for task in ("binary", "multiclass"):
        enc_domain_model_bars(full, task, out, f"{task}_enc_domain")
    legend_bars(out, "legend_bars")


if __name__ == "__main__":
    main()
