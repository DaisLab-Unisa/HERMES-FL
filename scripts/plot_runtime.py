"""Scalability and computational overhead (Fig. 6 of the paper).

Generates plots/runtime_plot and plots/legend_runtime_plot.

    python scripts/plot_runtime.py [--out plots]
"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from common import (BAR_COLORS, CFG, MODELS, MODEL_LABEL, PAIR_LABEL, PARAMS, SETTINGS, TOPOLOGIES,
                    load_results, output_dir, save, style_grid)

PATIENCE = 5                                   # early stopping patience (rounds)
SLOTS = 4096                                   # CKKS slots per ciphertext (n = 8192)
CT_BYTES = 2 * 8192 * (60 + 40 + 40) / 8       # serialized ciphertext, data level (~280 KB)
FLOAT_BYTES = 4                                # float32 plaintext parameter
ENC_LEGEND = {"plain": "PLAINTEXT", "ckks": "CKKS"}


def head_tensors(model, layers, img, n_classes=2):
    """Sizes of the trainable tensors exchanged at each round (weights and biases)."""
    if model == "ConvNet":
        side = (img - 7) // 3 + 1
        dims = [(4 * side * side, 32), (32, 16), (16, n_classes)]
        sizes = [4 * 1 * 7 * 7, 4]
    else:
        d = {"ResNet18": 512, "ResNet34": 512, "ResNet50": 2048, "ResNet101": 2048,
             "GoogLeNet": 1024, "AlexNet": 9216}[model]
        widths = [256] if layers == 2 else [256, 128, 64]
        chain = [d] + widths + [n_classes]
        dims = list(zip(chain[:-1], chain[1:]))
        sizes = []
    for a, b in dims:
        sizes += [a * b, b]
    return sizes


def update_size(model):
    """(plaintext bytes, ciphertexts, CKKS bytes) of one model update, averaged over the
    head depths (pre-trained backbones) or the input resolutions (ConvNet)."""
    variants = [(2, 128), (2, 224)] if model == "ConvNet" else [(2, 224), (4, 224)]
    out = []
    for layers, img in variants:
        t = head_tensors(model, layers, img)
        n_ct = sum(int(np.ceil(x / SLOTS)) for x in t)
        out.append((sum(t) * FLOAT_BYTES, n_ct, n_ct * CT_BYTES))
    return tuple(np.mean(out, axis=0))


def messages_per_round(topology, n):
    """Model transfers per global round: star = N uploads + N broadcasts; hierarchical
    (R_e = 2) = 2N client uploads + N intermediate edge models + N global models to the
    clients, plus N/2 edge-to-server uploads and N/2 server-to-edge broadcasts."""
    return 2 * n if topology == "star" else 4 * n + n


def scalability_frame(full):
    """Executed rounds (best round + early-stopping patience, capped at T), time per round,
    and CKKS overhead per round (CKKS minus plaintext run time of the same configuration)."""
    d = full.copy()
    d["rounds_exec"] = np.minimum(d.global_epoch, d.best_round + PATIENCE)
    d["time_round_min"] = d.total_duration_sec / d.rounds_exec / 60
    pv = d.pivot_table(index=["topology"] + CFG, columns="enc",
                       values=["total_duration_sec", "rounds_exec"]).dropna()
    over = ((pv[("total_duration_sec", "ckks")] - pv[("total_duration_sec", "plain")])
            / pv[("rounds_exec", "ckks")]).rename("crypto_round_sec").reset_index()
    return d, over


def scalability_plot(full, out, name):
    """Four panels (log scale), backbones ordered by exchanged parameters:
    (a) time per round, (b) CKKS overhead per round, (c) traffic per round, (d) total time."""
    order = sorted(MODELS, key=lambda m: (PARAMS[m], MODELS.index(m)))
    d, over = scalability_frame(full)
    # same blue/red as the bar charts: light = PLAINTEXT, dark = CKKS; star / triangle markers
    style = {(t, e): (BAR_COLORS[(t, "FedAvg" if e == "plain" else "FedProx")],
                      "*" if t == "star" else "^") for t, e in SETTINGS}
    labels = [MODEL_LABEL[m] for m in order]
    x = np.arange(len(order))
    fig, axes = plt.subplots(2, 2, figsize=(16, 9.6), sharex=True,
                             gridspec_kw={"hspace": 0.12, "wspace": 0.2})

    def line(ax, vals, c, mk, label):
        ax.plot(x, vals, color=c, marker=mk, lw=3, ms=17 if mk == "*" else 12, mew=1.0,
                markeredgecolor=tuple(np.array(matplotlib.colors.to_rgb(c)) * 0.6),
                label=label)

    def finish(ax, ylabel, title):
        ax.set_yscale("log")
        lo, hi = ax.get_ylim()
        ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200,
                             500, 1000, 2000, 5000) if lo <= t <= hi]
        ax.set_yticks(ticks, [f"{t:g}" for t in ticks])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        style_grid(ax)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_xticks(x, labels, rotation=30, ha="right")
        ax.tick_params(axis="x", which="minor", bottom=False)

    # (a) time per round
    ax = axes[0, 0]
    med = d.groupby(["topology", "enc", "model_name"]).time_round_min.median()
    for topo, enc in SETTINGS:
        c, mk = style[(topo, enc)]
        line(ax, med.loc[(topo, enc)].reindex(order).values, c, mk,
             f"{PAIR_LABEL[topo]} - {ENC_LEGEND[enc]}")
    ax.set_ylim(med.min() * 0.7, med.max() * 2)
    finish(ax, "Time per round (minutes)", "(a) Time per global round")

    # (b) CKKS overhead per round (encryption, decryption, serialization, aggregation)
    ax = axes[0, 1]
    ov = over.groupby(["topology", "model_name"]).crypto_round_sec.median().clip(lower=0.5)
    for topo in TOPOLOGIES:
        c, mk = style[(topo, "ckks")]                      # CKKS only: dark shade
        line(ax, ov.loc[topo].reindex(order).values, c, mk, PAIR_LABEL[topo])
    ax.set_ylim(ov.min() * 0.7, ov.max() * 2)
    finish(ax, "CKKS overhead per round (seconds)", "(b) Encryption/decryption overhead")

    # (c) communication per round, N = 8
    ax = axes[1, 0]
    n = 8
    for topo, enc in SETTINGS:
        c, mk = style[(topo, enc)]
        size = [update_size(m)[0 if enc == "plain" else 2] for m in order]
        mb = np.array(size) * messages_per_round(topo, n) / 2 ** 20
        line(ax, mb, c, mk, f"{PAIR_LABEL[topo]} - {ENC_LEGEND[enc]}")
    finish(ax, "Traffic per round (MB, $N=8$)", "(c) Communication cost")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi * 2)
    finish(ax, "Traffic per round (MB, $N=8$)", "(c) Communication cost")

    # (d) total run time
    ax = axes[1, 1]
    tot = full.groupby(["topology", "enc", "model_name"]).total_duration_sec.median() / 60
    for topo, enc in SETTINGS:
        c, mk = style[(topo, enc)]
        line(ax, tot.loc[(topo, enc)].reindex(order).values, c, mk,
             f"{PAIR_LABEL[topo]} - {ENC_LEGEND[enc]}")
    ax.set_ylim(tot.min() * 0.7, tot.max() * 2)
    finish(ax, "Run time (minutes)", "(d) Total run time")
    handles, lbls = axes[0, 0].get_legend_handles_labels()
    save(fig, out, name)
    # one legend shared by all panels, saved as a separate file (like the bar charts)
    lfig = plt.figure(figsize=(14, 0.6))
    lfig.gca().set_axis_off()
    lfig.legend(handles, lbls, loc="center", ncol=len(lbls), frameon=True, shadow=True,
                fontsize=15, handlelength=2.6, columnspacing=1.6)
    save(lfig, out, f"legend_{name}")


def main():
    out = output_dir()
    scalability_plot(load_results(), out, "runtime_plot")


if __name__ == "__main__":
    main()
