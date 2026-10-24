"""Generates every result figure of the paper from the four result CSVs.

    python scripts/make_all_figures.py [--out plots]
"""
import plot_enc_vs_plain
import plot_runtime
import plot_star_vs_hierarchical

if __name__ == "__main__":
    plot_enc_vs_plain.main()
    plot_star_vs_hierarchical.main()
    plot_runtime.main()
