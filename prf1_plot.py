import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def plot_prf_boxplots(prf_csv, out_dir=None):
    """
    Read per_phenotype_prf.csv and plot boxplots for Precision, Recall, F1.
    """
    df = pd.read_csv(prf_csv)

    # Melt to long format for easier plotting
    long_df = df.melt(
        id_vars=["Gene Set Name"],
        value_vars=["Precision", "Recall", "F1"],
        var_name="Metric",
        value_name="Score"
    )

    if out_dir is None:
        out_dir = os.path.join(os.path.dirname(prf_csv), "plots")
    os.makedirs(out_dir, exist_ok=True)

    # Boxplot
    plt.figure(figsize=(6, 4))
    sns.boxplot(x="Metric", y="Score", data=long_df)
    plt.ylim(0, 1)
    plt.title("Per-phenotype Precision / Recall / F1")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "prf_boxplot.png"), dpi=300)
    plt.close()

    # Violin plot
    plt.figure(figsize=(6, 4))
    sns.violinplot(x="Metric", y="Score", data=long_df, cut=0)
    plt.ylim(0, 1)
    plt.title("Per-phenotype Precision / Recall / F1")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "prf_violin.png"), dpi=300)
    plt.close()

    print(f"Saved plots to {out_dir}")

if __name__ == "__main__":
    prf_csv = "out/genesets/qwen3:32b/evaluation/per_phenotype_prf.csv"
    plot_prf_boxplots(prf_csv)
