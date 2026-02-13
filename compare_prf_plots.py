import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def add_summary_stats(df):
    """
    Print mean and median per construction and metric.
    """
    metrics = ["Precision", "Recall", "F1"]
    for metric in metrics:
        print("\n=== {} ===".format(metric))
        stats = df.groupby("Construction")[metric].agg(["mean", "median"])
        print(stats.round(3))


def plot_metric_boxplots(df, metric, out_dir):
    """
    Boxplot of the given metric across constructions.
    Different colors per construction, plus mean marker and
    text labels with mean/median.
    """
    os.makedirs(out_dir, exist_ok=True)

    plt.figure(figsize=(7, 4))
    ax = sns.boxplot(
        x="Construction",
        y=metric,
        data=df,
        palette="Set2",       # different colors
        showfliers=False
    )

    # Compute summary stats
    grouped = df.groupby("Construction")[metric]
    means = grouped.mean()
    medians = grouped.median()

    # Overlay mean marker
    x_positions = range(len(means))
    plt.scatter(
        x_positions,
        means.values,
        color="black",
        marker="D",
        s=30,
        zorder=3,
        label="Mean"
    )

    # Add text labels for mean and median
    for i, (label, mean_val) in enumerate(means.items()):
        median_val = medians[label]
        # Slight vertical offsets so text doesn't overlap box/marker
        ax.text(
            i,
            mean_val + 0.02,
            "mean={:.2f}".format(mean_val),
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
            rotation=0,
        )
        ax.text(
            i,
            median_val - 0.04,
            "med={:.2f}".format(median_val),
            ha="center",
            va="top",
            fontsize=8,
            color="dimgray",
            rotation=0,
        )

    plt.ylim(0, 1)
    plt.ylabel(metric)
    plt.xlabel("")
    plt.title("{} per phenotype".format(metric))
    plt.xticks(rotation=30, ha="right")
    plt.legend(loc="lower left", fontsize="small")
    plt.tight_layout()

    out_path = os.path.join(out_dir, "{}_boxplot.png".format(metric.lower()))
    plt.savefig(out_path, dpi=300)
    plt.close()
    print("Saved {} boxplot to {}".format(metric, out_path))


def load_prf_tables(csv_paths, labels):
    """
    Load multiple per-phenotype PRF tables and add a 'Construction' column.
    Each CSV must have columns: 'Gene Set Name', 'Precision', 'Recall', 'F1'.
    """
    all_dfs = []
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        # Basic sanity check
        for col in ["Precision", "Recall", "F1"]:
            if col not in df.columns:
                raise ValueError("{} not found in {}".format(col, path))
        df["Construction"] = label
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple PRF tables via boxplots."
    )
    parser.add_argument(
        "--csvs",
        nargs="+",
        required=True,
        help="List of per_phenotype_prf.csv paths (one per construction).",
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        required=True,
        help="List of labels for each construction (same order as --csvs).",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="prf_comparison_plots",
        help="Output directory for plots.",
    )
    args = parser.parse_args()

    if len(args.csvs) != len(args.labels):
        raise ValueError("Number of --csvs must match number of --labels")

    df = load_prf_tables(args.csvs, args.labels)
    add_summary_stats(df)
    for metric in ["Precision", "Recall", "F1"]:
        plot_metric_boxplots(df, metric, args.out_dir)


if __name__ == "__main__":
    main()
