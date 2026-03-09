import os
import argparse
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm


def add_summary_stats(df):
    metrics = ["Precision", "Recall", "F1"]
    for metric in metrics:
        print("\n=== {} ===".format(metric))
        stats = df.groupby("Configuration")[metric].agg(["mean", "median"])
        print(stats.round(3))


def get_configuration_order(df):
    f1_medians = df.groupby("Configuration")["F1"].median().sort_values(ascending=False)
    return list(f1_medians.index)


def get_palette(config_order):
    base_palette = sns.color_palette("Set2", n_colors=len(config_order))
    return {cfg: color for cfg, color in zip(config_order, base_palette)}


def plot_metric_boxplots(df, metric, out_dir, config_order, palette_map):
    os.makedirs(out_dir, exist_ok=True)

    plt.rcParams['font.family'] = 'Liberation Serif'
    plt.rcParams['font.weight'] = 'bold'

    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(
        x="Configuration",
        y=metric,
        data=df,
        order=config_order,
        palette=[palette_map[cfg] for cfg in config_order],
        showfliers=False,
        notch=True,
        ax=ax,
        linewidth=1.8,
    )

    # Median labels — bigger and bolder
    grouped = df.groupby("Configuration")[metric]
    medians = grouped.median()
    for i, cfg in enumerate(config_order):
        median_val = medians[cfg]
        ax.text(
            i,
            median_val + 0.02,
            f"{median_val:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="#222222",
        )

    ax.set_ylim(0, 1.08)

    # Axis labels
    ax.set_ylabel(metric, fontsize=16, fontweight="bold", labelpad=12)
    ax.set_xlabel("Configuration", fontsize=16, fontweight="bold", labelpad=12)

    # Tick labels
    ax.set_xticks(range(len(config_order)))
    ax.set_xticklabels(
        config_order,
        rotation=35,
        ha="right",
        fontsize=12,
        fontweight="bold",
    )
    ax.tick_params(axis='y', labelsize=12)
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")

    # Grid
    ax.grid(True, axis='y', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_axisbelow(True)

    sns.despine(trim=True)
    plt.tight_layout()

    out_path = os.path.join(out_dir, f"{metric.lower()}_boxplot.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved {metric} boxplot to {out_path}")


def load_prf_tables(csv_paths, labels):
    all_dfs = []
    for path, label in zip(csv_paths, labels):
        df = pd.read_csv(path)
        for col in ["Precision", "Recall", "F1"]:
            if col not in df.columns:
                raise ValueError("{} not found in {}".format(col, path))
        df["Configuration"] = label
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(
        description="Compare multiple PRF tables via boxplots."
    )
    parser.add_argument("--csvs", nargs="+", required=True)
    parser.add_argument("--labels", nargs="+", required=True)
    parser.add_argument("--out_dir", type=str, default="prf_comparison_plots")
    args = parser.parse_args()

    if len(args.csvs) != len(args.labels):
        raise ValueError("Number of --csvs must match number of --labels")

    df = load_prf_tables(args.csvs, args.labels)
    add_summary_stats(df)

    config_order = get_configuration_order(df)
    palette_map = get_palette(config_order)

    for metric in ["Precision", "Recall", "F1"]:
        plot_metric_boxplots(df, metric, args.out_dir, config_order, palette_map)


if __name__ == "__main__":
    main()