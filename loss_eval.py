import pandas as pd

# --- config ---
INPUT_CSV  = "out/genesets/evaluation/gene_set_comparison.csv"           # change to your filename
OUTPUT_CSV = "gene_loss_comparison_with_stats.csv"

# thresholds for “low / medium / high” loss in percent
LOW_MAX  = 20    # < 20% loss
HIGH_MIN = 60    # > 60% loss

def main():
    df = pd.read_csv(INPUT_CSV)

    # column mapping based on your headers
    # Gene Set Name
    # Common Genes
    # Newly Added Genes
    # Lost Genes
    # # Common
    # # New
    # # Lost
    # # Original

    # % loss relative to original set
    df["percent_loss"] = (df["# Lost"] / df["# Original"]) * 100

    # overall mean loss
    mean_loss = df["percent_loss"].mean()

    # above / below average loss
    df["loss_vs_mean"] = df["percent_loss"].apply(
        lambda x: "below_mean" if x < mean_loss else "above_mean"
    )

    # no new genes added or removed
    df["no_change"] = (df["# Lost"] == 0) & (df["# New"] == 0)

    # bins: low / medium / high loss
    def loss_bin(pct):
        if pct < LOW_MAX:
            return "low_loss"
        elif pct > HIGH_MIN:
            return "high_loss"
        else:
            return "medium_loss"

    df["loss_bin"] = df["percent_loss"].apply(loss_bin)

    # --- summary numbers ---
    n_above_mean = (df["loss_vs_mean"] == "above_mean").sum()
    n_below_mean = (df["loss_vs_mean"] == "below_mean").sum()
    n_no_change  = df["no_change"].sum()

    n_low_loss    = (df["loss_bin"] == "low_loss").sum()
    n_medium_loss = (df["loss_bin"] == "medium_loss").sum()
    n_high_loss   = (df["loss_bin"] == "high_loss").sum()

    print(f"Mean % loss: {mean_loss:.2f}")
    print(f"Gene sets above mean loss:  {n_above_mean}")
    print(f"Gene sets below mean loss:  {n_below_mean}")
    print(f"Gene sets with no change (no new or lost genes): {n_no_change}")
    print()
    print(f"Low-loss   (<{LOW_MAX}%): {n_low_loss}")
    print(f"Medium-loss ({LOW_MAX}–{HIGH_MIN}%): {n_medium_loss}")
    print(f"High-loss  (>{HIGH_MIN}%): {n_high_loss}")

    # save augmented CSV
    df.to_csv(OUTPUT_CSV, index=False)

if __name__ == "__main__":
    main()
