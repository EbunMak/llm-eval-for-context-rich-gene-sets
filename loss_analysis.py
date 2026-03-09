# loss_analysis.py
import argparse
import os
import json
import requests
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, linregress

plt.style.use("seaborn-v0_8-whitegrid")

HPO_API = "https://ontology.jax.org/api/hp/terms/{}"  # takes HP%3A0001760

def get_hpo_descendants(hpo_id: str) -> int | None:
    """Return descendantCount for an HPO term ID like 'HP:0001760'."""
    url = HPO_API.format(hpo_id.replace(":", "%3A"))
    try:
        r = requests.get(url, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"Request error for {hpo_id}: {e}")
        return None

    if r.status_code != 200:
        print(f"Failed to fetch descendants for {hpo_id} (status {r.status_code})")
        return None

    data = r.json()
    # descendantCount is an integer
    return data.get("descendantCount", None)

def main(json_path: str, csv_path: str, out_dir: str = "out"):
    os.makedirs(out_dir, exist_ok=True)

    # Load phenotype details from JSON
    with open(json_path, "r") as f:
        phenotype_details = json.load(f)

    # Load gene comparison CSV
    df = pd.read_csv(csv_path)
    df["% Lost"] = 100 * df["# Lost"] / df["# Original"]


    # Expect columns: 'Gene Set Name', '% Lost'
    if "% Lost" not in df.columns:
        raise ValueError("CSV must contain a '% Lost' column.")

    hpo_ids = []
    loss_pct = []

    for _, row in df.iterrows():
        gene_set_name = row["Gene Set Name"]
        lost_percent = row["% Lost"]

        # Match gene set name to phenotype entry in JSON to get HPO ID
        hpo_id = None
        for entry in phenotype_details:
            if entry["name"] == gene_set_name:
                hpo_id = entry["id"]
                break

        if hpo_id is None:
            print(f"No HPO ID found for {gene_set_name}")
            continue

        hpo_ids.append(hpo_id)
        loss_pct.append(lost_percent)

    # Load or build descendant cache
    cache_path = os.path.join(out_dir, "hpo_descendant_counts.csv")
    if os.path.exists(cache_path):
        cache_df = pd.read_csv(cache_path)
    else:
        cache_df = pd.DataFrame(columns=["HPO ID", "Descendant Count"])

    descendant_counts = []
    for hpo_id in hpo_ids:
        if hpo_id in cache_df["HPO ID"].values:
            desc = int(cache_df.loc[cache_df["HPO ID"] == hpo_id,
                                    "Descendant Count"].values[0])
        else:
            desc = get_hpo_descendants(hpo_id)
            cache_df = pd.concat(
                [cache_df,
                 pd.DataFrame({"HPO ID": [hpo_id],
                               "Descendant Count": [desc]})],
                ignore_index=True
            )
        descendant_counts.append(desc)

    # Save updated cache
    cache_df.to_csv(cache_path, index=False)

    # Drop any rows where descendant count is missing
    valid_mask = pd.notnull(descendant_counts)
    x = pd.Series(descendant_counts)[valid_mask].astype(float)
    y = pd.Series(loss_pct)[valid_mask].astype(float)

    # Correlation
    r, p = pearsonr(x, y)
    print(f"Pearson correlation between % loss and # descendants: {r:.3f} (p = {p:.3e})")

    # Scatter + regression line
    plt.figure(figsize=(6, 4.5))
    plt.scatter(x, y, alpha=0.6, edgecolor="k", linewidth=0.5)

    # Regression line
    slope, intercept, _, _, _ = linregress(x, y)
    x_line = pd.Series(sorted(x))
    y_line = slope * x_line + intercept
    plt.plot(x_line, y_line, color="tab:red", linewidth=2, label="Linear fit")

    plt.xlabel("Number of HPO descendants", fontsize=11)
    plt.ylabel("% genes lost", fontsize=11)
    plt.title("Relationship between HPO term generality\n(# descendants) and % gene loss", fontsize=12)
    plt.legend(frameon=False)
    plt.tight_layout()

    fig_path = os.path.join(out_dir, "descendants_vs_loss_pct.png")
    plt.savefig(fig_path, dpi=300)
    plt.close()
    print(f"Saved figure to {fig_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze correlation between HPO descendants and % gene loss."
    )
    parser.add_argument("json_path", help="Path to phenotype details JSON file.")
    parser.add_argument("csv_path", help="Path to gene comparison CSV file.")
    parser.add_argument(
        "--out_dir", default="out", help="Directory to save cache and figures."
    )
    args = parser.parse_args()
    main(args.json_path, args.csv_path, args.out_dir)
