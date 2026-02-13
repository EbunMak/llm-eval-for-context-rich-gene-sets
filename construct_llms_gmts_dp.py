import os
import argparse


# ----------------------------
# GMT utilities
# ----------------------------

def load_gmt(filepath):
    """
    Load GMT into:
      { phenotype_name : set(genes) }
    """
    gene_sets = {}
    with open(filepath) as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 2:
                gene_sets[parts[0]] = set(parts[2:])
    return gene_sets


def extract_phenotypes_from_gmt(gmt_path):
    """
    Returns a set of phenotype names from a GMT
    """
    phenotypes = set()
    with open(gmt_path) as f:
        for line in f:
            parts = line.strip().split("\t")
            if parts:
                phenotypes.add(parts[0])
    return phenotypes


# ----------------------------
# Direct-prompting consensus
# ----------------------------

def make_direct_consensus_gmt_filtered_by_rag(
    llama_gmt,
    deepseek_gmt,
    rag_consensus_gmt,
    out_gmt="direct_prompting_consensus_filtered.gmt",
):
    llama = load_gmt(llama_gmt)
    deepseek = load_gmt(deepseek_gmt)

    # Trusted phenotype space from RAG
    trusted_phenotypes = load_gmt(rag_consensus_gmt) #extract_phenotypes_from_gmt(rag_consensus_gmt)

    valid_phenotypes = (
        set(llama.keys())
        & set(deepseek.keys())
        & set(trusted_phenotypes.keys())
    )

    os.makedirs(os.path.dirname(out_gmt), exist_ok=True)

    written = 0

    with open(out_gmt, "w") as out:
        for pheno in sorted(valid_phenotypes):
            g_l = llama.get(pheno, set())
            g_d = deepseek.get(pheno, set())

            # Gene-level consensus (intersection)
            consensus_genes = sorted(g_l & g_d)

            if consensus_genes:
                out.write(
                    pheno
                    + "\tdirect_prompting_consensus\t"
                    + "\t".join(consensus_genes)
                    + "\n"
                )
                written += 1

    print(f"Direct-prompting consensus GMT written to: {out_gmt}")
    print(f"Phenotypes written: {written}")
    print(f"Phenotypes considered: {len(valid_phenotypes)}")


# ----------------------------
# Optional: simple phenotype filtering (not used in main)
# ----------------------------

def filter_direct_gmt_by_rag(
    direct_gmt,
    rag_gmt,
    out_gmt,
    tag="direct_prompting_filtered"
):
    """
    Keep only phenotypes present in the RAG GMT
    """
    rag_phenotypes = extract_phenotypes_from_gmt(rag_gmt)
    direct_sets = load_gmt(direct_gmt)

    os.makedirs(os.path.dirname(out_gmt), exist_ok=True)

    kept = 0

    with open(out_gmt, "w") as out:
        for phenotype, genes in sorted(direct_sets.items()):
            if phenotype in rag_phenotypes and genes:
                out.write(
                    phenotype
                    + f"\t{tag}\t"
                    + "\t".join(sorted(genes))
                    + "\n"
                )
                kept += 1

    print(f"Filtered GMT written to: {out_gmt}")
    print(f"Phenotypes kept: {kept} / {len(direct_sets)}")


# ----------------------------
# CLI
# ----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build direct-prompting consensus GMT filtered by RAG phenotypes"
    )

    parser.add_argument(
        "--llama_gmt",
        required=True,
        help="Direct-prompting LLaMA GMT (entrez or symbols)"
    )

    parser.add_argument(
        "--deepseek_gmt",
        required=True,
        help="Direct-prompting DeepSeek GMT (entrez or symbols)"
    )

    parser.add_argument(
        "--rag_consensus_gmt",
        required=True,
        help="Trusted RAG / consensus GMT used as phenotype whitelist"
    )

    parser.add_argument(
        "--out_gmt",
        default="out/genesets/direct_prompting_consensus_filtered.gmt",
        help="Output consensus GMT"
    )

    args = parser.parse_args()

    make_direct_consensus_gmt_filtered_by_rag(
        llama_gmt=args.llama_gmt,
        deepseek_gmt=args.deepseek_gmt,
        rag_consensus_gmt=args.rag_consensus_gmt,
        out_gmt=args.out_gmt,
    )


if __name__ == "__main__":
    main()