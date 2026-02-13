import os
import json
import argparse

from utils import phenotype_json_reader, read_gmt, read_phenotype_to_gene_sets
from rag_pipeline_gene_maker_aa import create_control_flow as create_maker_flow
from rag_pipeline_gene_checker_aa import create_control_flow as create_checker_flow

GMT_PATH = "out/phenotype_to_gene_sets.txt"


def get_llm_file_paths(llm_name, num_abstracts):
    """Generate LLM-specific file paths and ensure they exist."""
    os.makedirs("out", exist_ok=True)
    
    paths = {
        "processed_file": f"out/aa/{num_abstracts}/processed_phenotypes_{llm_name}.txt",
        "processed_genes_file": f"out/aa/{num_abstracts}/processed_genes_{llm_name}.json",
        "processed_sets_file": f"out/aa/{num_abstracts}/processed_gene_sets_{llm_name}.txt"
    }
    
    # Create files if they don't exist
    for file_path in paths.values():
        if not os.path.exists(file_path):
            with open(file_path, "w") as f:
                if file_path.endswith(".json"):
                    f.write("{}")
                else:
                    f.write("")
    
    return paths


def load_processed(processed_file):
    """Load phenotype names that have completed both pipelines."""
    if not os.path.exists(processed_file):
        return set()
    with open(processed_file, "r") as f:
        return set(line.strip() for line in f.readlines())


def mark_processed(phenotype_name, processed_file):
    """Append a phenotype to the processed file."""
    with open(processed_file, "a") as f:
        f.write(f"{phenotype_name}\n")

def load_processed_genes(processed_genes_file):
    """Load processed genes per gene set for the checker pipeline."""
    if os.path.exists(processed_genes_file):
        with open(processed_genes_file, "r") as f:
            return json.load(f)
    return {}


def save_processed_genes(processed, processed_genes_file):
    """Save processed genes dictionary."""
    with open(processed_genes_file, "w") as f:
        json.dump(processed, f, indent=2)


def mark_gene_processed(gene_set, gene, processed, processed_genes_file):
    """Mark a gene as processed for a given gene set and persist to disk."""
    processed.setdefault(gene_set, [])
    if gene not in processed[gene_set]:
        processed[gene_set].append(gene)
        save_processed_genes(processed, processed_genes_file)


def load_completed_sets(processed_sets_file):
    """Load gene sets that have been completely checked."""
    if not os.path.exists(processed_sets_file):
        return set()
    with open(processed_sets_file, "r") as f:
        return set(line.strip() for line in f if line.strip())


def mark_set_complete(gene_set, processed_sets_file):
    """Append completed gene set to processed_gene_sets file."""
    with open(processed_sets_file, "a") as f:
        f.write(f"{gene_set}\n")


def run_checker_for_phenotype(phenotype, genes, llm_name, file_paths):
    """
    Run the checker pipeline for a single phenotype name and its list of genes.
    Uses intersection logic: we only call this if the phenotype exists in the GMT.
    """
    phenotype_name = phenotype["name"]
    print(f"Running checker pipeline for phenotype: {phenotype_name}")

    processed_genes = load_processed_genes(file_paths["processed_genes_file"])
    completed_sets = load_completed_sets(file_paths["processed_sets_file"])

    if phenotype_name in completed_sets:
        print(f"Gene set for {phenotype_name} already completed. Skipping checker.")
        return

    processed_genes.setdefault(phenotype_name, [])

    graph = create_checker_flow()

    for gene in genes:
        if gene in processed_genes[phenotype_name]:
            print(f"  Skipping {gene} (already processed for {phenotype_name})")
            continue

        phenotype_state = {
            "name": phenotype_name,
            "gene": gene,
            "definition": phenotype.get("definition", ""),
            "synonyms": phenotype.get("synonyms", [])
        }

        print(f"  Checking {gene} for {phenotype_name}")

        try:
            # Run checker graph for this phenotype+gene
            for _ in graph.stream({"phenotype": phenotype_state, "llm_name": llm_name}, stream_mode="values"):
                pass
            mark_gene_processed(phenotype_name, gene, processed_genes, file_paths["processed_genes_file"])
            print(f"  Completed {gene} for {phenotype_name}")
        except Exception as e:
            print(f"  Error processing {gene} in {phenotype_name}: {e}")

    mark_set_complete(phenotype_name, file_paths["processed_sets_file"])
    print(f"Completed checker for gene set: {phenotype_name}")


def main():
    parser = argparse.ArgumentParser(description="Run maker and checker pipelines for phenotypes.")
    parser.add_argument(
        "--input_file",
        type=str,
        default="out/in_db_and_p2g_details.json",
        help="Path to the input JSON file with phenotype details."
    )
    # add llm to use as argument
    parser.add_argument(
        "--llm",
        type=str,
        default="deepseek-r1:8b",
        help="LLM to use for grading abstracts."
    )
    # add num_of_abstracts as argument
    parser.add_argument(
        "--num_of_abstracts",
        type=int,
        default=100,
        help="Number of abstracts to retrieve."
    )
    args = parser.parse_args()

    llm_name = args.llm
    num_abstracts = args.num_of_abstracts
    # Generate LLM-specific file paths
    file_paths = get_llm_file_paths(llm_name, num_abstracts)

    # Load phenotypes
    phenotypes = phenotype_json_reader(args.input_file)

    # Load already fully processed phenotypes (maker + checker)
    processed = load_processed(file_paths["processed_file"])

    # Load GMT gene sets once
    if not os.path.exists(GMT_PATH):
        raise FileNotFoundError(f"GMT file not found at {GMT_PATH}")
    gene_sets = read_phenotype_to_gene_sets(GMT_PATH)

    # Intersection logic:
    #   Checker only runs for phenotypes whose name appears in gene_sets. 
    to_process = [p for p in phenotypes if p["name"] not in processed]

    print(f"Total phenotypes in file: {len(phenotypes)}")
    print(f"Already fully processed (maker + checker): {len(processed)}")
    print(f"Remaining to process: {len(to_process)}")
    print(f"Using LLM-specific files with suffix: {llm_name}")

    if not to_process:
        print("All phenotypes already processed. Nothing to do.")
        return

    for phenotype in to_process:
        name = phenotype["name"]
        print(f"\nProcessing phenotype: {name}")

        # Maker pipeline
        try:
            maker_graph = create_maker_flow()
            inputs = {"phenotype": phenotype, "llm_name": llm_name, "num_of_abstracts": num_abstracts}

            for _ in maker_graph.stream(inputs, stream_mode="values"):
                pass

            print(f"Maker pipeline completed for {name}")
        except Exception as e:
            print(f"Error in maker pipeline for {name}: {e}")
            # Do not mark as processed; continue to next phenotype
            continue

        # Checker pipeline only if this phenotype appears in GMT
        if name in gene_sets:
            genes = gene_sets[name]
            run_checker_for_phenotype(phenotype, genes, llm_name, file_paths)
        else:
            print(f"No matching gene set in gene set database for phenotype '{name}'. Skipping checker for this phenotype.")

        # Only now mark phenotype as fully processed
        mark_processed(name, file_paths["processed_file"])
        print(f"Finished phenotype: {name}")

    print(f"\nAll phenotypes processed. Progress saved.")


if __name__ == "__main__":
    main()
