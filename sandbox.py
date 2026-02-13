# # Determine the phenotypes for which has been completed for the checker and generation
# def determine_completed_phenotypes(generation_paths, checker_paths):
import json
import random
import os


def create_abstract_ablation_phenotypes(pheno_list, full_phenotypes_file, output_file, num_samples=130):
    # full_phenotypes_file is a JSON file with a list of full phenotype dicts with name, definition, 
    # pheno_list if a text file of phenotype names to extract the substet from
    # output_file is the path to save the new JSON file with the subset of phenotypes with only name and empty definition
    # sample scheme in full_phenotypes_file:
    #     {
    #     "id": "HP:0004904",
    #     "name": "Maturity-onset diabetes of the young",
    #     "description": "The term Maturity-onset diabetes of the young (MODY) was initially used for patients diagnosed with fasting hyperglycemia that could be treated without insulin for more than two years, where the initial diagnosis was made at a young age (under 25 years). Thus, MODY combines characteristics of type 1 diabetes (young age at diagnosis) and type 2 diabetes (less insulin dependence than type 1 diabetes). The term MODY is now most often used to refer to a group of monogenic diseases with these characteristics. Here, the term is used to describe hyperglycemia diagnosed at a young age with no or minor insulin dependency, no evidence of insulin resistance, and lack of evidence of autoimmune destruction of the beta cells.",
    #     "synonyms": [
    #       "Maturity onset diabetes of the young",
    #       "MODY"
    #     ]
    #   }
    # load pheno_list
    with open(pheno_list, "r") as f:
        pheno_list = [line.strip() for line in f if line.strip()]

    # randomly select num_samples phenotypes from pheno_list
    selected_phenotypes = random.sample(pheno_list, min(num_samples, len(pheno_list)))
    # load full phenotypes
    with open(full_phenotypes_file, "r") as f:
        full_phenotypes = json.load(f)
    # create a dict of phenotype name to full phenotype
    full_phenotype_dict = {pheno["name"]: pheno for pheno in full_phenotypes}
    # create new list of phenotypes with all the same schema as the full phenotypes but only for the selected_phenotypes
    abstract_phenotypes = []
    for pheno_name in selected_phenotypes:
        if pheno_name in full_phenotype_dict:
            abstract_phenotypes.append({
                "id": full_phenotype_dict[pheno_name]["id"],
                "name": pheno_name,
                "description": "",
                "synonyms": full_phenotype_dict[pheno_name].get("synonyms", [])
            })
    # save to output_file
    with open(output_file, "w") as f:
        json.dump(abstract_phenotypes, f, indent=2)

create_abstract_ablation_phenotypes("out/processed_phenotypes_qwen3:32b.txt", "out/in_db_and_p2g_details.json", "out/aa/sample_subset.json", num_samples=130)  

def update_processed_phenotypes(processed_file, genes_dir, checked_genes_dir):
    """
    Compare processed phenotypes (in a .txt file) with actual phenotypes
    having gene extraction JSON files in a directory, and update the file.

    Args:
        processed_file (str): Path to text file containing processed phenotype names.
        genes_dir (str): Directory containing gene extraction JSON files.
        checked_genes_dir (str): Directory containing a directory for each phenotype, each json file represents one gene.
    """
    # Load processed phenotypes from file
    processed_phenotypes = set()
    if os.path.exists(processed_file):
        with open(processed_file, "r") as f:
            processed_phenotypes = set(line.strip() for line in f if line.strip())

    # List all JSON files in the genes directory
    all_files = os.listdir(genes_dir)
    json_files = [f for f in all_files if f.endswith(".json")]

    checked_genes_dir_files = os.listdir(checked_genes_dir)
    #  Filter json_files to only include those that have a corresponding directory in checked_genes_dir
    json_files = [f for f in json_files if os.path.splitext(f)[0] in checked_genes_dir_files]


    # Extract phenotype names from JSON filenames
    phenotype_names_in_dir = set(os.path.splitext(f)[0] for f in json_files)

    # Determine unprocessed phenotypes
    unprocessed_phenotypes = phenotype_names_in_dir - processed_phenotypes

    # Append unprocessed phenotypes to the processed file
    if unprocessed_phenotypes:
        with open(processed_file, "a") as f:
            for phenotype in sorted(unprocessed_phenotypes):
                f.write(f"{phenotype}\n")
    # delete from processed_phenotypes any phenotype that does not have a corresponding json file in genes_dir
    processed_phenotypes = processed_phenotypes.intersection(phenotype_names_in_dir)
    # write the updated processed_phenotypes back to the processed_file
    with open(processed_file, "w") as f:
        for phenotype in sorted(processed_phenotypes):
            f.write(f"{phenotype}\n")

# # deepseek-r1:8b
# update_processed_phenotypes("out/processed_phenotypes_deepseek-r1:8b.txt", "out/phenotype_generations/deepseek-r1:8b", "out/phenotype_checks/deepseek-r1:8b")
# # llama3.1:8b
# update_processed_phenotypes("out/processed_phenotypes_llama3.1:8b.txt", "out/phenotype_generations/llama3.1:8b", "out/phenotype_checks/llama3.1:8b")
# #qwen3:32b
# update_processed_phenotypes("out/processed_phenotypes_qwen3:32b.txt", "out/phenotype_generations/qwen3:32b", "out/phenotype_checks/qwen3:32b")

