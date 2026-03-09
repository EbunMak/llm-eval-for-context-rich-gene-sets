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
                "description": full_phenotype_dict[pheno_name].get("description", ""),
                "synonyms": full_phenotype_dict[pheno_name].get("synonyms", [])
            })
    # save to output_file
    with open(output_file, "w") as f:
        json.dump(abstract_phenotypes, f, indent=2)

# create_abstract_ablation_phenotypes("out/processed_phenotype_list.txt", "out/in_db_and_p2g_details.json", "out/direct-prompting/sample_subset.json", num_samples=1230)  

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

# read all files in qwen dp folder - and extract the phenotypes names from all files - bot json and .txt (the txt files end with _raw.txt)
# then read the gmt of concensus genes and get the phenotype names from the gmt and compare with the phenotypes names from the qwen dp folder - 
# to determine which phenotypes have been processed for the gmt construction and which have not been processed yet
def determine_completed_phenotypes(generation_dir, gmt_file):
    # read phenotypes from generation_dir
    generated_phenotypes = set()
    for fname in os.listdir(generation_dir):
        if fname.endswith(".json") or fname.endswith("_raw.txt"):
            phenotype_name = os.path.splitext(fname)[0].replace("_raw", "")
            generated_phenotypes.add(phenotype_name)
    print(f"generated phenotypes: {len(generated_phenotypes)}")

    # read phenotypes from gmt_file
    gmt_phenotypes = set()
    with open(gmt_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 0:
                gmt_phenotypes.add(parts[0])
    print(f"gmt phenotypes: {len(gmt_phenotypes)}")

    # determine completed and uncompleted phenotypes
    completed_phenotypes = generated_phenotypes.intersection(gmt_phenotypes)
    uncompleted_phenotypes = gmt_phenotypes - generated_phenotypes

    return completed_phenotypes, uncompleted_phenotypes

# completed_phenotypes, uncompleted_phenotypes = determine_completed_phenotypes("out/direct-prompting/phenotype_generations/qwen3:32b", "out/genesets/consensus_gene_sets.gmt")
# print(f"Completed phenotypes: {len(completed_phenotypes)}")
# print(f"Uncompleted phenotypes: {len(uncompleted_phenotypes)}")

# split uncompleted phenotypes into 3 files
def split_uncompleted_phenotypes(uncompleted_phenotypes, output_dir, num_files=3):
    uncompleted_phenotypes = list(uncompleted_phenotypes)
    random.shuffle(uncompleted_phenotypes)
    split_size = (len(uncompleted_phenotypes) + num_files - 1) // num_files
    for i in range(num_files):
        split_phenotypes = uncompleted_phenotypes[i*split_size:(i+1)*split_size]
        with open(os.path.join(output_dir, f"part_{i+1}.txt"), "w") as f:
            for pheno in split_phenotypes:
                f.write(f"{pheno}\n")
# split_uncompleted_phenotypes(uncompleted_phenotypes, "out")

#read two gmts and determine which phenotypes are in one gmt but not the other - to determine which phenotypes have lost genes and which phenotypes have gained genes
def compare_gmts(gmt_file1, gmt_file2):
    phenotypes_gmt1 = set()
    phenotypes_gmt2 = set()
    with open(gmt_file1, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 0:
                phenotypes_gmt1.add(parts[0])
    with open(gmt_file2, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 0:
                phenotypes_gmt2.add(parts[0])
    lost_genes_phenotypes = phenotypes_gmt1 - phenotypes_gmt2
    gained_genes_phenotypes = phenotypes_gmt2 - phenotypes_gmt1
    return lost_genes_phenotypes, gained_genes_phenotypes

# lost_genes_phenotypes, gained_genes_phenotypes = compare_gmts("out/genesets/consensus_gene_sets.gmt", "out/direct-prompting/gmts/llama3/genesets_entrez_llama3.gmt")
# print(f"Phenotypes with lost genes: {len(lost_genes_phenotypes)}")
# print(f"Phenotypes with gained genes: {len(gained_genes_phenotypes)}")

# #save lost genes in a text file
# with open("out/lost_genes_phenotypes_dp_llama.txt", "w") as f:
#     for pheno in sorted(lost_genes_phenotypes):
#         f.write(f"{pheno}\n")  


# look through two directories of phenotype json, if it appreas in both delete the one in the first directory
def remove_completed_phenotypes_from_dir(dir1, dir2):
    phenotypes_dir1 = set()
    phenotypes_dir2 = set()
    for fname in os.listdir(dir1):
        if fname.endswith(".json") or fname.endswith("_raw.txt"):
            phenotype_name = os.path.splitext(fname)[0].replace("_raw", "")
            phenotypes_dir1.add(phenotype_name)
    for fname in os.listdir(dir2):
        if fname.endswith(".json") or fname.endswith("_raw.txt"):
            phenotype_name = os.path.splitext(fname)[0].replace("_raw", "")
            phenotypes_dir2.add(phenotype_name)
    completed_phenotypes = phenotypes_dir1.intersection(phenotypes_dir2)
    for pheno in completed_phenotypes:
        file_to_remove = os.path.join(dir1, f"{pheno}.json")
        if os.path.exists(file_to_remove):
            os.remove(file_to_remove)
        file_to_remove_raw = os.path.join(dir1, f"{pheno}_raw.txt")
        if os.path.exists(file_to_remove_raw):
            os.remove(file_to_remove_raw)
# remove_completed_phenotypes_from_dir("out/direct-prompting/phenotype_generations/deepseek-r1:8b", "out/direct-prompting/phenotype_generations/deepseek-r1:8b_1.0")


# read this directory and see if the _raw.txt is the same as the .json files because they should be equal, if not return a list of the
# _raw.txt files that have no .json
def check_raw_txt_vs_json(dir):
    raw_txt_phenotypes = set()
    json_phenotypes = set()
    for fname in os.listdir(dir):
        if fname.endswith(".json"):
            phenotype_name = os.path.splitext(fname)[0]
            json_phenotypes.add(phenotype_name)
        
        elif fname.endswith("_raw.txt"):
            phenotype_name = os.path.splitext(fname)[0].replace("_raw", "")
            raw_txt_phenotypes.add(phenotype_name)
    print(f"json phenotypes: {len(json_phenotypes)}")
    print(f"raw txt phenotypes: {len(raw_txt_phenotypes)}")
    phenotypes_with_raw_no_json = raw_txt_phenotypes - json_phenotypes
    return phenotypes_with_raw_no_json
# phenotypes_with_raw_no_json = check_raw_txt_vs_json("out/gene_generations")
# print(f"Phenotypes with _raw.txt but no .json: {len(phenotypes_with_raw_no_json)}")


# read gmt and remove _ from phenotype names and save to same gmt file
def remove_underscores_from_gmt(gmt_file):
    lines = []
    with open(gmt_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 0:
                parts[0] = parts[0].replace("_", " ")
                lines.append("\t".join(parts))
    with open(gmt_file, "w") as f:
        for line in lines:
            f.write(f"{line}\n")
# remove_underscores_from_gmt("out/direct-prompting/gmts/genesets_entrez_qwen.gmt")

# compare the phenotypes in the gmt file with the phenotypes in the directory and see if there are any phenotypes in the gmt file that are not in the directory - to determine which phenotypes have been processed for the gmt construction and which have not been processed yet
def compare_gmt_with_dir(gmt_file, dir):
    gmt_phenotypes = set()
    with open(gmt_file, "r") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) > 0:
                gmt_phenotypes.add(parts[0])
    dir_phenotypes = set()
    for fname in os.listdir(dir):
        if fname.endswith(".json") or fname.endswith("_raw.txt"):
            phenotype_name = os.path.splitext(fname)[0].replace("_raw", "")
            #remove _ from phenotype_name
            phenotype_name = phenotype_name.replace("_", " ")
            dir_phenotypes.add(phenotype_name)
    unprocessed_phenotypes = dir_phenotypes- gmt_phenotypes
    # these unnprocessed ones do not have any genes returned by the model
    # so add to the gmt them put with empty gene sets
    with open(gmt_file, "a") as f:
        for pheno in sorted(unprocessed_phenotypes):
            f.write(f"{pheno}\t{pheno}_empty\n")    
    return unprocessed_phenotypes
unprocessed_phenotypes = compare_gmt_with_dir("out/direct-prompting/gmts/genesets_symbols_qwen.gmt", "out/gene_generations")
print(f"Phenotypes in gmt but not in dir: {len(unprocessed_phenotypes)}")