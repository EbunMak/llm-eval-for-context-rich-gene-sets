import os
import json
from typing import Dict, List
from utils import id_mapping


def extract_llama3_genes(data: dict) -> List[str]:
    genes = []
    for entry in data.get("genes", []):
        if isinstance(entry.get("gene"), list):
            genes.extend(entry["gene"])
    return list(dict.fromkeys(genes))


def build_gmts_from_dir(
    extracted_dir: str,
    out_symbols: str,
    out_entrez: str,
    model_tag: str,
):
    os.makedirs(os.path.dirname(out_symbols), exist_ok=True)

    unmapped: Dict[str, List[str]] = {}

    with open(out_symbols, "w") as f_sym, open(out_entrez, "w") as f_ent:
        for fname in sorted(os.listdir(extracted_dir)):
            if not fname.endswith(".json"):
                continue

            phenotype = os.path.splitext(fname)[0]
            fpath = os.path.join(extracted_dir, fname)

            with open(fpath, "r") as f:
                data = json.load(f)

            genes = extract_llama3_genes(data)
            if not genes:
                continue

            mapped_ids, valid_syms, invalid_syms = id_mapping(
                genes, mode="entrezgene"
            )

            if invalid_syms:
                unmapped[phenotype] = invalid_syms

            if not valid_syms:
                continue

            # SYMBOL GMT
            f_sym.write(
                phenotype
                + f"\t{model_tag}_extracted(symbols)\t"
                + "\t".join(valid_syms)
                + "\n"
            )

            # ENTREZ GMT
            f_ent.write(
                phenotype
                + f"\t{model_tag}_extracted(entrez)\t"
                + "\t".join(map(str, mapped_ids))
                + "\n"
            )

    return unmapped


def main():
    model = "llama3"
    extracted_dir = f"out/direct-prompting/phenotype_generations/llama3.1:8b_1.0"
    out_dir = f"out/direct-prompting/gmts"

    symbols_gmt = os.path.join(out_dir, f"genesets_symbols_{model}.gmt")
    entrez_gmt  = os.path.join(out_dir, f"genesets_entrez_{model}.gmt")
    unmapped_out = os.path.join(out_dir, f"unmapped_genes_{model}.json")

    unmapped = build_gmts_from_dir(
        extracted_dir,
        symbols_gmt,
        entrez_gmt,
        model
    )

    with open(unmapped_out, "w") as f:
        json.dump(unmapped, f, indent=2)


if __name__ == "__main__":
    main()
