"""
Gene Query Pipeline using Groq API (Qwen3-32B)
Processes phenotype JSON and generates associated gene lists.
Follows the structure of direct_prompting_utils.py and direct_prompting_split.py
"""

import json
import os
import time
import re
import threading
from typing import List, Dict, Any, Set
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage


# ============================================================================
# CONFIGURATION
# ============================================================================

GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
MODEL_NAME = "qwen/qwen3-32b"
MAX_WORKERS = 5  # Tune based on your paid tier rate limits

# Thread-safe lock for writing to the processed file
_processed_lock = threading.Lock()

# Prompt Instructions
DIRECT_PROMPTING_INSTRUCTIONS = """
Using your internal biological knowledge, list genes associated with:

{context}

Return your answer as multiple separate JSON objects, one per line (NDJSON format).
Each line must be a complete, self-contained JSON object with a single key "gene" and a single gene symbol string as the value.

Example:
{{"gene": "GJB2"}}
{{"gene": "MYO7A"}}
{{"gene": "OTOF"}}

Output one gene per line. Stop when you have listed all confident genes. Do not output anything else.
"""


# ============================================================================
# LLM INITIALIZATION
# ============================================================================

def get_llm(model_name: str = MODEL_NAME) -> ChatGroq:
    """Initialize Groq ChatGroq client."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY environment variable not set")
    
    return ChatGroq(
        model=model_name,
        temperature=0,
        api_key=GROQ_API_KEY,
    )


# ============================================================================
# JSON PARSING & VALIDATION
# ============================================================================

def safe_json_loads_ndjson(raw_output: str) -> List[str]:
    """
    Parse NDJSON output - one JSON object per line.
    Each line should have format: {"gene": "SYMBOL"}
    Returns list of gene symbols.
    """
    genes = []
    for line in raw_output.strip().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
            if isinstance(obj, dict) and "gene" in obj:
                gene = obj["gene"]
                if isinstance(gene, str) and gene.strip():
                    genes.append(gene.strip())
        except json.JSONDecodeError:
            continue
    return genes


def filter_hallucinated_genes(genes: List[str]) -> List[str]:
    """Remove obvious hallucinated gene symbols."""
    valid_format = [
        g for g in genes
        if isinstance(g, str) and re.match(r'^[A-Z][A-Z0-9\-]{0,19}$', g)
    ]
    
    def is_likely_hallucinated(symbol: str) -> bool:
        m = re.match(r'^([A-Z]+\d*[A-Z]*)(\d+)$', symbol)
        if m:
            number = int(m.group(2))
            return number > 100
        return False
    
    return [g for g in valid_format if not is_likely_hallucinated(g)]


# ============================================================================
# PHENOTYPE PROCESSING
# ============================================================================

def build_context(phenotype: Dict[str, Any]) -> str:
    """Build context string from phenotype data."""
    context_parts = []
    
    if "name" in phenotype:
        context_parts.append(f"{phenotype['name']}")
    if "id" in phenotype:
        context_parts.append(f"({phenotype['id']})")
    if "description" in phenotype:
        context_parts.append(f"Defined as: {phenotype['description']}")
    if "synonyms" in phenotype and phenotype["synonyms"]:
        synonyms = phenotype["synonyms"]
        if isinstance(synonyms, list):
            context_parts.append(f"Synonyms: {', '.join(synonyms)}")
        elif isinstance(synonyms, str):
            context_parts.append(f"Synonyms: {synonyms}")
    
    return " ".join(context_parts)


def generate_genes(phenotype: Dict[str, Any], llm: ChatGroq, output_dir: str = "out/gene_generations") -> Dict[str, Any]:
    """Generate gene list for a phenotype using Groq API."""
    phenotype_name = phenotype.get("name", "unknown")
    print(f"  → Processing: {phenotype_name}")
    
    context_text = build_context(phenotype)
    
    messages = [
        SystemMessage(content="You are an expert in human genetics. Respond ONLY in valid NDJSON format. Do not explain. Do not add commentary."),
        HumanMessage(content=DIRECT_PROMPTING_INSTRUCTIONS.format(context=context_text))
    ]
    
    os.makedirs(output_dir, exist_ok=True)
    safe_name = re.sub(r'[^a-zA-Z0-9_-]', '_', phenotype_name)
    json_outfile = os.path.join(output_dir, f"{safe_name}.json")
    raw_outfile = os.path.join(output_dir, f"{safe_name}_raw.txt")
    
    genes = []
    
    try:
        result = llm.invoke(messages)
        raw_output = result.content.strip()
        
        with open(raw_outfile, "w") as f:
            f.write(raw_output)
        
        genes = safe_json_loads_ndjson(raw_output)
        genes = list(set(genes))  # deduplicate
        print(f"  ✓ {phenotype_name}: {len(genes)} genes")
        
    except Exception as e:
        print(f"  ✗ ERROR ({phenotype_name}): {e}")
        with open(raw_outfile, "w") as f:
            f.write(f"ERROR: {str(e)}")
    
    result_dict = {"genes": genes}
    with open(json_outfile, "w") as f:
        json.dump(result_dict, f, indent=2)
    
    return {
        "phenotype": phenotype_name,
        "phenotype_id": phenotype.get("id", "N/A"),
        "genes": genes,
        "gene_count": len(genes),
        "output_file": json_outfile
    }


# ============================================================================
# BATCH PROCESSING — PARALLEL
# ============================================================================

def load_phenotypes_json(input_file: str) -> List[Dict[str, Any]]:
    """Load phenotypes from JSON file."""
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file not found: {input_file}")
    with open(input_file, "r") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    elif isinstance(data, dict) and "phenotypes" in data:
        return data["phenotypes"]
    else:
        return [data]


def load_processed_phenotypes(processed_file: str) -> Set[str]:
    """Load phenotype names that have already been processed."""
    if not os.path.exists(processed_file):
        return set()
    with open(processed_file, "r") as f:
        return set(line.strip() for line in f if line.strip())


def mark_phenotype_processed(phenotype_name: str, processed_file: str) -> None:
    """Thread-safe: mark a phenotype as processed."""
    with _processed_lock:
        with open(processed_file, "a") as f:
            f.write(f"{phenotype_name}\n")


def process_single(phenotype: Dict[str, Any], llm: ChatGroq, output_dir: str, processed_file: str) -> Dict[str, Any]:
    """Worker function: process one phenotype and mark it done."""
    result = generate_genes(phenotype, llm, output_dir)
    mark_phenotype_processed(phenotype.get("name", "unknown"), processed_file)
    return result


def process_phenotypes(
    input_file: str,
    output_dir: str = "out/direct-prompting/phenotype_generations/qwen-refix",
    processed_file: str = "out/direct-prompting/processed_phenotypes_qwen3_refix.txt",
    limit: int = None,
    max_workers: int = MAX_WORKERS
) -> List[Dict[str, Any]]:
    """Process all phenotypes in parallel using a thread pool."""
    print("=" * 80)
    print(f"GENE QUERY PIPELINE - GROQ API (QWEN3-32B) — {max_workers} workers")
    print("=" * 80)
    
    phenotypes = load_phenotypes_json(input_file)
    print(f"Loaded {len(phenotypes)} phenotypes from {input_file}")
    
    processed = load_processed_phenotypes(processed_file)
    print(f"Already processed: {len(processed)}")
    
    to_process = [p for p in phenotypes if p.get("name") not in processed]
    if limit:
        to_process = to_process[:limit]
    print(f"Remaining to process: {len(to_process)} — launching {max_workers} parallel workers\n")
    
    if not to_process:
        print("All phenotypes already processed!")
        return []
    
    # Each worker gets its own LLM instance to avoid sharing state
    llm_pool = [get_llm(MODEL_NAME) for _ in range(max_workers)]
    
    results = []
    os.makedirs(output_dir, exist_ok=True)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                process_single,
                phenotype,
                llm_pool[idx % max_workers],  # round-robin LLM instances
                output_dir,
                processed_file
            ): phenotype
            for idx, phenotype in enumerate(to_process)
        }
        
        for future in as_completed(futures):
            phenotype = futures[future]
            try:
                result = future.result()
                results.append(result)
                completed += 1
                print(f"[{completed}/{len(to_process)}] done: {result['phenotype']} ({result['gene_count']} genes)")
            except Exception as e:
                print(f"  ✗ FAILED: {phenotype.get('name', 'unknown')} — {e}")

    print("\n" + "=" * 80)
    print("PROCESSING COMPLETE")
    print("=" * 80)
    print(f"Total processed: {len(results)}")
    print(f"Total genes generated: {sum(r['gene_count'] for r in results)}")
    print(f"Output directory: {output_dir}")
    
    return results


# ============================================================================
# SINGLE PHENOTYPE QUERY (interactive use)
# ============================================================================

def query_single_phenotype(
    name: str,
    description: str = None,
    synonyms: List[str] = None,
    phenotype_id: str = None,
    output_dir: str = "out/gene_generations"
) -> Dict[str, Any]:
    """Query for genes associated with a single phenotype."""
    phenotype = {"name": name, "description": description, "synonyms": synonyms or []}
    if phenotype_id:
        phenotype["id"] = phenotype_id
    llm = get_llm(MODEL_NAME)
    return generate_genes(phenotype, llm, output_dir)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Query Groq API (Qwen3-32B) for genes associated with phenotypes"
    )
    parser.add_argument("--input", type=str, default="phenotypes.json")
    parser.add_argument("--output", type=str, default="out/gene_generations")
    parser.add_argument("--processed", type=str, default="out/direct-prompting/processed_phenotypes_qwen3_refix.txt")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=MAX_WORKERS, help="Number of parallel workers")
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--description", type=str, default=None)
    
    args = parser.parse_args()
    
    if args.name:
        result = query_single_phenotype(name=args.name, description=args.description, output_dir=args.output)
        print(json.dumps(result, indent=2))
    else:
        process_phenotypes(
            input_file=args.input,
            output_dir=args.output,
            processed_file=args.processed,
            limit=args.limit,
            max_workers=args.workers
        )