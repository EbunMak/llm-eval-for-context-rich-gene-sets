import time
import json 
import os
import asyncio
from langgraph.graph import END, StateGraph
from pubtator import Pubtator
from utils import AAGraphState, get_llm, get_llm_json_mode, clean_model_output, check_is_gene_annotated
from langchain_core.messages import HumanMessage, SystemMessage
from instructs import rag_prompt, grade_abstracts_instructions

CHECKED_PMIDS_FILE = "checked_pmids.json"
PMIDS_FILE = "abstracts/pmids.txt"

# Load gene-annotated PMIDs once
if os.path.exists(PMIDS_FILE):
    with open(PMIDS_FILE, "r") as f:
        ga_pmids = list({int(line.strip()) for line in f if line.strip()})
else:
    ga_pmids = []
    print(f"Warning: {PMIDS_FILE} not found. No PMIDs loaded.")

def retrieve_pubtator_abstracts(state: AAGraphState):
    """
    Retrieves raw abstracts from PubTator and stores them in state['documents'].
    """
    phenotype = state["phenotype"]
    name = phenotype["name"].strip()
    num_of_abstracts = state["num_of_abstracts"]
    outfile = f"abstracts/gene_annotated_abstracts/{name}.json"

    # If already downloaded, just load and return
    if os.path.exists(outfile):
        print(f"Cached abstracts found for {name}. Loading...")
        with open(outfile, "r") as f:
            # check if num_of_abstracts limit applies
            if num_of_abstracts > 0:
                all_abstracts = json.load(f)
                if len(all_abstracts) >= num_of_abstracts:
                    return {"documents": all_abstracts[:num_of_abstracts]}
                # if the cached abstracts are more than the limit, continue to download more
            else:
                return {"documents": all_abstracts}

    # Else: download abstracts
    print(f"Downloading abstracts for phenotype: {name}")

    if os.path.exists(CHECKED_PMIDS_FILE):
        with open(CHECKED_PMIDS_FILE, "r") as f:
            checked_pmids = json.load(f)
    else:
        checked_pmids = {}

    num_pages = num_of_abstracts // 10 + (1 if num_of_abstracts % 10 != 0 else 0) if num_of_abstracts > 0 else None
    print(f"Fetching up to {num_of_abstracts} abstracts ({num_pages} pages)")
    pmids = Pubtator.search_pubtator_ID(query=name, limit=num_pages)
    pmids = check_is_gene_annotated(pmids, ga_pmids)

    abstracts = []
    MAX_REQUESTS_PER_SECOND = 3
    DELAY = 1.0 / MAX_REQUESTS_PER_SECOND

    for pmid in pmids:
        if str(pmid) in checked_pmids and not checked_pmids[str(pmid)]["has_genes"]:
            continue

        try:
            abs_data = Pubtator.export_abstract(pmid)
            has_genes = abs_data is not None
            checked_pmids[str(pmid)] = {"has_genes": has_genes}

            if has_genes:
                abstracts.append(abs_data)

            time.sleep(DELAY)
        except Exception:
            time.sleep(DELAY)
            continue

    # Write updated PMIDs file
    with open(CHECKED_PMIDS_FILE, "w") as f:
        json.dump(checked_pmids, f, indent=2)

    # Save raw abstracts
    os.makedirs("abstracts/gene_annotated_abstracts", exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(abstracts, f, indent=2)

    print(f"Saved {len(abstracts)} abstracts to {outfile}")
    
    # Return to the shared 'documents' key
    return {"documents": abstracts}


def grade_abstracts(state: AAGraphState):
    """
    Grades abstracts and saves them to state['documents_filtered'][llm_name].
    """
    llm_name = state.get("llm_name", "deepseek-r1:8b")
    print(f"---CHECK ABSTRACT RELEVANCE ({llm_name})---")

    phenotype = state["phenotype"]
    # Read from shared raw documents
    documents = state.get("documents", [])

    if not documents:
        # Return empty list for this specific LLM
        return {"documents_filtered": {llm_name: []}}

    question = (
        f"Is this abstract relevant to the phenotype '{phenotype['name']}', "
        f"defined as '{phenotype.get('definition', 'N/A')}', or its synonyms: "
        f"{', '.join(phenotype.get('synonyms', [])) if phenotype.get('synonyms') else 'None'}?"
    )

    llm = get_llm_json_mode(llm_name)
    filtered = []

    for doc in documents:
        abstract_text = (
            f"Title: {doc.get('title','')}\n"
            f"Journal: {doc.get('journal','')}\n"
            f"Abstract: {doc.get('abstract','')}"
        )

        try:
            result = llm.invoke([
                SystemMessage(content=grade_abstracts_instructions),
                HumanMessage(content=f"Question: {question}\n\nAbstract:\n{abstract_text}")
            ])
            
            # Parse JSON score
            grade_data = json.loads(result.content)
            grade = grade_data.get("binary_score", "no").strip().lower()
            
            if grade == "yes":
                filtered.append(doc)
        except Exception as e:
            # print(f"Error grading doc with {llm_name}: {e}")
            continue

    print(f"[{llm_name}] Filtered {len(documents)} -> {len(filtered)} docs")
    
    # RETURN format: { "key_name": { "llm_name": [value] } }
    # The merge_dicts reducer in utils.py will handle merging this into the state
    return {"documents_filtered": {llm_name: filtered}}


def safe_json_loads(raw_output):
    """
    Safely parse LLM output into JSON, attempting to repair common truncation issues.
    """
    import json
    try:
        return json.loads(raw_output)
    except json.JSONDecodeError as e:
        partial = raw_output[:e.pos]
        # Try to balance braces/brackets
        if partial.count("{") > partial.count("}"):
            partial += "}"
        if partial.count("[") > partial.count("]"):
            partial += "]"
        try:
            return json.loads(partial)
        except Exception:
            return []


def generate(state: AAGraphState):
    """
    Generates gene list using state['documents_filtered'][llm_name].
    Saves result to state['generation'][llm_name].
    """
    llm_name = state.get("llm_name", "deepseek-r1:8b")
    print(f"---GENERATE ({llm_name})---")

    phenotype = state["phenotype"]
    safe_name = phenotype["name"]

    # 1. Retrieve filtered docs specific to this LLM
    # state["documents_filtered"] is a Dict[str, List]
    all_filtered = state.get("documents_filtered", {})
    documents = all_filtered.get(llm_name, [])

    if not documents:
        print(f"No filtered abstracts for {llm_name}")
        return {"generation": {llm_name: []}}

    # Build context
    context_text = "\n\n".join([
        f"PMID: {d.get('pmid')}\nTitle: {d.get('title')}\nJournal: {d.get('journal')}\nAbstract: {d.get('abstract')}"
        for d in documents
    ])

    question = (
        f"Identify genes associated with phenotype '{phenotype['name']}', "
        f"definition: '{phenotype.get('definition','N/A')}'."
    )

    llm = get_llm_json_mode(llm_name)

    messages = [
        SystemMessage(content="You are a precise biomedical text mining assistant. Respond only in valid JSON."),
        HumanMessage(content=rag_prompt.format(context=context_text, question=question))
    ]

    # Prepare output paths for debugging/logging
    out_dir = f"out/aa/{state['num_of_abstracts']}/phenotype_generations/{llm_name}"
    os.makedirs(out_dir, exist_ok=True)
    json_outfile = f"{out_dir}/{safe_name}.json"
    raw_outfile = f"{out_dir}/{safe_name}_raw.txt"

    generation = []
    
    try:
        result = llm.invoke(messages)
        raw_output = result.content.strip()

        # Try parsing
        generation = safe_json_loads(raw_output)

        # Fallback cleanup if parsing failed
        if not generation:
            print(f"[{llm_name}] Invalid JSON. Attempting cleanup...")
            cleaned = clean_model_output(raw_output)
            generation = safe_json_loads(cleaned)
            
            # Save raw output if cleanup also fails
            if not generation:
                with open(raw_outfile, "w") as f:
                    f.write(raw_output)
    
    except Exception as e:
        print(f"[{llm_name}] Invocation error: {e}")
        with open(raw_outfile, "w") as f:
            f.write(str(e))

    # Save result to disk
    if generation:
        with open(json_outfile, "w") as f:
            json.dump(generation, f, indent=2)
        print(f"[{llm_name}] Saved to {json_outfile}")
    
    # RETURN format: { "key_name": { "llm_name": [value] } }
    return {"generation": {llm_name: generation}}


def create_control_flow():
    # Ensure AAGraphState in utils.py is updated to include `merge_dicts` logic!
    workflow = StateGraph(AAGraphState)

    # 1. Retrieve Node
    workflow.add_node("retrieve", retrieve_pubtator_abstracts)

    # 2. Grading Nodes (Parallel logic supported by StateGraph, though edges here are sequential)
    workflow.add_node("grade", lambda s: grade_abstracts(s))
    # workflow.add_node("grade_qwen", lambda s: grade_abstracts(s, "qwen3:32b"))
    # workflow.add_node("grade_deepseek", lambda s: grade_abstracts(s, "deepseek-r1:8b"))
    # workflow.add_node("grade_llama3", lambda s: grade_abstracts(s, "llama3.1:8b"))

    # 3. Generation Nodes
    workflow.add_node("generate", lambda s: generate(s))
    # workflow.add_node("generate_qwen", lambda s: generate(s, "qwen3:32b"))
    # workflow.add_node("generate_deepseek", lambda s: generate(s, "deepseek-r1:8b"))
    # workflow.add_node("generate_llama3", lambda s: generate(s, "llama3.1:8b"))

    # Entry
    workflow.set_entry_point("retrieve")
    # # Flow
    # # Retrieve -> Grade Qwen -> Grade Deepseek -> Grade Llama
    # workflow.add_edge("retrieve", "grade_deepseek")
    # workflow.add_edge("grade_deepseek", "generate_deepseek")
    # workflow.add_edge("generate_deepseek",  END)


    # Flow
    # Retrieve -> Grade Qwen -> Grade Deepseek -> Grade Llama
    workflow.add_edge("retrieve", "grade")
    workflow.add_edge("grade", "generate")
    # workflow.add_edge("grade_qwen", "grade_deepseek")
    # workflow.add_edge("grade_deepseek", "grade_llama3")

    # # Grade Llama -> Gen Qwen -> Gen Deepseek -> Gen Llama
    # workflow.add_edge("grade_llama3", "generate_qwen")
    # workflow.add_edge("generate_qwen", "generate_deepseek")
    # workflow.add_edge("generate_deepseek", "generate_llama3")

    # End
    # workflow.add_edge("generate_llama3", END)
    workflow.add_edge("generate", END)

    return workflow.compile()