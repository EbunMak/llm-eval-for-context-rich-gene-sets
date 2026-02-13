import time
import json
import os
from langgraph.graph import END, StateGraph
from pubtator import Pubtator
from utils import GraphState, LLMGraphState, get_llm, get_llm_json_mode, clean_model_output, save_to_json_list
from langchain_core.messages import HumanMessage, SystemMessage
from instructs import rag_prompt2, grade_abstracts_instructions2

CHECKED_PMIDS_FILE = "checked_pmids_gene_checker.json"

def retrieve_pubtator_abstracts(state: LLMGraphState):
    """
    Retrieve abstracts for a given phenotype + gene.
    - If cached abstracts exist, load them from disk.
    - Otherwise, query PubTator, save the raw abstracts once, and return them.
    """
    phenotype = state["phenotype"]
    query = phenotype["name"].strip()
    gene = phenotype["gene"].strip()

    base_dir = "abstracts/gene_related_abstracts"
    os.makedirs(base_dir, exist_ok=True)
    cache_file = os.path.join(base_dir, f"{query}_{gene}.json")

    # If cached, just load and return
    if os.path.exists(cache_file):
        print(f"Loading cached abstracts for {query} / {gene}")
        with open(cache_file, "r") as f:
            abstracts = json.load(f)
        return {"documents": abstracts}

    # Else: query PubTator directly
    pmids = Pubtator.search_pubtator_ID(relation=f"@GENE_{gene} AND {query}", limit=1)
    print(f"Fetched {len(pmids)} PMIDs for {gene} and {query}")

    abstracts = []
    for pmid in pmids:
        try:
            abs_data = Pubtator.export_abstract(pmid, check_for_genes=False)
            if abs_data:
                abstracts.append(abs_data)
        except Exception as e:
            print(f"Error fetching PMID {pmid}: {e}")

    # Save raw abstracts once
    save_to_json_list(abstracts, cache_file)
    print(f"Saved {len(abstracts)} abstracts to {cache_file}")

    # Return to the shared 'documents' list
    return {"documents": abstracts}


def grade_abstracts(state: LLMGraphState):
    """
    Grade abstracts for phenotype+gene relevance.
    Saves to state['documents_filtered'][llm_name].
    """
    llm_name = state.get("llm_name", "deepseek-r1:8b")
    phenotype = state["phenotype"]
    gene = phenotype["gene"]

    # 1. Access shared raw documents
    documents = state.get("documents", [])
    if not documents:
        print(f"No abstracts to grade for {phenotype['name']} / {gene}")
        # Return empty list for this LLM
        return {"documents_filtered": {llm_name: []}}

    question = (
        f"Does this abstract discuss BOTH the gene '{gene}' "
        f"and the phenotype '{phenotype['name']}' meaning ({phenotype.get('definition', 'N/A')})?"
    )

    llm = get_llm_json_mode(llm_name)
    filtered = []

    for doc in documents:
        abstract_text = (
            f"Title: {doc.get('title', '')}\n"
            f"Abstract: {doc.get('abstract', '')}"
        )
        
        try:
            result = llm.invoke([
                SystemMessage(content=grade_abstracts_instructions2),
                HumanMessage(content=f"Question: {question}\n\nAbstract:\n{abstract_text}")
            ])

            grade = json.loads(result.content)["binary_score"].strip().lower()
            if grade == "yes":
                filtered.append(doc)
        except Exception as e:
            print(f"[{llm_name}] Skipping abstract due to parse error: {e}")
            continue

    print(f"[{llm_name}] Kept {len(filtered)} abstracts after grading for {phenotype['name']} / {gene}")
    
    # 2. Return using the nested structure
    return {"documents_filtered": {llm_name: filtered}}


def generate(state: LLMGraphState):
    """
    Use only the filtered abstracts for this specific LLM to validate the association.
    """
    llm_name = state.get("llm_name", "deepseek-r1:8b")

    phenotype = state["phenotype"]
    gene = phenotype["gene"]
    safe_name = phenotype["name"]

    # 1. Retrieve filtered documents specifically for this LLM
    all_filtered = state.get("documents_filtered", {})
    documents = all_filtered.get(llm_name, [])

    if not documents:
        print(f"[{llm_name}] No filtered abstracts for {safe_name} / {gene}")
        return {"generation": {llm_name: []}}

    pmids = [d.get("pmid") for d in documents]
    formatted_docs = [
        f"PMID: {d.get('pmid')}\nTitle: {d.get('title')}\nJournal: {d.get('journal')}\nAbstract: {d.get('abstract')}"
        for d in documents
    ]
    context = "\n\n".join(formatted_docs)

    question = f"Is gene '{gene}' supported as being associated with phenotype '{phenotype['name']}'?"

    llm = get_llm_json_mode(llm_name)
    
    try:
        result = llm.invoke([
            SystemMessage(content="You are a precise biomedical reasoning model. Respond only in JSON."),
            HumanMessage(content=rag_prompt2.format(context=context, question=question))
        ])
        generation = json.loads(result.content)
    except Exception:
        # Fallback cleaning
        try:
            generation = json.loads(clean_model_output(result.content))
        except:
            generation = {}

    generation["PMIDS"] = pmids

    # Save to disk
    outfile = f"out/phenotype_checks/{llm_name}/{safe_name}/{gene}.json"
    os.makedirs(os.path.dirname(outfile), exist_ok=True)
    with open(outfile, "w") as f:
        json.dump(generation, f, indent=2)
    print(f"[{llm_name}] Saved generation for {gene} and {safe_name} to {outfile}")

    # 2. Return wrapped in a list to match LLMGraphState List type
    return {"generation": {llm_name: [generation]}}



def create_control_flow():
    # Ensure LLMGraphState in utils.py is updated to include `merge_dicts` logic!
    workflow = StateGraph(LLMGraphState)

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

# def create_control_flow():
    # Ensure LLMGraphState in utils.py is updated to include `merge_dicts` logic!
    # workflow = StateGraph(LLMGraphState)

    # # 1. Retrieve Node
    # workflow.add_node("retrieve", retrieve_pubtator_abstracts)

    # # 2. Grading Nodes (Parallel logic supported by StateGraph, though edges here are sequential)
    # workflow.add_node("grade_qwen", lambda s: grade_abstracts(s, "qwen3:32b"))
    # workflow.add_node("grade_deepseek", lambda s: grade_abstracts(s, "deepseek-r1:8b"))
    # workflow.add_node("grade_llama3", lambda s: grade_abstracts(s, "llama3.1:8b"))

    # # 3. Generation Nodes
    # workflow.add_node("generate_qwen", lambda s: generate(s, "qwen3:32b"))
    # workflow.add_node("generate_deepseek", lambda s: generate(s, "deepseek-r1:8b"))
    # workflow.add_node("generate_llama3", lambda s: generate(s, "llama3.1:8b"))

    # # Entry
    # workflow.set_entry_point("retrieve")

    # # Flow
    # # Retrieve -> Grade Qwen -> Grade Deepseek -> Grade Llama
    # workflow.add_edge("retrieve", "grade_qwen")
    # workflow.add_edge("grade_qwen", "grade_deepseek")
    # workflow.add_edge("grade_deepseek", "grade_llama3")

    # # Grade Llama -> Gen Qwen -> Gen Deepseek -> Gen Llama
    # workflow.add_edge("grade_llama3", "generate_qwen")
    # workflow.add_edge("generate_qwen", "generate_deepseek")
    # workflow.add_edge("generate_deepseek", "generate_llama3")

    # # End
    # workflow.add_edge("generate_llama3", END)

    # return workflow.compile()