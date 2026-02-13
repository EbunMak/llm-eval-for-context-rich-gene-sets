import time
import json 
import os
import asyncio
from langgraph.graph import END, StateGraph
from pubtator import Pubtator
from utils import DPGraphState, get_llm, get_llm_json_mode, clean_model_output, check_is_gene_annotated
from langchain_core.messages import HumanMessage, SystemMessage
from instructs import direct_prompting_rag_instructions, direct_prompting_rag_instructions_2

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


def generate(state: DPGraphState):
    """
    Generates gene list using through direct prompting.
    """
    llm_name = state.get("llm_name", "deepseek-r1:8b")
    print(f"---GENERATE ({llm_name})---")

    phenotype = state["phenotype"]
    safe_name = phenotype["name"]

    context_text = (
        f"{phenotype['name']} ({phenotype['id']}), "
        # f"defined as {phenotype.get('definition', 'N/A')}. "
        # f"Synonyms: "
        # f"{', '.join(phenotype.get('synonyms', [])) if phenotype.get('synonyms') else 'None'}."
    )
    llm = get_llm_json_mode(llm_name)

    messages = [
        SystemMessage(content="You are an expert in cellular and molecular biology.. Respond only in valid JSON."),
        HumanMessage(content=direct_prompting_rag_instructions_2.format(context=context_text))
    ]

    # Prepare output paths for debugging/logging
    out_dir = f"out/direct-prompting/phenotype_generations/{llm_name}"
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
    # Ensure DPGraphState in utils.py is updated to include `merge_dicts` logic!
    workflow = StateGraph(DPGraphState)

    # 1. Generation Nodes
    workflow.add_node("generate", lambda s: generate(s))
    # Entry
    workflow.set_entry_point("generate")

    # End
    workflow.add_edge("generate", END)

    return workflow.compile()