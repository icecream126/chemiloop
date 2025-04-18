# LangGraph-based multi-agent molecular generation system with tool execution, reflection, and feedback

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Union
from langchain_deepseek import ChatDeepSeek
from langchain.schema import HumanMessage
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from utils.args import parse_args, return_API_keys
from guacamol.utils.chemistry import canonicalize
from guacamol.common_scoring_functions import TanimotoScoringFunction
import utils.utils
from utils.metrics import get_isomer_c7h8n2o2_score, get_isomer_c9h10n2o2pf2cl_score, get_albuterol_similarity_score
import get_v2_prompts
import get_v2_json_prompts
from openai import OpenAI


import json
import csv
import os
import wandb
import re
import datetime

wandb.init(project="json_1000_pmo_v2_albutero_smilarity", name="history_pmo")# , mode='disabled')

current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_dir = f"./logs/{current_time}-{wandb.run.id}/"
os.makedirs(log_dir, exist_ok=True)

OPENAI_API_KEY = return_API_keys()["OPENAI_API_KEY"] 
DEEPSEEK_API_KEY = return_API_keys()["DEEPSEEK_API_KEY"]
LOG_PATH = f"{log_dir}log.txt"
SMILES_LOG_PATH = f"{log_dir}smiles.txt"
BEST_SMILES_LOG_PATH = f"{log_dir}best_smiles.txt"

scientist_llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)
reviewer_llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)

json_scientist_llm = OpenAI(
    api_key=DEEPSEEK_API_KEY,  # Replace with your actual API key
    base_url="https://api.deepseek.com"
)
json_reviewer_llm = OpenAI(
    api_key=DEEPSEEK_API_KEY,  # Replace with your actual API key
    base_url="https://api.deepseek.com"
)

albuterol_smiles = 'CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O'
albuterol_canonical_smiles = canonicalize(albuterol_smiles) # 'CC(C)(C)NCC(O)c1ccc(O)c(CO)c1'
albuterol_mol = Chem.MolFromSmiles(albuterol_smiles)
albuterol_functional_group = utils.utils.describe_albuterol_features(albuterol_mol)



scoring_fn = TanimotoScoringFunction(
    target=albuterol_canonical_smiles,
    fp_type='AP'  # you can also try 'AP', 'FCFP' etc.
)

BEST_SCORE = 0.0

# --------------------------
# Logging helper
# --------------------------
def log(msg: str):
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")

def SMILES_log(msg: str):
    with open(SMILES_LOG_PATH, "a") as f:
        f.write(msg + "\n")

def BEST_SMILES_log(msg: str):
    with open(BEST_SMILES_LOG_PATH, "a") as f:
        f.write(msg + "\n")

SMILES_HISTORY_LOG_PATH = f"{log_dir}smiles_history.txt"
def SMILES_HISTORY_log(msg: str):
    with open(SMILES_HISTORY_LOG_PATH, "a") as f:
        f.write(msg + "\n")
# --------------------------
# Graph State
# --------------------------
class GraphState(TypedDict):
    prompt: str
    iteration: int
    max_iterations: int
    reviewer_think: Dict[str, str]
    scientist_think: Dict[str, str]
    generated_smiles: str
    target_prop: List[str]
    score: float
    functional_groups: str
    json_output: bool

# --------------------------
# Scientist node
# --------------------------

SMILES = ""
BEST_SMILES = ""
BEST_SCORE = 0.0
oracle_buffer = []
BEST_TOP_10_AUC_ALL = 0.0
BEST_TOP_10_AUC_NO_1 = 0.0
SMILES_HISTORY = set()

def scientist_node(state: GraphState) -> GraphState:
    global SMILES
    global json_scientist_llm
    global scientist_llm
    global SMILES_HISTORY
    print("\n==== Scientist Node ==")

    TEXT_SMILES_HISTORY = utils.utils.format_set_as_text(SMILES_HISTORY)
    if state["json_output"]:
        if state["reviewer_think"] == "":
            user_prompt = get_v2_json_prompts.get_scientist_prompt(state["prompt"], TEXT_SMILES_HISTORY)
        else:
            user_prompt = get_v2_json_prompts.get_scientist_prompt_with_review(state["prompt"], state['scientist_think'], state['reviewer_think'], state["generated_smiles"], state["score"], state["functional_groups"], TEXT_SMILES_HISTORY)
        system_prompt = f"You are a skilled chemist."  # (include full system instructions here)
        # user_prompt = get_v2_json_prompts.get_scientist_prompt(state["prompt"], SMILES_HISTORY)  # (include full user prompt here)
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Call the model with enforced JSON output
        raw_response = json_scientist_llm.chat.completions.create(
            model="deepseek-chat",
            messages=prompt,
            response_format={"type": "json_object"},
            temperature=1.0,
        )

        # Since the API guarantees a JSON object, you can access it directly:
        try:
            response = raw_response.choices[0].message.content  # Already a JSON string
            result = json.loads(response)  # Just in case it's not parsed automatically
            SMILES = result.get("SMILES", "")
            scientist_think_dict ={
                'step1': result.get("step1", ""),
                'step2': result.get("step2", ""),
                'step3': result.get("step3", ""),
                'SMILES': SMILES,
            }
            if not SMILES:
                print("SMILES field is missing or empty.")
        except Exception as e:
            print("Error extracting SMILES:", e)
            SMILES = ""
            scientist_think_dict = {
                'step1': "",
                'step2': "",
                'step3': "",
                'SMILES': "",
            }
    else:
        prompt = get_v2_prompts.get_scientist_prompt(state["prompt"])
        response = scientist_llm([HumanMessage(content=prompt)]).content.strip()
        match = re.search(r"SMILES:\s*([^\s]+)", response)
        if match:
            SMILES = match.group(1)
        else:
            print("SMILES not found in the response.")
            SMILES = ""

        # Extract scientist's think dictionary from the response
        scientist_think_dict = utils.utils.get_scientist_think_dict(response)

        # process missing value of scientist_think_dict
        for step in ["step1", "step2", "step3"]:
            if step not in scientist_think_dict:
                scientist_think_dict[step] = ""
    
    utils.utils.add_with_limit(SMILES_HISTORY, SMILES)
    SMILES_HISTORY_log(str(SMILES_HISTORY))
    # Printing and logging
    print("Response from scientist node:", response)

    log(f"\n==== Scientist Node - {state['iteration']} ==")
    log("Prompt to scientist node:")
    log(str(prompt))
    log("\nResponse from scientist node:")
    log(str(response))
    
    wandb.log({
        "generated_smiles": SMILES,
    }, step=state["iteration"])

    return {
        **state,
        "scientist_think": scientist_think_dict,
        "generated_smiles": SMILES,
    }


def reviewer_node(state: GraphState) -> GraphState:
    global SMILES, BEST_SCORE, BEST_SMILES, oracle_buffer, BEST_TOP_10_AUC_ALL, BEST_TOP_10_AUC_NO_1
    global json_reviewer_llm
    global reviewer_llm
    mol = Chem.MolFromSmiles(state["generated_smiles"])
    if mol is None:
        print("Invalid SMILES detected, retrying scientist node.")
        score = 0.0
        state["scientist_think"]["SMILES"] += "(This SMILES is invalid, please retry.)"
    else:
        # TODO: Fix score to be list of scores (floats) for multiple target properties
        # define oracle score by target property
        if "molecular weight" in state["target_prop"]:
            score = Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(state["generated_smiles"]))
            # TODO: modify score by diff of target property and pred_value
        elif "albuterol_similarity" in state["target_prop"]:
            score = get_albuterol_similarity_score(state["generated_smiles"])
        else:
            raise NotImplementedError("Target property not implemented")

    SMILES_log(SMILES+" , "+str(score))

    oracle_buffer.append((SMILES, score))

    # update best score
    if int(score) != 1 and score > BEST_SCORE:
        BEST_SCORE = score
        BEST_SMILES = state["generated_smiles"]
        BEST_SMILES_log(BEST_SMILES)
        # log the best score to wandb

    if len(oracle_buffer) >= 10 :
        sorted_all = sorted(oracle_buffer, key=lambda x: x[1], reverse=True)
        top_10_all = sum(score for _, score in sorted_all[:10]) / 10
        sorted_no_ones = [entry for entry in sorted_all if entry[1] < 1.0]
        top_10_no_ones = sum(score for _, score in sorted_no_ones[:10]) / 10 if len(sorted_no_ones) >= 10 else None
        
        scores_all = [s for _, s in oracle_buffer]
        scores_no_ones = [s for _, s in oracle_buffer if s < 1.0]

        auc_top1_all = utils.utils.compute_auc_topk_online_torch(scores_all, k=1)
        auc_top10_all = utils.utils.compute_auc_topk_online_torch(scores_all, k=10)

        auc_top1_no_ones = utils.utils.compute_auc_topk_online_torch(scores_no_ones, k=1) if len(scores_no_ones) >= 1 else -1
        auc_top10_no_ones = utils.utils.compute_auc_topk_online_torch(scores_no_ones, k=10) if len(scores_no_ones) >= 10 else -1

        if auc_top10_all > BEST_TOP_10_AUC_ALL:
            BEST_TOP_10_AUC_ALL = auc_top10_all
        if auc_top10_no_ones > BEST_TOP_10_AUC_NO_1:
            BEST_TOP_10_AUC_NO_1 = auc_top10_no_ones

        wandb.log({
            "top_10_avg_score_all": top_10_all,
            "top_10_avg_score_no_1.0": top_10_no_ones if top_10_no_ones is not None else -1,
            "auc_top1_all": auc_top1_all,
            "auc_top10_all": auc_top10_all,
            "auc_top1_no_1.0": auc_top1_no_ones,
            "auc_top10_no_1.0": auc_top10_no_ones,
        },step=state["iteration"])
        log(f"Top-10 avg (with 1.0): {top_10_all}, Top-10 avg (no 1.0): {top_10_no_ones}")
        log(f"AUC all — Top-1: {auc_top1_all}, Top-10: {auc_top10_all}")
        log(f"AUC no-1.0 — Top-1: {auc_top1_no_ones}, Top-10: {auc_top10_no_ones}")
        log(f"Best AUC all — Top-1: {BEST_TOP_10_AUC_ALL}, Top-10: {BEST_TOP_10_AUC_NO_1}")

    wandb.log({
        "score": score,
        "best_score": BEST_SCORE,
        "best_smiles": BEST_SMILES,
    },step=state["iteration"])

    # analyze albutero-relsted functional group of generated smiles
    if mol is not None:
        functional_groups = utils.utils.describe_albuterol_features(mol)
    else:
        functional_groups = "No functional groups because your SMILES is invalid. Please retry."
    
    if state["json_output"]:
        system_prompt="You are a rigorous chemistry reviewer.\n"
        user_prompt = get_v2_json_prompts.get_reviewer_prompt(state["scientist_think"], score, functional_groups)
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        # Call the model with enforced JSON output
        raw_response = json_reviewer_llm.chat.completions.create(
            model="deepseek-chat",
            messages=prompt,
            response_format={"type": "json_object"},
            temperature=1.0,
        )

        # Since the API guarantees a JSON object, you can access it directly:
        try:
            response = raw_response.choices[0].message.content  # Already a JSON string
            result = json.loads(response)  # Just in case it's not parsed automatically
            SMILES = result.get("SMILES", "")
            reviewer_think_dict ={
                'step1': result.get("step1", ""),
                'step2': result.get("step2", ""),
                'step3': result.get("step3", ""),
            }
            if not SMILES:
                print("SMILES field is missing or empty.")
        except Exception as e:
            print("Error extracting SMILES:", e)
            SMILES = ""
            reviewer_think_dict = {
                'step1': "",
                'step2': "",
                'step3': "",
            }

    else:
        # get reviewer's prompt
        prompt = get_v2_prompts.get_reviewer_prompt(state["prompt"], state['scientist_think'], score, functional_groups)
        
        # get reviewer's response
        response = reviewer_llm([HumanMessage(content=prompt)]).content.strip()

        # get reviewer's think dictionary from the response
        reviewer_think_dict = utils.utils.get_reviewer_think_dict(response)

        # process missing value of reviewer_think_dict
        for step in ["step1", "step2", "step3"]:
            if step not in reviewer_think_dict:
                reviewer_think_dict[step] = ""


    print("Response from reviewer node:", response)
    log("\n==== Reviewer Node ==")
    log("Prompt to reviewer node:")
    log(str(prompt))
    log("\nResponse from reviewer node:")
    log(str(response))
    log("Evaluated score:")
    log(str(score))
    log("Functional groups:")
    log(str(functional_groups))
    return {
        **state,
        "iteration": state["iteration"] + 1,
        "reviewer_think": reviewer_think_dict,
        "score": score,
        "functional_groups": functional_groups,
    }

def should_continue(state: GraphState) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return END
    return "scientist_node"

# --------------------------
# Main execution
# --------------------------
if __name__ == "__main__":
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("scientist_node", scientist_node)
    builder.add_node("reviewer_node", reviewer_node)

    # Set entry point
    builder.set_entry_point("scientist_node")


    # After reviewer, decide whether to continue (reviewer → scientist OR END)
    builder.add_edge("scientist_node", "reviewer_node")
    # builder.add_conditional_edges("scientist_node", is_scientist_SMILES_valid)
    builder.add_conditional_edges("reviewer_node", should_continue)

    # Compile graph
    graph = builder.compile()

    input_state: GraphState = {
        "prompt": f"Design a drug-like molecule structurally similar to albuterol (SMILES: {albuterol_smiles}, canonical: {albuterol_canonical_smiles}). Preserve the core scaffold and key functional groups. Albuterol contains: {albuterol_functional_group}.",
        "iteration": 0,
        "max_iterations": 1000,
        "scientist_think": "",
        "reviewer_think": "",
        "target_prop": ["albuterol_similarity"],
        "score": 0.0,
        "functional_groups": "",
        "generated_smiles": "",
        "json_output": True,
    }
    # Run the graph
    final_state = graph.invoke(input_state, {"recursion_limit": 9999})

    # Print final state for confirmation
    print("\nFinal State:")
    for k, v in final_state.items():
        print(f"{k}: {v}")
    wandb.finish()