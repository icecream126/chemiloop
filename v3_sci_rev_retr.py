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
import prompts.v3.get_v3_json_prompts
import pandas as pd
from openai import OpenAI
from typing import List, Tuple




import json
import csv
import os
import wandb
import re
import datetime


args = parse_args()


wandb.init(project=f"pmo_v3_{args.task[0]}", name="pmo",config=vars(args))# , mode='disabled')
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
    task: List[str]
    score: float
    functional_groups: str
    json_output: bool
    topk_smiles: List[str]
    smiles_scores: List[Tuple[str, float]]
    scientist_message: List[Dict[str, str]]
    reviewer_message: List[Dict[str, str]]

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

# def retrieval_node(state: GraphState) -> GraphState:
#     global SMILES
#     global BEST_SCORE
#     global BEST_SMILES
#     global oracle_buffer
#     global BEST_TOP_10_AUC_ALL
#     global BEST_TOP_10_AUC_NO_1

#     # dataset: "/home/khm/chemiloop/dataset/guacamol.json"
#     # dataset has SMILES, albuterol_similarity score, and isomers_c7h8n2o2 score
#     # Take the user prompt state["prompt"]
#     # Dataset 자체에 score들을 다 저장해놓을까..
#     # If user prompt requires "albuterol_similarity", retrieve the SMILES based on top-K albuterol_similarity score
#     # If user prompt requires "isomers_c7h8n2o2", retrieve the SMILES based on top-K isomers_c7h8n2o2 score
#     # Return the retrieved SMILES and its score according to the task (e.g., such as albuterol similarity score)

def retrieval_node(state: GraphState) -> GraphState:
    
    # Load pre-computed top-k dataset by task
    # TODO: Extend this to entire train dataset
    # TODO: Add more tasks 
    # TODO: If not pre-computed, compute the top-k dataset
    if "albuterol_similarity" in state["task"]:
        dataset_path = "/home/khm/chemiloop/dataset/entire_top_5/albuterol_similarity_score.json"
    elif "isomers_c7h8n2o2" in state["task"]:
        dataset_path = "/home/khm/chemiloop/dataset/entire_top_5/isomer_c7h8n2o2_score.json"
    elif "isomers_c9h10n2o2pf2cl" in state["task"]:
        dataset_path = "/home/khm/chemiloop/dataset/entire_top_5/isomer_c9h10n2o2pf2cl_score.json"
    else:
        raise NotImplementedError("Unsupported task in retrieval_node.")
    
    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    topk_smiles = []
    for data in dataset:
        smiles = data["smiles"]
        albuterol_similarity_score = data.get("albuterol_similarity_score", 0)
        isomer_c7h8n2o2_score = data.get("isomer_c7h8n2o2_score", 0)
        isomer_c9h10n2o2pf2cl_score = data.get("isomer_c9h10n2o2pf2cl_score", 0)

        if "albuterol_similarity" in state["task"]:
            topk_smiles.append((smiles, albuterol_similarity_score))
        elif "isomers_c7h8n2o2" in state["task"]:
            topk_smiles.append((smiles, isomer_c7h8n2o2_score))
        elif "isomers_c9h10n2o2pf2cl" in state["task"]:
            topk_smiles.append((smiles, isomer_c9h10n2o2pf2cl_score))
    
    # Use the top SMILES as the initial suggestion
    # retrieved_smiles = top_df.iloc[0]["smiles"]
    # retrieved_score = top_df.iloc[0][score_column]

    print(f"[Retrieval Node] Retrieved SMILES: ", str(topk_smiles))

    return {
        **state,
        "topk_smiles": topk_smiles,
    }

def scientist_node(state: GraphState) -> GraphState:
    global SMILES
    global json_scientist_llm
    global scientist_llm
    global SMILES_HISTORY
    print("\n==== Scientist Node ==")

    TEXT_SMILES_HISTORY = utils.utils.format_set_as_text(SMILES_HISTORY)
    topk_smiles = utils.utils.format_topk_smiles(state["topk_smiles"])
    if state["reviewer_think"] == "":
        if "albuterol_similarity" in state["task"]:
            user_prompt = prompts.v3.get_v3_json_prompts.get_scientist_prompt_isomers_c7h8no2(TEXT_SMILES_HISTORY, topk_smiles)
        elif "isomers_c7h8n2o2" in state["task"]:
            user_prompt = prompts.v3.get_v3_json_prompts.get_scientist_prompt_isomers_c7h8no2(TEXT_SMILES_HISTORY, topk_smiles)
        else:
            raise NotImplementedError("Task not implemented")
    else:
        if "albuterol_similarity" in state["task"]:
            user_prompt = prompts.v3.get_v3_json_prompts.get_scientist_prompt_with_review(state["prompt"], state['scientist_think'], state['reviewer_think'], state["generated_smiles"], state["score"], state["functional_groups"], TEXT_SMILES_HISTORY, topk_smiles)
        elif "isomers_c7h8n2o2" in state["task"]:
            user_prompt = prompts.v3.get_v3_json_prompts.get_scientist_prompt_with_review_isomers_c7h8n2o2(state["prompt"], state['scientist_think'], state['reviewer_think'], state["generated_smiles"], state["score"], state["functional_groups"], TEXT_SMILES_HISTORY, topk_smiles)
    system_prompt = f"You are a skilled chemist."  # (include full system instructions here)
    # user_prompt = get_v2_json_prompts.get_scientist_prompt(state["prompt"], SMILES_HISTORY)  # (include full user prompt here)
    # prompt = [
        # {"role": "system", "content": system_prompt},
        # {"role": "user", "content": user_prompt}
    # ]
    state["scientist_message"].append({"role": "system", "content": system_prompt})
    state["scientist_message"].append({"role": "user", "content": user_prompt})

    print("Prompt to scientist node:", len(state["scientist_message"]))
    

    # Call the model with enforced JSON output
    raw_response = json_scientist_llm.chat.completions.create(
        model="deepseek-chat",
        messages=state["scientist_message"],
        response_format={"type": "json_object"},
        temperature=args.scientist_temperature,
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
    
    utils.utils.add_with_limit(SMILES_HISTORY, SMILES)
    SMILES_HISTORY_log(str(SMILES_HISTORY))
    # Printing and logging
    print("Response from scientist node:", response)

    log(f"\n==== Scientist Node - {state['iteration']} ==")
    log("Prompt to scientist node:")
    log(str(state["scientist_message"][-2:]))
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
    print("\n==== Reviewer Node ==")
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
        if "molecular weight" in state["task"]:
            score = Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(state["generated_smiles"]))
            # TODO: modify score by diff of target property and pred_value
        elif "albuterol_similarity" in state["task"]:
            score = get_albuterol_similarity_score(state["generated_smiles"])
        elif "isomers_c7h8n2o2" in state["task"]:
            score = get_isomer_c7h8n2o2_score(state["generated_smiles"])
        else:
            raise NotImplementedError("Target property not implemented")

    state["smiles_scores"].append((state["generated_smiles"], score))
    SMILES_log(SMILES+" , "+str(score))

    # oracle_buffer.append((SMILES, score))

    # update best score
    if int(score) != 1 and score > BEST_SCORE:
        BEST_SCORE = score
        BEST_SMILES = state["generated_smiles"]
        BEST_SMILES_log(BEST_SMILES)
        # log the best score to wandb

    sorted_all = sorted(oracle_buffer, key=lambda x: x[1], reverse=True)
    top_10_all = sum(score for _, score in sorted_all[:10]) / 10
    
    scores_all = [s for _, s in oracle_buffer]

    # auc_top1_all = utils.utils.compute_auc_topk_online_torch(scores_all, k=1)
    # auc_top10_all = utils.utils.compute_auc_topk_online_torch(scores_all, k=10)
    import utils.auc
    auc_top10_all = utils.auc.compute_topk_auc(state["smiles_scores"], top_k=10, max_oracle_calls=1000, freq_log=1)[0]
    auc_top1_all = utils.auc.compute_topk_auc(state["smiles_scores"], top_k=1, max_oracle_calls=1000, freq_log=1)[0]

    if auc_top10_all > BEST_TOP_10_AUC_ALL:
        BEST_TOP_10_AUC_ALL = auc_top10_all

    wandb.log({
        "top_10_avg_score_all": top_10_all,
        "auc_top1_all": auc_top1_all,
        "auc_top10_all": auc_top10_all,
        "score": score,
        "best_score": BEST_SCORE,
        "best_smiles": BEST_SMILES,
    },step=state["iteration"])
    log(f"Top-10 avg (with 1.0): {top_10_all}")
    log(f"AUC all — Top-1: {auc_top1_all}, Top-10: {auc_top10_all}")
    log(f"Best AUC all — Top-1: {BEST_TOP_10_AUC_ALL}, Top-10: {BEST_TOP_10_AUC_NO_1}")


    # analyze albutero-relsted functional group of generated smiles
    if mol is not None:
        functional_groups = utils.utils.describe_albuterol_features(mol)
    else:
        functional_groups = "No functional groups because your SMILES is invalid. Please retry."
    
    system_prompt="You are a rigorous chemistry reviewer.\n"
    if "albuterol_similarity" in state["task"]:
        user_prompt = prompts.v3.get_v3_json_prompts.get_reviewer_prompt(state["scientist_think"], score, functional_groups)
    elif "isomers_c7h8n2o2" in state["task"]:
        user_prompt = prompts.v3.get_v3_json_prompts.get_reviewer_prompt_isomers_c7h8n2o2(state["scientist_think"], score, functional_groups)
    # prompt = [
    #     {"role": "system", "content": system_prompt},
    #     {"role": "user", "content": user_prompt}
    # ]
    state["reviewer_message"].append({"role": "system", "content": system_prompt})
    state["reviewer_message"].append({"role": "user", "content": user_prompt})

    # Call the model with enforced JSON output
    raw_response = json_reviewer_llm.chat.completions.create(
        model="deepseek-chat",
        messages=state["reviewer_message"],
        response_format={"type": "json_object"},
        temperature=1.0,
    )

    # Since the API guarantees a JSON object, you can access it directly:
    try:
        response = raw_response.choices[0].message.content  # Already a JSON string
        result = json.loads(response)  # Just in case it's not parsed automatically
        reviewer_think_dict ={
            'step1': result.get("step1", ""),
            'step2': result.get("step2", ""),
            'step3': result.get("step3", ""),
        }
    except Exception as e:
        print("Error extracting SMILES:", e)
        reviewer_think_dict = {
            'step1': "",
            'step2': "",
            'step3': "",
        }


    print("Response from reviewer node:", response)
    log("\n==== Reviewer Node ==")
    log("Prompt to reviewer node:")
    log(str(state["reviewer_message"][-2:]))
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
    builder.add_node("retrieval_node", retrieval_node)
    
    builder.set_entry_point("retrieval_node")  # if retrieval is the first step
    
    # After reviewer, decide whether to continue (reviewer → scientist OR END)
    builder.add_edge("retrieval_node", "scientist_node")
    builder.add_edge("scientist_node", "reviewer_node")
    # builder.add_conditional_edges("scientist_node", is_scientist_SMILES_valid)
    builder.add_conditional_edges("reviewer_node", should_continue)

    # Compile graph
    graph = builder.compile()

    user_prompt = prompts.v3.get_v3_json_prompts.get_user_prompt(args.task)

    input_state: GraphState = {
        "prompt": user_prompt,
        "iteration": 0,
        "max_iterations": args.max_iter,
        "scientist_think": "",
        "reviewer_think": "",
        "task": args.task,
        "score": 0.0,
        "functional_groups": "",
        "generated_smiles": "",
        "json_output": True,
        "topk_smiles": [],
        "smiles_scores":[],
        "scientist_message": [],
        "reviewer_message": [],
    }
    # Run the graph
    final_state = graph.invoke(input_state, {"recursion_limit": 9999})

    # Print final state for confirmation
    print("\nFinal State:")
    for k, v in final_state.items():
        print(f"{k}: {v}")
    wandb.finish()
