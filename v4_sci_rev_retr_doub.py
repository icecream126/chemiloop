# LangGraph-based multi-agent molecular generation system with tool execution, reflection, and feedback

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Union
from langchain_deepseek import ChatDeepSeek
from langchain.schema import HumanMessage
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import prompts.v4.albuterol_similarity
from utils.args import parse_args, return_API_keys
from guacamol.utils.chemistry import canonicalize
from guacamol.common_scoring_functions import TanimotoScoringFunction
import utils.utils
from utils.metrics import get_isomers_c7h8n2o2_score, get_isomers_c9h10n2o2pf2cl_score, get_albuterol_similarity_score
import prompts.v4
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

task = args.task[0] if type(args.task)==list else args.task 
wandb.init(project=f"pmo_v4_{task}", name="pmo",config=vars(args))# , mode='disabled')
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_dir = f"./logs/{current_time}-{wandb.run.id}/"
os.makedirs(log_dir, exist_ok=True)

OPENAI_API_KEY = return_API_keys()["OPENAI_API_KEY"] 
DEEPSEEK_API_KEY = return_API_keys()["DEEPSEEK_API_KEY"]
LOG_PATH = f"{log_dir}log.txt"
SMILES_LOG_PATH = f"{log_dir}smiles.txt"
BEST_SMILES_LOG_PATH = f"{log_dir}best_smiles.txt"
WHOLE_SMILES_LOG_PATH = f"{log_dir}whole_smiles.txt"

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
json_double_checker_llm = OpenAI(
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

def whole_SMILES_log(msg: str):
    with open(WHOLE_SMILES_LOG_PATH, "a") as f:
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
    double_checker_think: Dict[str, str]
    generated_smiles: str
    task: List[str]
    score: float
    functional_groups: str
    json_output: bool
    topk_smiles: List[str]
    smiles_scores: List[Tuple[str, float]]
    in_double_checking_process: bool


# --------------------------
# Global Variables
# --------------------------

SMILES = ""
BEST_SMILES = ""
BEST_SCORE = 0.0
oracle_buffer = []
BEST_TOP_10_AUC_ALL = 0.0
BEST_TOP_10_AUC_NO_1 = 0.0
SMILES_HISTORY = set()

# Mapping for retrieval_node datasets
TASK_TO_DATASET_PATH = {
    "albuterol_similarity": "/home/khm/chemiloop/dataset/entire_top_5/albuterol_similarity_score.json",
    "isomers_c7h8n2o2": "/home/khm/chemiloop/dataset/entire_top_5/isomers_c7h8n2o2_score.json",
    "isomers_c9h10n2o2pf2cl": "/home/khm/chemiloop/dataset/entire_top_5/isomers_c9h10n2o2pf2cl_score.json",
}

# Mapping for scoring functions
TASK_TO_SCORING_FUNCTION = {
    "albuterol_similarity": get_albuterol_similarity_score,
    "isomers_c7h8n2o2": get_isomers_c7h8n2o2_score,
    "isomers_c9h10n2o2pf2cl": get_isomers_c9h10n2o2pf2cl_score,
}

# Mapping for scientist prompt functions
TASK_TO_SCIENTIST_PROMPT = {
    "albuterol_similarity": prompts.v4.albuterol_similarity.get_scientist_prompt,
    "isomers_c7h8n2o2": prompts.v4.isomers_c7h8n2o2.get_scientist_prompt,
}

# Mapping for scientist prompt with reviewer
TASK_TO_SCIENTIST_PROMPT_WITH_REVIEW = {
    "albuterol_similarity": prompts.v4.albuterol_similarity.get_scientist_prompt_with_review,
    "isomers_c7h8n2o2": prompts.v4.isomers_c7h8n2o2.get_scientist_prompt_with_review,
}

# Mapping for reviewer prompt
TASK_TO_REVIEWER_PROMPT = {
    "albuterol_similarity": prompts.v4.albuterol_similarity.get_reviewer_prompt,
    "isomers_c7h8n2o2": prompts.v4.isomers_c7h8n2o2.get_reviewer_prompt,
}

# Mapping for scientist prompt with double checker
TASK_TO_SCIENTIST_PROMPT_WITH_DOUBLE_CHECKER = {
    "albuterol_similarity": prompts.v4.albuterol_similarity.get_scientist_prompt_with_double_checker_review,
    "isomers_c7h8n2o2": prompts.v4.isomers_c7h8n2o2.get_scientist_prompt_with_double_checker_review,
}

TASK_TO_DOUBLE_CHECKER_PROMPT = {
    "albuterol_similarity": prompts.v4.albuterol_similarity.get_double_checker_prompt,
    "isomers_c7h8n2o2": prompts.v4.isomers_c7h8n2o2.get_double_checker_prompt,
}


def retrieval_node(state: GraphState) -> GraphState:
    
    # Load pre-computed top-k dataset by task
    # TODO: Extend this to entire train dataset
    # TODO: Add more tasks 
    # TODO: If not pre-computed, compute the top-k dataset
    task = state["task"][0] if type(state["task"])==list else state["task"]
    dataset_path = TASK_TO_DATASET_PATH.get(task, None)
    
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    topk_smiles = []
    for data in dataset:
        smiles = data["smiles"]
        score = data.get(f"{task}_score", 0)
        topk_smiles.append((smiles, score))

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
    task = state["task"][0] if type(state["task"])==list else state["task"] 

    if state["in_double_checking_process"] == True:       
        user_prompt = TASK_TO_SCIENTIST_PROMPT_WITH_DOUBLE_CHECKER[task](
            state["scientist_think"], state['generated_smiles'], state["double_checker_think"]
        )
            
    else:       
        if len(state["reviewer_think"]) == 0:
            # if "albuterol_similarity" in state["task"]:
            #     user_prompt = prompts.v4.get_v4_json_prompts.get_scientist_prompt_isomers_c7h8n2o2(TEXT_SMILES_HISTORY, topk_smiles)
            # elif "isomers_c7h8n2o2" in state["task"]:
            #     user_prompt = prompts.v4.get_v4_json_prompts.get_scientist_prompt_isomers_c7h8n2o2(TEXT_SMILES_HISTORY, topk_smiles)
            # else:
            #     raise NotImplementedError("Task not implemented")
            user_prompt = TASK_TO_SCIENTIST_PROMPT[task](
                TEXT_SMILES_HISTORY, topk_smiles
            )
        else:
            # if "albuterol_similarity" in state["task"]:
            #     user_prompt = prompts.v4.get_v4_json_prompts.get_scientist_prompt_with_review(state["prompt"], state['scientist_think'], state['reviewer_think'], state["generated_smiles"], state["score"], state["functional_groups"], TEXT_SMILES_HISTORY, topk_smiles)
            # elif "isomers_c7h8n2o2" in state["task"]:
            #     user_prompt = prompts.v4.get_v4_json_prompts.get_scientist_prompt_with_review_isomers_c7h8n2o2(state["prompt"], state['scientist_think'], state['reviewer_think'], state["generated_smiles"], state["score"], state["functional_groups"], TEXT_SMILES_HISTORY, topk_smiles)
            user_prompt = TASK_TO_SCIENTIST_PROMPT_WITH_REVIEW[task](
            state['scientist_think'], state['reviewer_think'],
            state["generated_smiles"], state["score"], state["functional_groups"],
                TEXT_SMILES_HISTORY, topk_smiles
            )

    system_prompt = f"You are a skilled chemist."  # (include full system instructions here)
    # user_prompt = get_v2_json_prompts.get_scientist_prompt(state["prompt"], SMILES_HISTORY)  # (include full user prompt here)
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    # state["scientist_message"].append({"role": "system", "content": system_prompt})
    # state["scientist_message"].append({"role": "user", "content": user_prompt})

    # print("Prompt to scientist node:", len(state["scientist_message"]))


    raw_response = utils.utils.safe_llm_call(prompt, json_scientist_llm, args.scientist_model_name, args.scientist_temperature)

    # Since the API guarantees a JSON object, you can access it directly:
    try:
        response = raw_response.choices[0].message.content  # Already a JSON string
        result = json.loads(response)  # Just in case it's not parsed automatically
        result = {k.lower(): v for k, v in result.items()}
        SMILES = result.get("smiles", "")
        scientist_think_dict ={
            'step1': result.get("step1", ""),
            'step2': result.get("step2", ""),
            'step3': result.get("step3", ""),
            'smiles': SMILES,
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
            'smiles': "",
        }
    
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

def double_checker_node(state: GraphState) -> GraphState:
    state["in_double_checking_process"] = False
    print("\n==== Double Checker Node ==")
    
    task = state["task"][0] if type(state["task"])==list else state["task"] 
    system_prompt = f"You are a meticulous double-checker LLM. Your task is to verify whether each step of the scientist’s reasoning is chemically valid and faithfully and logically reflected in the final SMILES string."  # (include full system instructions here)
    user_prompt = TASK_TO_DOUBLE_CHECKER_PROMPT[task](
                state["scientist_think"], state['generated_smiles']
    )
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    raw_response = utils.utils.safe_llm_call(prompt, json_double_checker_llm, args.double_checker_model_name, args.double_checker_temperature)
    
    try:
        response = raw_response.choices[0].message.content  # Already a JSON string
        result = json.loads(response)  # Just in case it's not parsed automatically
        result = {k.lower(): v for k, v in result.items()}
        # consistency = result.get("Consistency", "")
        double_checker_think_dict ={
            'step1': result.get("step1", ""),
            'step2': result.get("step2", ""),
            'step3': result.get("step3", ""),
            'consistency': result.get("consistency", ""),
        }
    except Exception as e:
        print("Error extracting SMILES:", e)
        SMILES = ""
        double_checker_think_dict = {
            'step1': "",
            'step2': "",
            'step3': "",
            'consistency': "",
        }
    
    print("Response from double checker node:", response)
    log("\n==== Double Checker Node ==")
    log("Prompt for double checker:")
    log(str(prompt))
    log("Response from double checker node:")
    log(str(response))
    return {
        **state, 
        "double_checker_think": double_checker_think_dict
    }


def reviewer_node(state: GraphState) -> GraphState:
    print("\n==== Reviewer Node ==")
    global SMILES, BEST_SCORE, BEST_SMILES, oracle_buffer, BEST_TOP_10_AUC_ALL, BEST_TOP_10_AUC_NO_1
    global json_reviewer_llm
    global reviewer_llm
    mol = Chem.MolFromSmiles(state["generated_smiles"])
    task = state["task"][0] if type(state["task"])==list else state["task"] 
    if mol is None:
        print("Invalid SMILES detected, retrying scientist node.")
        score = 0.0
        state["scientist_think"]["smiles"] += "(This SMILES is invalid, please retry.)"
    else:
        score = TASK_TO_SCORING_FUNCTION[task](state['generated_smiles'])
        

    state["smiles_scores"].append((state["generated_smiles"], score))
    SMILES_log(SMILES+" , "+str(score))
    whole_SMILES_log(SMILES+" , "+str(score))

    oracle_buffer.append((SMILES, score))

    # update best score
    if int(score) != 1 and score > BEST_SCORE:
        BEST_SCORE = score
        BEST_SMILES = state["generated_smiles"]
        BEST_SMILES_log(BEST_SMILES)
        # log the best score to wandb

    sorted_all = sorted(oracle_buffer, key=lambda x: x[1], reverse=True)
    top_10_all = sum(score for _, score in sorted_all[:10]) / 10
    
    scores_all = [s for _, s in oracle_buffer]

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
    log(f"Generated SMILES score: {score}")
    log(f"Top-10 avg (with 1.0): {top_10_all}")
    log(f"AUC all — Top-1: {auc_top1_all}, Top-10: {auc_top10_all}")
    log(f"Best AUC all — Top-1: {BEST_TOP_10_AUC_ALL}, Top-10: {BEST_TOP_10_AUC_NO_1}")


    # analyze albutero-relsted functional group of generated smiles
    if mol is not None:
        functional_groups = utils.utils.describe_albuterol_features(mol)
    else:
        functional_groups = "No functional groups because your SMILES is invalid. Please retry."
    
    system_prompt="You are a rigorous chemistry reviewer.\n"
    # if "albuterol_similarity" in state["task"]:
    #     user_prompt = prompts.v4.get_v4_json_prompts.get_reviewer_prompt(state["scientist_think"], score, functional_groups)
    # elif "isomers_c7h8n2o2" in state["task"]:
    #     user_prompt = prompts.v4.get_v4_json_prompts.get_reviewer_prompt_isomers_c7h8n2o2(state["scientist_think"], score, functional_groups)


    user_prompt = TASK_TO_REVIEWER_PROMPT[task](
        state["scientist_think"], score, functional_groups
    )
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    raw_response = utils.utils.safe_llm_call(prompt, json_reviewer_llm, args.reviewer_model_name, args.reviewer_temperature)

    # Since the API guarantees a JSON object, you can access it directly:
    try:
        response = raw_response.choices[0].message.content  # Already a JSON string
        result = json.loads(response)  # Just in case it's not parsed automatically
        result = {k.lower(): v for k, v in result.items()}
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
    # log(str(state["reviewer_message"][-2:]))
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

def route_after_double_checker(state: GraphState) -> str:
    if state["double_checker_think"]["consistency"].strip().lower() == "consistent":
        return "reviewer_node"
    state["in_double_checking_process"] = True
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
    builder.add_node("double_checker_node", double_checker_node)
    
    builder.set_entry_point("retrieval_node")  # if retrieval is the first step
    
    # After reviewer, decide whether to continue (reviewer → scientist OR END)
    builder.add_edge("retrieval_node", "scientist_node")
    builder.add_edge("scientist_node", "double_checker_node")
    builder.add_conditional_edges("double_checker_node", route_after_double_checker)
    builder.add_conditional_edges("reviewer_node", should_continue)

    # Compile graph
    graph = builder.compile()

    input_state: GraphState = {
        "iteration": 0,
        "max_iterations": args.max_iter,
        "scientist_think":{},
        "reviewer_think": {},
        "double_checker_think": {},
        "task": args.task,
        "score": 0.0,
        "functional_groups": "",
        "generated_smiles": "",
        "json_output": True,
        "topk_smiles": [],
        "smiles_scores":[],
        "scientist_message": [],
        "reviewer_message": [],
        "in_double_checking_process": False,
    }
    # Run the graph
    final_state = graph.invoke(input_state, {"recursion_limit": 9999})

    # Print final state for confirmation
    print("\nFinal State:")
    for k, v in final_state.items():
        print(f"{k}: {v}")
    wandb.finish()
