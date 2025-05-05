# LangGraph-based multi-agent molecular generation system with tool execution, reflection, and feedback

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Union
from langchain_deepseek import ChatDeepSeek
from langchain.schema import HumanMessage
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import prompts.task_prompts.albuterol_similarity
from utils.args import parse_args, return_API_keys
from guacamol.utils.chemistry import canonicalize
from guacamol.common_scoring_functions import TanimotoScoringFunction
import utils.utils
from utils.metrics import *
import prompts.task_prompts
import pandas as pd
from openai import OpenAI
from typing import List, Tuple
import utils.auc
from utils.task_dicts import get_task_to_condition_dict, get_task_to_dataset_path_dict, get_task_to_score_dict, get_task_to_scientist_prompt_dict, get_task_to_scientist_prompt_with_review_dict, get_task_to_reviewer_prompt_dict, get_task_to_scientist_prompt_with_double_checker_dict, get_task_to_double_checker_prompt_dict, get_task_to_functional_group_dict
import utils.tools as tools


import json
import csv
import os
import wandb
import re
import datetime


args = parse_args()

task = args.task[0] if type(args.task)==list else args.task 
wandb.init(project=f"pmo_v5", name=f"{task}",config=vars(args))#  , mode='disabled')
current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_dir = f"./logs/{current_time}-{wandb.run.id}/"
os.makedirs(log_dir, exist_ok=True)

OPENAI_API_KEY = return_API_keys()["OPENAI_API_KEY"] 
DEEPSEEK_API_KEY = return_API_keys()[f"DEEPSEEK_API_KEY_{args.api_num}"]
LOG_PATH = f"{log_dir}log.txt"
SMILES_LOG_PATH = f"{log_dir}smiles.txt"
BEST_SMILES_LOG_PATH = f"{log_dir}best_smiles.txt"
WHOLE_SMILES_LOG_PATH = f"{log_dir}whole_smiles.txt"

scientist_llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)
reviewer_llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)

json_tool_call_llm = OpenAI(
    api_key=DEEPSEEK_API_KEY,  # Replace with your actual API key
    base_url="https://api.deepseek.com"
)
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
    smiles_history: List[str]
    redundant_smiles: bool
    tools_to_use: List[str]


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

TASK_TO_CONDITION= get_task_to_condition_dict()
TASK_TO_DATASET_PATH = get_task_to_dataset_path_dict()
TASK_TO_SCORING_FUNCTION = get_task_to_score_dict()
TASK_TO_SCIENTIST_PROMPT = get_task_to_scientist_prompt_dict()
TASK_TO_SCIENTIST_PROMPT_WITH_REVIEW = get_task_to_scientist_prompt_with_review_dict()
TASK_TO_REVIEWER_PROMPT = get_task_to_reviewer_prompt_dict()
TASK_TO_SCIENTIST_PROMPT_WITH_DOUBLE_CHECKER = get_task_to_scientist_prompt_with_double_checker_dict()
TASK_TO_DOUBLE_CHECKER_PROMPT = get_task_to_double_checker_prompt_dict()
TASK_TO_FUNCTIONAL_GROUP = get_task_to_functional_group_dict()

def tool_call_node(state:GraphState) -> GraphState:
    tool_path = "/home/khm/chemiloop/dataset/filtered_rdkit_tool.json"

    with open(tool_path, "r") as tool_json:
        tool_specs = json.load(tool_json)

    task = state["task"][0] if type(state["task"])==list else state["task"] 
    system_prompt = """You are a professional AI chemistry assistant.
You are given a molecule design condition and a set of available chemical tools (RDKit functions). Your goal is to:

1. Analyze the molecule design condition which is the goal of the task.
2. Identify the key functional groups, structural features and miscellaneous molecular features (e.g., properties) related with the condition.
3. Choose **as many tools as necessary** from the toolset that are relevant to solving the task.
- The number of selected tools is **not limited**.
4. Explain why each tool is useful for this task."""

    user_prompt = f"""This is a molecule design condition of the {task} task:
{TASK_TO_CONDITION[task]}
                
Now output the tools to use by using the following JSON format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "tools_to_use": [
    {{"name": "function_name_1", "reason": "Why this function is useful."}},
    {{"name": "function_name_2", "reason": "Why this function is useful."}}
  ]
}}
```"""

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    raw_response = utils.utils.safe_llm_call(prompt = prompt, llm = json_tool_call_llm, llm_type = args.tool_call_model_name, llm_temperature = args.tool_call_temperature, max_retries=10, sleep_sec=2, tools=tool_specs)
    response = raw_response.choices[0].message.content
    try:
        tool_json = json.loads(response)
        tools_to_use = tool_json.get("tools_to_use", [])
    except Exception as e:
        print("Failed to parse JSON:", e)
        tools_to_use = []
    
    log(f"\n==== Tool call Node - {state['iteration']} ==")
    log("Prompt to tool call node:")
    log(str(prompt))
    log("\nResponse from tool call node:")
    log(str(response))

    return {
        **state,
        "tools_to_use": tools_to_use,
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
    user_prompt = ""
    if state["in_double_checking_process"] == True: 
        user_prompt = TASK_TO_SCIENTIST_PROMPT_WITH_DOUBLE_CHECKER[task](
            state["scientist_think"], state['generated_smiles'], state["double_checker_think"], TEXT_SMILES_HISTORY
        )
            
    else:       
        if len(state["reviewer_think"]) == 0:
            if state["redundant_smiles"]:
                user_prompt+=f"You generate the SMILES {state['generated_smiles']} again. DO NOT GENERATE THIS {state['generated_smiles']} SMILES AGAIN. \n"
                state["redundant_smiles"] = False
                
            user_prompt += TASK_TO_SCIENTIST_PROMPT[task](
                topk_smiles
            )
        else:
            if state["redundant_smiles"]:
                user_prompt+=f"You generate the SMILES {state['generated_smiles']} again. DO NOT GENERATE THIS {state['generated_smiles']} SMILES AGAIN. \n"
                state["redundant_smiles"] = False
                
            user_prompt += TASK_TO_SCIENTIST_PROMPT_WITH_REVIEW[task](
            state['scientist_think'], state['reviewer_think'],
            state["generated_smiles"], state["score"], state["functional_groups"],
                TEXT_SMILES_HISTORY, topk_smiles
            )

    system_prompt = f"You are a skilled chemist."  # (include full system instructions here)
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    raw_response = utils.utils.safe_llm_call(prompt, json_scientist_llm, args.scientist_model_name, args.scientist_temperature)
    response=""
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
        response=""
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

    return {
        **state,
        "scientist_think": scientist_think_dict,
        "generated_smiles": SMILES,
    }

def double_checker_node(state: GraphState) -> GraphState:
    global SMILES_HISTORY
    TEXT_SMILES_HISTORY = utils.utils.format_set_as_text(SMILES_HISTORY)
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
    response=""
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
        response=""
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
    
    finish = state["iteration"]+1>=state["max_iterations"]
    auc_top10_all, mol_buffer = utils.auc.compute_topk_auc(state["smiles_scores"], top_k=10, max_oracle_calls=1000, freq_log=1, buffer_max_idx = state["iteration"]+1, finish=finish)
    auc_top1_all, mol_buffer = utils.auc.compute_topk_auc(state["smiles_scores"], top_k=1, max_oracle_calls=1000, freq_log=1, buffer_max_idx = state["iteration"]+1, finish=finish)
    
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


    # get the functional_groups information via LLM chosen tools
    functional_groups = ["Functional groups and molecular features detected by RDKit tools:"]
    if mol is not None:
        for tool in state["tools_to_use"]:
            func_name = tool["name"].lower()
            reason = tool["reason"]
            
            try:
                func = getattr(tools, func_name)
                result = func(mol)

                report = f"""Tool: `{func_name}`
        Reason: {reason}
        Output: `{result}`

        """
                functional_groups.append(report)

            except Exception as e:
                report = f"""Tool: `{func_name}`
        Reason: {reason}
        Error: Could not execute `{func_name}` — {str(e)}

        """
                functional_groups.append(report)
        functional_groups = "\n".join(functional_groups)
        
    else:
        functional_groups = "No functional groups because your SMILES is invalid. Please retry."
    
    system_prompt="You are a rigorous chemistry reviewer.\n"

    user_prompt = TASK_TO_REVIEWER_PROMPT[task](
        state["scientist_think"], score, functional_groups
    )
    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    raw_response = utils.utils.safe_llm_call(prompt, json_reviewer_llm, args.reviewer_model_name, args.reviewer_temperature)
    response=""
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

def check_redundant_smiles(state: GraphState) -> str:
    if state["generated_smiles"] in state["smiles_history"]:
        state["redundant_smiles"] = True
        print("Redundant SMILES detected, retrying scientist node.")
        return "scientist_node"
    else:
        state["smiles_history"].append(state["generated_smiles"])
        return "double_checker_node"

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
    builder.add_node("tool_call_node", tool_call_node)
    builder.add_node("scientist_node", scientist_node)
    builder.add_node("reviewer_node", reviewer_node)
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("double_checker_node", double_checker_node)
    
    builder.set_entry_point("tool_call_node")  # if tool_call is the first step
    
    # After reviewer, decide whether to continue (reviewer → scientist OR END)
    builder.add_edge("tool_call_node", "retrieval_node")
    builder.add_edge("retrieval_node", "scientist_node")
    builder.add_conditional_edges("scientist_node", check_redundant_smiles)
    # builder.add_edge("scientist_node", "double_checker_node")
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
        "smiles_history": [],
        "redundant_smiles": False,
        "tools_to_use":[],
    }
    # Run the graph
    final_state = graph.invoke(input_state, {"recursion_limit": 9999})

    # Print final state for confirmation
    print("\nFinal State:")
    for k, v in final_state.items():
        print(f"{k}: {v}")
    wandb.finish()
