# LangGraph-based multi-agent molecular generation system with tool execution, reflection, and feedback

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Union, Tuple
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
from prompts.task_prompts.task_specific_prompts import get_task_specific_prompt, get_task_functional_groups

import json
import csv
import os
import wandb
import re
import datetime
from dataclasses import dataclass
from pathlib import Path
import logging

@dataclass
class Config:
    """Configuration class to hold all global settings"""
    task: str
    max_iter: int
    api_num: int
    tool_call_model_name: str
    tool_call_temperature: float
    scientist_model_name: str
    scientist_temperature: float
    reviewer_model_name: str
    reviewer_temperature: float
    double_checker_model_name: str
    double_checker_temperature: float
    
    def __post_init__(self):
        self.current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        self.log_dir = Path(f"./logs/{self.current_time}-{wandb.run.id}/")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # API Keys
        api_keys = return_API_keys()
        self.openai_api_key = api_keys["OPENAI_API_KEY"]
        self.deepseek_api_key = api_keys[f"DEEPSEEK_API_KEY_{self.api_num}"]
        
        # File paths
        self.log_path = self.log_dir / "log.txt"
        self.smiles_log_path = self.log_dir / "smiles.txt"
        self.best_smiles_log_path = self.log_dir / "best_smiles.txt"
        self.whole_smiles_log_path = self.log_dir / "whole_smiles.txt"
        self.smiles_history_log_path = self.log_dir / "smiles_history.txt"

class Logger:
    """Centralized logging class"""
    def __init__(self, config: Config):
        self.config = config
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.config.log_path),
                logging.StreamHandler()
            ]
        )
        
    def log(self, msg: str):
        """Log a message to both file and console"""
        logging.info(msg)
        
    def log_smiles(self, msg: str):
        """Log SMILES to SMILES log file"""
        with open(self.config.smiles_log_path, "a") as f:
            f.write(msg + "\n")
            
    def log_best_smiles(self, msg: str):
        """Log best SMILES to best SMILES log file"""
        with open(self.config.best_smiles_log_path, "a") as f:
            f.write(msg + "\n")
            
    def log_smiles_history(self, msg: str):
        """Log SMILES history"""
        with open(self.config.smiles_history_log_path, "a") as f:
            f.write(msg + "\n")
            
    def log_whole_smiles(self, msg: str):
        """Log all SMILES"""
        with open(self.config.whole_smiles_log_path, "a") as f:
            f.write(msg + "\n")

# Initialize configuration and logger
args = parse_args()
config = Config(
    task=args.task[0] if isinstance(args.task, list) else args.task,
    max_iter=args.max_iter,
    api_num=args.api_num,
    tool_call_model_name=args.tool_call_model_name,
    tool_call_temperature=args.tool_call_temperature,
    scientist_model_name=args.scientist_model_name,
    scientist_temperature=args.scientist_temperature,
    reviewer_model_name=args.reviewer_model_name,
    reviewer_temperature=args.reviewer_temperature,
    double_checker_model_name=args.double_checker_model_name,
    double_checker_temperature=args.double_checker_temperature
)

# Initialize wandb
wandb.init(project="pmo_v5", name=config.task, config=vars(args))

# Initialize logger
logger = Logger(config)

scientist_llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=config.deepseek_api_key)
reviewer_llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=config.deepseek_api_key)

json_tool_call_llm = OpenAI(
    api_key=config.deepseek_api_key,  # Replace with your actual API key
    base_url="https://api.deepseek.com"
)
json_scientist_llm = OpenAI(
    api_key=config.deepseek_api_key,  # Replace with your actual API key
    base_url="https://api.deepseek.com"
)
json_reviewer_llm = OpenAI(
    api_key=config.deepseek_api_key,  # Replace with your actual API key
    base_url="https://api.deepseek.com"
)
json_double_checker_llm = OpenAI(
    api_key=config.deepseek_api_key,  # Replace with your actual API key
    base_url="https://api.deepseek.com"
)

BEST_SCORE = 0.0

# --------------------------
# Logging helper
# --------------------------
def log(msg: str):
    logger.log(msg)

def SMILES_log(msg: str):
    logger.log_smiles(msg)

def BEST_SMILES_log(msg: str):
    logger.log_best_smiles(msg)

SMILES_HISTORY_LOG_PATH = f"{config.log_dir}smiles_history.txt"
def SMILES_HISTORY_log(msg: str):
    logger.log_smiles_history(msg)

def whole_SMILES_log(msg: str):
    logger.log_whole_smiles(msg)
# --------------------------
# Graph State
# --------------------------
class GraphState(TypedDict):
    """State class for the LangGraph workflow"""
    # Core state
    iteration: int
    max_iterations: int
    task: Union[str, List[str]]
    score: float
    generated_smiles: str
    
    # Thinking states
    scientist_think: Dict[str, str]
    reviewer_think: Dict[str, str]
    double_checker_think: Dict[str, str]
    
    # SMILES tracking
    smiles_history: List[str]
    topk_smiles: List[Tuple[str, float]]
    smiles_scores: List[Tuple[str, float]]
    
    # Analysis results
    functional_groups: str
    
    # Control flags
    json_output: bool
    in_double_checking_process: bool
    redundant_smiles: bool
    
    # Tool configuration
    tools_to_use: List[Dict[str, str]]

# Constants
TASK_TO_CONDITION = get_task_to_condition_dict()
TASK_TO_DATASET_PATH = get_task_to_dataset_path_dict()
TASK_TO_SCORING_FUNCTION = get_task_to_score_dict()
TASK_TO_SCIENTIST_PROMPT = get_task_to_scientist_prompt_dict()
TASK_TO_SCIENTIST_PROMPT_WITH_REVIEW = get_task_to_scientist_prompt_with_review_dict()
TASK_TO_REVIEWER_PROMPT = get_task_to_reviewer_prompt_dict()
TASK_TO_SCIENTIST_PROMPT_WITH_DOUBLE_CHECKER = get_task_to_scientist_prompt_with_double_checker_dict()
TASK_TO_DOUBLE_CHECKER_PROMPT = get_task_to_double_checker_prompt_dict()
TASK_TO_FUNCTIONAL_GROUP = get_task_to_functional_group_dict()

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

class MoleculeError(Exception):
    """Custom exception for molecule-related errors"""
    pass

def validate_smiles(smiles: str) -> bool:
    """Validate if a SMILES string is valid"""
    mol = Chem.MolFromSmiles(smiles)
    return mol is not None

def get_task_name(state: GraphState) -> str:
    """Get the task name from state, handling both string and list cases"""
    return state["task"][0] if isinstance(state["task"], list) else state["task"]

def format_smiles_history(smiles_history: set) -> str:
    """Format SMILES history for display"""
    if not smiles_history:
        return "Currently no history"
    return "\n".join(sorted(smiles_history))

def update_best_score(current_score: float, current_smiles: str, best_score: float, best_smiles: str) -> Tuple[float, str]:
    """Update best score and SMILES if current score is better"""
    if current_score > best_score:
        return current_score, current_smiles
    return best_score, best_smiles

def calculate_metrics(smiles_scores: List[Tuple[str, float]], iteration: int, max_iterations: int) -> Dict[str, float]:
    """Calculate various metrics for the current state"""
    finish = iteration + 1 >= max_iterations
    auc_top10_all, _ = utils.auc.compute_topk_auc(smiles_scores, top_k=10, max_oracle_calls=1000, freq_log=1, buffer_max_idx=iteration+1, finish=finish)
    auc_top1_all, _ = utils.auc.compute_topk_auc(smiles_scores, top_k=1, max_oracle_calls=1000, freq_log=1, buffer_max_idx=iteration+1, finish=finish)
    
    sorted_all = sorted(smiles_scores, key=lambda x: x[1], reverse=True)
    top_10_all = sum(score for _, score in sorted_all[:10]) / 10
    
    return {
        "top_10_avg_score_all": top_10_all,
        "auc_top1_all": auc_top1_all,
        "auc_top10_all": auc_top10_all
    }

def tool_call_node(state: GraphState) -> GraphState:
    """Node for selecting tools to use in the workflow"""
    tool_path = "/home/khm/chemiloop/dataset/filtered_rdkit_tool.json"
    
    try:
        with open(tool_path, "r") as tool_json:
            tool_specs = json.load(tool_json)
    except Exception as e:
        logger.log(f"Error loading tool specs: {e}")
        return {**state, "tools_to_use": []}

    task = get_task_name(state)
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

    try:
        raw_response = utils.utils.safe_llm_call(
            prompt=prompt,
            llm=json_tool_call_llm,
            llm_type=config.tool_call_model_name,
            llm_temperature=config.tool_call_temperature,
            max_retries=10,
            sleep_sec=2,
            tools=tool_specs
        )
        response = raw_response.choices[0].message.content
        tool_json = json.loads(response)
        tools_to_use = tool_json.get("tools_to_use", [])
    except Exception as e:
        logger.log(f"Error in tool call node: {e}")
        tools_to_use = []
    
    logger.log(f"\n==== Tool call Node - {state['iteration']} ==")
    logger.log("Prompt to tool call node:")
    logger.log(str(prompt))
    logger.log("\nResponse from tool call node:")
    logger.log(str(response))

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
    """Generate new SMILES based on current state"""
    global SMILES, SMILES_HISTORY
    
    logger.log("Scientist node is thinking...")
    
    # Format SMILES history and top-k SMILES
    smiles_history = format_smiles_history(state["smiles_history"])
    topk_smiles = utils.utils.format_topk_smiles(state["topk_smiles"])
    
    # Get task name and functional groups
    task_name = get_task_name(state)
    functional_groups = get_task_functional_groups(task_name)
    
    # Get appropriate prompt based on state
    if state.get("in_double_checking_process"):
        prompt = get_task_specific_prompt(
            task_name=task_name,
            prompt_type="scientist_with_double_checker",
            target_smiles=state["target_smiles"],
            canonical_smiles=state["canonical_smiles"],
            functional_groups=functional_groups,
            previous_thinking=state["scientist_think_dict"],
            previous_smiles=state["generated_smiles"],
            double_checker_feedback=state["double_checker_think_dict"],
            SMILES_HISTORY=smiles_history
        )
    elif state.get("reviewer_think_dict"):
        prompt = get_task_specific_prompt(
            task_name=task_name,
            prompt_type="scientist_with_review",
            target_smiles=state["target_smiles"],
            canonical_smiles=state["canonical_smiles"],
            functional_groups=functional_groups,
            scientist_think_dict=state["scientist_think_dict"],
            reviewer_feedback_dict=state["reviewer_think_dict"],
            previous_smiles=state["generated_smiles"],
            score=state["score"],
            functional_groups_detected=state["functional_groups"],
            SMILES_HISTORY=smiles_history,
            topk_smiles=topk_smiles
        )
    else:
        prompt = get_task_specific_prompt(
            task_name=task_name,
            prompt_type="scientist",
            target_smiles=state["target_smiles"],
            canonical_smiles=state["canonical_smiles"],
            functional_groups=functional_groups,
            topk_smiles=topk_smiles
        )
    
    try:
        response = llm.invoke(prompt)
        response_dict = json.loads(response)
        
        if "SMILES" not in response_dict:
            raise MoleculeError("No SMILES generated in response")
        
        state["scientist_think_dict"] = response_dict
        state["generated_smiles"] = response_dict["SMILES"]
        state["smiles_history"] = utils.utils.add_with_limit(state["smiles_history"], response_dict["SMILES"])
        
        return state
    except Exception as e:
        logger.log(f"Error in scientist node: {str(e)}")
        raise

def double_checker_node(state: GraphState) -> GraphState:
    """Verify scientist's reasoning and SMILES generation"""
    logger.log("Double checker node is thinking...")
    
    # Get task name
    task_name = get_task_name(state)
    
    # Get double checker prompt
    prompt = get_task_specific_prompt(
        task_name=task_name,
        prompt_type="double_checker",
        target_smiles=state["target_smiles"],
        canonical_smiles=state["canonical_smiles"],
        functional_groups=get_task_functional_groups(task_name),
        thinking=state["scientist_think_dict"],
        improved_smiles=state["generated_smiles"]
    )
    
    try:
        response = llm.invoke(prompt)
        state["double_checker_think_dict"] = json.loads(response)
        return state
    except Exception as e:
        logger.log(f"Error in double checker node: {str(e)}")
        state["double_checker_think_dict"] = {
            "step1": "",
            "step2": "",
            "step3": "",
            "consistency": "Inconsistent"
        }
        return state

def reviewer_node(state: GraphState) -> GraphState:
    """Review and score generated SMILES"""
    logger.log("Reviewer node is thinking...")
    
    # Validate SMILES
    if not validate_smiles(state["generated_smiles"]):
        logger.log("Invalid SMILES generated")
        state["score"] = 0.0
        return state
    
    # Get task name and scoring function
    task_name = get_task_name(state)
    scoring_function = TASK_TO_SCORING_FUNCTION[task_name]
    
    # Calculate score
    state["score"] = scoring_function(state["generated_smiles"])
    
    # Update best score if needed
    update_best_score(state)
    
    # Calculate metrics
    metrics = calculate_metrics(state["smiles_scores"], state["iteration"], state["max_iterations"])
    wandb.log(metrics)
    
    # Analyze functional groups
    try:
        functional_groups = tool_call_node(state)
        state["functional_groups"] = functional_groups
    except Exception as e:
        logger.log(f"Error in tool call: {str(e)}")
        state["functional_groups"] = "Error in functional group analysis"
    
    # Get reviewer prompt
    prompt = get_task_specific_prompt(
        task_name=task_name,
        prompt_type="reviewer",
        target_smiles=state["target_smiles"],
        canonical_smiles=state["canonical_smiles"],
        functional_groups=get_task_functional_groups(task_name),
        scientist_think_dict=state["scientist_think_dict"],
        score=state["score"],
        functional_groups_detected=state["functional_groups"]
    )
    
    try:
        response = llm.invoke(prompt)
        state["reviewer_think_dict"] = json.loads(response)
        return state
    except Exception as e:
        logger.log(f"Error in reviewer node: {str(e)}")
        raise

def should_continue(state: GraphState) -> str:
    if state["iteration"] >= state["max_iterations"]:
        return END
    return "scientist_node"

# def check_redundant_smiles(state: GraphState) -> str:
#     if state["generated_smiles"] in state["smiles_history"]:
#         state["redundant_smiles"] = True
#         print("Redundant SMILES detected, retrying scientist node.")
#         return "scientist_node"
#     else:
#         state["smiles_history"].append(state["generated_smiles"])
#         return "double_checker_node"

def route_after_double_checker(state: GraphState) -> str:
    if state["double_checker_think"]["consistency"].strip().lower() == "consistent":
        return "reviewer_node"
    state["in_double_checking_process"] = True
    return "scientist_node"

def create_graph() -> StateGraph:
    """Create and configure the LangGraph workflow"""
    builder = StateGraph(GraphState)
    
    # Add nodes
    builder.add_node("tool_call_node", tool_call_node)
    builder.add_node("scientist_node", scientist_node)
    builder.add_node("reviewer_node", reviewer_node)
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("double_checker_node", double_checker_node)
    
    # Set entry point
    builder.set_entry_point("tool_call_node")
    
    # Add edges
    builder.add_edge("tool_call_node", "retrieval_node")
    builder.add_edge("retrieval_node", "scientist_node")
    # builder.add_conditional_edges("scientist_node", check_redundant_smiles)
    builder.add_edge("scientist_node", "double_checker_node")
    builder.add_conditional_edges("double_checker_node", route_after_double_checker)
    builder.add_conditional_edges("reviewer_node", should_continue)
    
    return builder.compile()

def create_initial_state() -> GraphState:
    """Create the initial state for the graph"""
    return {
        "iteration": 0,
        "max_iterations": config.max_iter,
        "scientist_think": {},
        "reviewer_think": {},
        "double_checker_think": {},
        "task": config.task,
        "score": 0.0,
        "functional_groups": "",
        "generated_smiles": "",
        "json_output": True,
        "topk_smiles": [],
        "smiles_scores": [],
        "scientist_message": [],
        "reviewer_message": [],
        "in_double_checking_process": False,
        "smiles_history": [],
        "redundant_smiles": False,
        "tools_to_use": [],
    }

if __name__ == "__main__":
    try:
        # Create and run the graph
        graph = create_graph()
        initial_state = create_initial_state()
        final_state = graph.invoke(initial_state, {"recursion_limit": 9999})
        
        # Print final state
        logger.log("\nFinal State:")
        for k, v in final_state.items():
            logger.log(f"{k}: {v}")
            
    except Exception as e:
        logger.log(f"Error in main execution: {e}")
    finally:
        wandb.finish()
