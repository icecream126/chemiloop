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




import json
import csv
import os
import wandb
import re
import datetime


args = parse_args()

task = args.task[0] if type(args.task)==list else args.task 
wandb.init(project=f"pmo_v4_no_redundancy", name=f"{task}",config=vars(args))# , mode='disabled')
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
    "albuterol_similarity": "/home/khm/chemiloop/dataset/10k_top_5/albuterol_similarity_score.json",
    "isomers_c7h8n2o2": "/home/khm/chemiloop/dataset/10k_top_5/isomers_c7h8n2o2_score.json",
    "isomers_c9h10n2o2pf2cl": "/home/khm/chemiloop/dataset/10k_top_5/isomers_c9h10n2o2pf2cl_score.json",
    "amlodipine_mpo": "/home/khm/chemiloop/dataset/10k_top_5/amlodipine_mpo_score.json",
    "celecoxib_rediscovery": "/home/khm/chemiloop/dataset/10k_top_5/celecoxib_rediscovery_score.json",
    "deco_hop": "/home/khm/chemiloop/dataset/10k_top_5/deco_hop_score.json",
    "drd2": "/home/khm/chemiloop/dataset/10k_top_5/drd2_score.json",
    "fexofenadine_mpo": "/home/khm/chemiloop/dataset/10k_top_5/fexofenadine_mpo_score.json",
    "gsk3b": "/home/khm/chemiloop/dataset/10k_top_5/gsk3b_score.json",
    "jnk3": "/home/khm/chemiloop/dataset/10k_top_5/jnk3_score.json",
    "median1": "/home/khm/chemiloop/dataset/10k_top_5/median1_score.json",
    "median2": "/home/khm/chemiloop/dataset/10k_top_5/median2_score.json",
    "mestranol_similarity": "/home/khm/chemiloop/dataset/10k_top_5/mestranol_similarity_score.json",
    "osimertinib_mpo": "/home/khm/chemiloop/dataset/10k_top_5/osimertinib_mpo_score.json",
    "perindopril_mpo": "/home/khm/chemiloop/dataset/10k_top_5/perindopril_mpo_score.json",
    "qed": "/home/khm/chemiloop/dataset/10k_top_5/qed_score.json",
    "ranolazine_mpo": "/home/khm/chemiloop/dataset/10k_top_5/ranolazine_mpo_score.json",
    "scaffold_hop": "/home/khm/chemiloop/dataset/10k_top_5/scaffold_hop_score.json",
    "sitagliptin_mpo": "/home/khm/chemiloop/dataset/10k_top_5/sitagliptin_mpo_score.json",
    "thiothixene_rediscovery": "/home/khm/chemiloop/dataset/10k_top_5/thiothixene_rediscovery_score.json",
    "troglitazon_rediscovery": "/home/khm/chemiloop/dataset/10k_top_5/troglitazon_rediscovery_score.json",
    "valsartan_smarts": "/home/khm/chemiloop/dataset/10k_top_5/valsartan_smarts_score.json",
    "zaleplon_mpo": "/home/khm/chemiloop/dataset/10k_top_5/zaleplon_mpo_score.json",

}

# Mapping for scoring functions
TASK_TO_SCORING_FUNCTION = {
    "albuterol_similarity": get_albuterol_similarity_score,
    "isomers_c7h8n2o2": get_isomers_c7h8n2o2_score,
    "isomers_c9h10n2o2pf2cl": get_isomers_c9h10n2o2pf2cl_score,
    "amlodipine_mpo": get_amlodipine_mpo_score,
    "celecoxib_rediscovery": get_celecoxib_rediscovery_score,
    "deco_hop": get_deco_hop_score,
    "drd2": get_drd2_score,
    "fexofenadine_mpo": get_fexofenadine_mpo_score,
    "gsk3b": get_gsk3b_score,
    "jnk3": get_jnk3_score,
    "median1": get_median1_score,
    "median2": get_median2_score,
    "mestranol_similarity": get_mestranol_similarity_score,
    "osimertinib_mpo": get_osimertinib_mpo_score,
    "perindopril_mpo": get_perindopril_mpo_score,
    "qed":get_qed_score,
    "ranolazine_mpo": get_ranolazine_mpo_score,
    "scaffold_hop": get_scaffold_hop_score,
    "sitagliptin_mpo": get_sitagliptin_mpo_score,
    "thiothixene_rediscovery": get_thiothixene_rediscovery_score,
    "troglitazon_rediscovery": get_troglitazon_rediscovery_score,
    "valsartan_smarts": get_valsartan_smarts_score,
    "zaleplon_mpo": get_zaleplon_mpo_score,
}

# Mapping for scientist prompt functions
TASK_TO_SCIENTIST_PROMPT = {
    "albuterol_similarity": prompts.task_prompts.albuterol_similarity.get_scientist_prompt,
    "isomers_c7h8n2o2": prompts.task_prompts.isomers_c7h8n2o2.get_scientist_prompt,
    "isomers_c9h10n2o2pf2cl": prompts.task_prompts.isomers_c9h10n2o2pf2cl.get_scientist_prompt,
    "amlodipine_mpo": prompts.task_prompts.amlodipine_mpo.get_scientist_prompt,
    "celecoxib_rediscovery": prompts.task_prompts.celecoxib_rediscovery.get_scientist_prompt,
    "deco_hop": prompts.task_prompts.deco_hop.get_scientist_prompt,
    "drd2": prompts.task_prompts.drd2.get_scientist_prompt,
    "fexofenadine_mpo": prompts.task_prompts.fexofenadine_mpo.get_scientist_prompt,
    "gsk3b": prompts.task_prompts.gsk3b.get_scientist_prompt,
    "jnk3": prompts.task_prompts.jnk3.get_scientist_prompt,
    "median1": prompts.task_prompts.median1.get_scientist_prompt,
    "median2": prompts.task_prompts.median2.get_scientist_prompt,
    "mestranol_similarity": prompts.task_prompts.mestranol_similarity.get_scientist_prompt,
    "osimertinib_mpo": prompts.task_prompts.osimertinib_mpo.get_scientist_prompt,
    "perindopril_mpo": prompts.task_prompts.perindopril_mpo.get_scientist_prompt,
    "qed": prompts.task_prompts.qed.get_scientist_prompt,
    "ranolazine_mpo": prompts.task_prompts.ranolazine_mpo.get_scientist_prompt,
    "scaffold_hop": prompts.task_prompts.scaffold_hop.get_scientist_prompt,
    "sitagliptin_mpo": prompts.task_prompts.sitagliptin_mpo.get_scientist_prompt,
    "thiothixene_rediscovery": prompts.task_prompts.thiothixene_rediscovery.get_scientist_prompt,
    "troglitazon_rediscovery": prompts.task_prompts.troglitazon_rediscovery.get_scientist_prompt,
    "valsartan_smarts": prompts.task_prompts.valsartan_smarts.get_scientist_prompt,
    "zaleplon_mpo": prompts.task_prompts.zaleplon_mpo.get_scientist_prompt,
}

# Mapping for scientist prompt with reviewer
TASK_TO_SCIENTIST_PROMPT_WITH_REVIEW = {
    "albuterol_similarity": prompts.task_prompts.albuterol_similarity.get_scientist_prompt_with_review,
    "isomers_c7h8n2o2": prompts.task_prompts.isomers_c7h8n2o2.get_scientist_prompt_with_review,
    "isomers_c9h10n2o2pf2cl": prompts.task_prompts.isomers_c9h10n2o2pf2cl.get_scientist_prompt_with_review,
    "amlodipine_mpo": prompts.task_prompts.amlodipine_mpo.get_scientist_prompt_with_review,
    "celecoxib_rediscovery": prompts.task_prompts.celecoxib_rediscovery.get_scientist_prompt_with_review,
    "deco_hop": prompts.task_prompts.deco_hop.get_scientist_prompt_with_review,
    "drd2": prompts.task_prompts.drd2.get_scientist_prompt_with_review,
    "fexofenadine_mpo": prompts.task_prompts.fexofenadine_mpo.get_scientist_prompt_with_review,
    "gsk3b": prompts.task_prompts.gsk3b.get_scientist_prompt_with_review,
    "jnk3": prompts.task_prompts.jnk3.get_scientist_prompt_with_review,
    "median1": prompts.task_prompts.median1.get_scientist_prompt_with_review,
    "median2": prompts.task_prompts.median2.get_scientist_prompt_with_review,
    "mestranol_similarity": prompts.task_prompts.mestranol_similarity.get_scientist_prompt_with_review,
    "osimertinib_mpo": prompts.task_prompts.osimertinib_mpo.get_scientist_prompt_with_review,
    "perindopril_mpo": prompts.task_prompts.perindopril_mpo.get_scientist_prompt_with_review,
    "qed": prompts.task_prompts.qed.get_scientist_prompt_with_review,
    "ranolazine_mpo": prompts.task_prompts.ranolazine_mpo.get_scientist_prompt_with_review,
    "scaffold_hop": prompts.task_prompts.scaffold_hop.get_scientist_prompt_with_review,
    "sitagliptin_mpo": prompts.task_prompts.sitagliptin_mpo.get_scientist_prompt_with_review,
    "thiothixene_rediscovery": prompts.task_prompts.thiothixene_rediscovery.get_scientist_prompt_with_review,
    "troglitazon_rediscovery": prompts.task_prompts.troglitazon_rediscovery.get_scientist_prompt_with_review,
    "valsartan_smarts": prompts.task_prompts.valsartan_smarts.get_scientist_prompt_with_review,
    "zaleplon_mpo": prompts.task_prompts.zaleplon_mpo.get_scientist_prompt_with_review,

}


TASK_TO_REVIEWER_PROMPT = {
    "albuterol_similarity": prompts.task_prompts.albuterol_similarity.get_reviewer_prompt,
    "isomers_c7h8n2o2": prompts.task_prompts.isomers_c7h8n2o2.get_reviewer_prompt,
    "isomers_c9h10n2o2pf2cl": prompts.task_prompts.isomers_c9h10n2o2pf2cl.get_reviewer_prompt,
    "amlodipine_mpo": prompts.task_prompts.amlodipine_mpo.get_reviewer_prompt,
    "celecoxib_rediscovery": prompts.task_prompts.celecoxib_rediscovery.get_reviewer_prompt,
    "deco_hop": prompts.task_prompts.deco_hop.get_reviewer_prompt,
    "drd2": prompts.task_prompts.drd2.get_reviewer_prompt,
    "fexofenadine_mpo": prompts.task_prompts.fexofenadine_mpo.get_reviewer_prompt,
    "gsk3b": prompts.task_prompts.gsk3b.get_reviewer_prompt,
    "jnk3": prompts.task_prompts.jnk3.get_reviewer_prompt,
    "median1": prompts.task_prompts.median1.get_reviewer_prompt,
    "median2": prompts.task_prompts.median2.get_reviewer_prompt,
    "mestranol_similarity": prompts.task_prompts.mestranol_similarity.get_reviewer_prompt,
    "osimertinib_mpo": prompts.task_prompts.osimertinib_mpo.get_reviewer_prompt,
    "perindopril_mpo": prompts.task_prompts.perindopril_mpo.get_reviewer_prompt,
    "qed": prompts.task_prompts.qed.get_reviewer_prompt,
    "ranolazine_mpo": prompts.task_prompts.ranolazine_mpo.get_reviewer_prompt,
    "scaffold_hop": prompts.task_prompts.scaffold_hop.get_reviewer_prompt,
    "sitagliptin_mpo": prompts.task_prompts.sitagliptin_mpo.get_reviewer_prompt,
    "thiothixene_rediscovery": prompts.task_prompts.thiothixene_rediscovery.get_reviewer_prompt,
    "troglitazon_rediscovery": prompts.task_prompts.troglitazon_rediscovery.get_reviewer_prompt,
    "valsartan_smarts": prompts.task_prompts.valsartan_smarts.get_reviewer_prompt,
    "zaleplon_mpo": prompts.task_prompts.zaleplon_mpo.get_reviewer_prompt,
}

# Mapping for scientist prompt with double checker
TASK_TO_SCIENTIST_PROMPT_WITH_DOUBLE_CHECKER = {
    "albuterol_similarity": prompts.task_prompts.albuterol_similarity.get_scientist_prompt_with_double_checker_review,
    "isomers_c7h8n2o2": prompts.task_prompts.isomers_c7h8n2o2.get_scientist_prompt_with_double_checker_review,
    "isomers_c9h10n2o2pf2cl": prompts.task_prompts.isomers_c9h10n2o2pf2cl.get_scientist_prompt_with_double_checker_review,
    "amlodipine_mpo": prompts.task_prompts.amlodipine_mpo.get_scientist_prompt_with_double_checker_review,
    "celecoxib_rediscovery": prompts.task_prompts.celecoxib_rediscovery.get_scientist_prompt_with_double_checker_review,
    "deco_hop": prompts.task_prompts.deco_hop.get_scientist_prompt_with_double_checker_review,
    "drd2": prompts.task_prompts.drd2.get_scientist_prompt_with_double_checker_review,
    "fexofenadine_mpo": prompts.task_prompts.fexofenadine_mpo.get_scientist_prompt_with_double_checker_review,
    "gsk3b": prompts.task_prompts.gsk3b.get_scientist_prompt_with_double_checker_review,
    "jnk3": prompts.task_prompts.jnk3.get_scientist_prompt_with_double_checker_review,
    "median1": prompts.task_prompts.median1.get_scientist_prompt_with_double_checker_review,
    "median2": prompts.task_prompts.median2.get_scientist_prompt_with_double_checker_review,
    "mestranol_similarity": prompts.task_prompts.mestranol_similarity.get_scientist_prompt_with_double_checker_review,
    "osimertinib_mpo": prompts.task_prompts.osimertinib_mpo.get_scientist_prompt_with_double_checker_review,
    "perindopril_mpo": prompts.task_prompts.perindopril_mpo.get_scientist_prompt_with_double_checker_review,
    "qed": prompts.task_prompts.qed.get_scientist_prompt_with_double_checker_review,
    "ranolazine_mpo": prompts.task_prompts.ranolazine_mpo.get_scientist_prompt_with_double_checker_review,
    "scaffold_hop": prompts.task_prompts.scaffold_hop.get_scientist_prompt_with_double_checker_review,
    "sitagliptin_mpo": prompts.task_prompts.sitagliptin_mpo.get_scientist_prompt_with_double_checker_review,
    "thiothixene_rediscovery": prompts.task_prompts.thiothixene_rediscovery.get_scientist_prompt_with_double_checker_review,
    "troglitazon_rediscovery": prompts.task_prompts.troglitazon_rediscovery.get_scientist_prompt_with_double_checker_review,
    "valsartan_smarts": prompts.task_prompts.valsartan_smarts.get_scientist_prompt_with_double_checker_review,
    "zaleplon_mpo": prompts.task_prompts.zaleplon_mpo.get_scientist_prompt_with_double_checker_review
}

TASK_TO_DOUBLE_CHECKER_PROMPT = {
    "albuterol_similarity": prompts.task_prompts.albuterol_similarity.get_double_checker_prompt,
    "isomers_c7h8n2o2": prompts.task_prompts.isomers_c7h8n2o2.get_double_checker_prompt,
    "isomers_c9h10n2o2pf2cl": prompts.task_prompts.isomers_c9h10n2o2pf2cl.get_double_checker_prompt,
    "amlodipine_mpo": prompts.task_prompts.amlodipine_mpo.get_double_checker_prompt,
    "celecoxib_rediscovery": prompts.task_prompts.celecoxib_rediscovery.get_double_checker_prompt,
    "deco_hop": prompts.task_prompts.deco_hop.get_double_checker_prompt,
    "drd2": prompts.task_prompts.drd2.get_double_checker_prompt,
    "fexofenadine_mpo": prompts.task_prompts.fexofenadine_mpo.get_double_checker_prompt,
    "gsk3b": prompts.task_prompts.gsk3b.get_double_checker_prompt,
    "jnk3": prompts.task_prompts.jnk3.get_double_checker_prompt,
    "median1": prompts.task_prompts.median1.get_double_checker_prompt,
    "median2": prompts.task_prompts.median2.get_double_checker_prompt,
    "mestranol_similarity": prompts.task_prompts.mestranol_similarity.get_double_checker_prompt,
    "osimertinib_mpo": prompts.task_prompts.osimertinib_mpo.get_double_checker_prompt,
    "perindopril_mpo": prompts.task_prompts.perindopril_mpo.get_double_checker_prompt,
    "qed": prompts.task_prompts.qed.get_double_checker_prompt,
    "ranolazine_mpo": prompts.task_prompts.ranolazine_mpo.get_double_checker_prompt,
    "scaffold_hop": prompts.task_prompts.scaffold_hop.get_double_checker_prompt,
    "sitagliptin_mpo": prompts.task_prompts.sitagliptin_mpo.get_double_checker_prompt,
    "thiothixene_rediscovery": prompts.task_prompts.thiothixene_rediscovery.get_double_checker_prompt,
    "troglitazon_rediscovery": prompts.task_prompts.troglitazon_rediscovery.get_double_checker_prompt,
    "valsartan_smarts": prompts.task_prompts.valsartan_smarts.get_double_checker_prompt,
    "zaleplon_mpo": prompts.task_prompts.zaleplon_mpo.get_double_checker_prompt
}

TASK_TO_FUNCTIONAL_GROUP = {
    "albuterol_similarity": utils.utils.describe_albuterol_features,
    "isomers_c7h8n2o2": utils.utils.count_atoms,
    "isomers_c9h10n2o2pf2cl": utils.utils.count_atoms,
    "amlodipine_mpo": utils.utils.describe_albuterol_features,
    "celecoxib_rediscovery": utils.utils.describe_celecoxib_features,
    "deco_hop": utils.utils.describe_deco_hop_features,
    "drd2": utils.utils.describe_drd2_features,
    "fexofenadine_mpo": utils.utils.describe_fexofenadine_features,
    "gsk3b": utils.utils.describe_gsk3b_features,
    "jnk3": utils.utils.describe_jnk3_features,
    "median1": utils.utils.describe_median1_features,
    "median2": utils.utils.describe_median2_features,
    "mestranol_similarity": utils.utils.describe_mestranol_features,
    "osimertinib_mpo": utils.utils.describe_osimertinib_features,
    "perindopril_mpo": utils.utils.describe_albuterol_features,
    "qed": utils.utils.describe_qed_features,
    "ranolazine_mpo": utils.utils.describe_ranolazine_features,
    "scaffold_hop": utils.utils.describe_scaffold_hop_features,
    "sitagliptin_mpo": utils.utils.describe_sitagliptin_features,
    "thiothixene_rediscovery": utils.utils.describe_thiothixene_features,
    "troglitazon_rediscovery": utils.utils.describe_troglitazon_features,
    "valsartan_smarts": utils.utils.describe_valsartan_features,
    "zaleplon_mpo": utils.utils.describe_zaleplon_features,
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
                TEXT_SMILES_HISTORY, topk_smiles
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
    if SMILES not in state["smiles_history"]:
        state["smiles_history"].append(SMILES)
    else:
        state["redundant_smiles"] = True
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
        functional_groups = TASK_TO_FUNCTIONAL_GROUP[task](mol)
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
    builder.add_node("scientist_node", scientist_node)
    builder.add_node("reviewer_node", reviewer_node)
    builder.add_node("retrieval_node", retrieval_node)
    builder.add_node("double_checker_node", double_checker_node)
    
    builder.set_entry_point("retrieval_node")  # if retrieval is the first step
    
    # After reviewer, decide whether to continue (reviewer → scientist OR END)
    builder.add_edge("retrieval_node", "scientist_node")
    # builder.add_conditional_edges("scientist_node", check_redundant_smiles)
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
        "smiles_history": [],
        "redundant_smiles": False,
    }
    # Run the graph
    final_state = graph.invoke(input_state, {"recursion_limit": 9999})

    # Print final state for confirmation
    print("\nFinal State:")
    for k, v in final_state.items():
        print(f"{k}: {v}")
    wandb.finish()
