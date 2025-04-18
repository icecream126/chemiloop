# LangGraph-based multi-agent molecular generation system with tool execution, reflection, and feedback

from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict, Union
from langchain_deepseek import ChatDeepSeek
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import json
import csv
from utils.args import parse_args, return_API_keys
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.chains import RetrievalQA
from langchain.vectorstores.base import VectorStoreRetriever
from utils.langchain_utils import split_document, get_vectorstore

from guacamol.goal_directed_benchmark import GoalDirectedBenchmark
from guacamol.utils.chemistry import canonicalize
from guacamol.common_scoring_functions import TanimotoScoringFunction
import utils.utils
import get_prompts
# import rdKit


import json
import csv
import os
import wandb
import re
import datetime

wandb.init(project="pmo_test", name="pmo_test")

current_time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
log_dir = f"./logs/{current_time}-{wandb.run.id}/"
os.makedirs(log_dir, exist_ok=True)

OPENAI_API_KEY = return_API_keys()["OPENAI_API_KEY"] 
DEEPSEEK_API_KEY = return_API_keys()["DEEPSEEK_API_KEY"]
chromophore_text_file_path = "/home/khm/chromophore/dataset/guacamol.txt"
rdkit_txt_file_path = "/home/khm/chromophore/tests/rdkit_tool_registry_summary.txt"
LOG_PATH = f"{log_dir}output.txt"

# retrieval_agent_llm = llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)
generation_agent_llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)
tool_select_llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)
scientist_llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)
double_checker_llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)
reviewer_llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)

albuterol_smiles = 'CC(C)(C)NCC(C1=CC(=CC=C1)O)O'
canonical_smiles = canonicalize(albuterol_smiles)


scoring_fn = TanimotoScoringFunction(
    target=canonical_smiles,
    fp_type='AP'  # you can also try 'AP', 'FCFP' etc.
)

# --------------------------
# Logging helper
# --------------------------
def log(msg: str):
    with open(LOG_PATH, "a") as f:
        f.write(msg + "\n")
# --------------------------
# Graph State
# --------------------------
class GraphState(TypedDict):
    prompt: str
    target_prop: List[str]
    retrieved_smiles: List[str]
    generated_smiles: str
    selected_tools: List[str]
    tool_outputs: List[Union[str, List[float], float]]
    generation_thinking: dict
    scientist_thinking: dict
    reviewer_thinking: dict
    improved_smiles: str
    iteration: int
    max_iterations: int
    review: str
    double_checker_feedback: str
    smiles_history: List[str]
    thinking_history: List[str]
    cumulative_oracle_score: List[float]
    auc_topk_over_time: List[float]

# --------------------------
# Agent to retrieve similar molecules from vectorstore
# --------------------------
def retrieval_agent(state: GraphState) -> GraphState:
    print("\n==== Retrieval Agent ==")
    # llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)
    text_loader = TextLoader(chromophore_text_file_path)
    text_docs = text_loader.load()
    docs = split_document(text_docs)
    embedding_model = OpenAIEmbeddings(api_key=OPENAI_API_KEY)
    vectorstore = get_vectorstore(docs, embedding_model, 50)

    retriever: VectorStoreRetriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    top_docs = retriever.get_relevant_documents(state['prompt'])

    # 각 문서에서 SMILES만 추출 (문서 내용에 SMILES만 포함되어 있다고 가정)
    smiles_list = []
    for doc in top_docs:
        match = re.search(r"SMILES (.*?) has", doc.page_content)
        if match:
            smiles = match.group(1).strip()
            if Chem.MolFromSmiles(smiles):
                smiles_list.append(smiles)
    
    print("Response from retrieval agent:", top_docs)
    print("Retrieved SMILES:", smiles_list)

    log("\n==== Retrieval Agent ==")
    log("Response from retrieval agent:")
    log(str(top_docs))
    log("\nRetrieved SMILES:")
    log(str(smiles_list))
    return {**state, "retrieved_smiles": smiles_list}

def generation_agent(state: GraphState) -> GraphState:
    print("\n==== Generation node ==")
    # llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)
    prompt = get_prompts.get_generation_node_prompt(state["prompt"], state["retrieved_smiles"])
    response = generation_agent_llm([HumanMessage(content=prompt)]).content.strip()

    print("Response from generation node:", response)
    log("Prompt for generation:")
    log(prompt)
    log("Response from generation node:")
    log(response)

    # Extract steps and SMILES
    step_dict = {}
    current_step = None
    lines = response.splitlines()
    for line in lines:
        line = line.strip()
        if line.lower().startswith("step"):
            match = re.match(r"(Step \d+):", line)
            if match:
                current_step = match.group(1).lower().replace(" ", "")  # e.g., 'step1'
                step_dict[current_step] = line
        elif current_step:
            step_dict[current_step] += " " + line

    # Extract final SMILES
    smiles_match = re.search(r"Final proposed SMILES:\s*(.*)$", response, re.IGNORECASE)
    smiles = smiles_match.group(1).strip() if smiles_match else ""

    # Save structured steps and SMILES
    state_out = {
        **state,
        "generated_smiles": smiles,
        "thinking": step_dict,
        # "smiles_history": state.get("smiles_history", []) + [smiles],
        # "thinking_history": state.get("thinking_history", []) + [step_dict],
        # "iteration": state["iteration"] + 1
    }
    return state_out

# --------------------------
# Node to select which tools to use
# --------------------------
def tool_select_node(state: GraphState) -> GraphState:
    print("\n==== Tool Selection Node ==")
    query = state["prompt"] 

    docs = utils.utils.load_tool_descriptions(rdkit_txt_file_path)
    vs = utils.utils.build_vectorstore(docs)
    descriptions, tool_names = utils.utils.retrieve_top_k_tools(vs, query)
    descriptions = utils.utils.get_pretty_description_str(descriptions)

    prompt = get_prompts.get_tool_select_prompt(descriptions, query)
    response = tool_select_llm([HumanMessage(content=prompt)]).content
    selected = [name.strip() for name in response.split(",")] # if name.strip() in TOOL_REGISTRY]
    print("Response from tool selection:", response)
    print("Selected tools:", selected)

    log("\n==== Tool Selection Node ==")
    log("Prompt for tool selection:")
    log(prompt)
    log("Response from tool selection:")
    log(response)
    log("Selected tools:")
    log(str(selected))
    return {**state, "selected_tools": selected}

# --------------------------
# Tool execution node
# --------------------------
def tool_executor_node(state: GraphState) -> GraphState:
    print("\n==== Tool Output Node ==")
    mol = Chem.MolFromSmiles(state["generated_smiles"])
    if mol is None:
        return {**state, "tool_outputs": ["Invalid SMILES"]}
    outputs = []
    for tool_name in state.get("selected_tools", []):
        try:
            tool_fn = getattr(rdMolDescriptors, tool_name)
            outputs.append(tool_fn(mol))
        except Exception as e:
            outputs.append(f"[{tool_name}] error: {e}")
    
    print("Tool outputs:", outputs)

    log("\n==== Tool Output Node ==")
    log("Tool outputs:")
    log(str(outputs))
    return {**state, "tool_outputs": outputs}

# --------------------------
# Scientist node
# --------------------------

def scientist_node(state: GraphState) -> GraphState:
    print("\n==== Scientist Node ==")
    # llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)
    prompt = get_prompts.get_scientist_prompt(state["prompt"], state["retrieved_smiles"], state["retrieved_smiles"], state["tool_outputs"])
    response = scientist_llm([HumanMessage(content=prompt)]).content.strip()

    # Step-by-step extraction
    step_dict = {}
    current_step = None
    lines = response.splitlines()
    for line in lines:
        line = line.strip()
        if line.lower().startswith("step"):
            match = re.match(r"(Step \d+):", line)
            if match:
                current_step = match.group(1).lower().replace(" ", "")  # e.g., 'step1'
                step_dict[current_step] = line
        elif current_step:
            step_dict[current_step] += " " + line

    # SMILES extraction
    smiles_match = re.search(r"Final proposed SMILES:\s*(.*)$", response, re.IGNORECASE)
    smiles = smiles_match.group(1).strip() if smiles_match else ""

    print("Response from scientist node:", response)
    print("Step-wise thinking:", step_dict)
    print("Improved SMILES:", smiles)

    log("\n==== Scientist Node ==")
    log("Response from scientist node:")
    log(response)
    log("Step-wise thinking:")
    log(str(step_dict))
    log("Improved SMILES:")
    log(smiles)
    

    return {
        **state,
        "scientist_thinking": step_dict,
        "improved_smiles": smiles,
        "generated_smiles": smiles,
        # "iteration": state["iteration"] + 1,
        # "smiles_history": state.get("smiles_history", []) + [smiles],
        # "thinking_history": state.get("thinking_history", []) + [step_dict]
    }

# --------------------------
# Double checker node
# --------------------------
def double_checker_node(state: GraphState) -> GraphState:
    print("\n==== Double Checker Node ==")
    # llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)
    pred_value = scoring_fn.score(state["improved_smiles"])
    
    mol = Chem.MolFromSmiles(state["improved_smiles"])
    from rdkit.Chem import Draw
    img_path = os.path.join(log_dir, f"{state['iteration']}_intermediate_structure.jpg")
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        img.save(img_path)
    
    wandb.log({f"intermediate_score-{state['iteration']}": pred_value})
    wandb.log({f"intermediate_structure_image-{state['iteration']}": wandb.Image(img_path)})

    prompt = get_prompts.get_double_checker_prompt(state["prompt"], state["scientist_thinking"], state["improved_smiles"])
    
    response = double_checker_llm([HumanMessage(content=prompt)]).content.strip()
    
    print("Response from double checker node:", response)
    log("\n==== Double Checker Node ==")
    log("Prompt for double checker:")
    log(prompt)
    log("Response from double checker node:")
    log(response)
    return {**state, "double_checker_feedback": response}

# --------------------------
# Reviewer node
# --------------------------
# from guacamol.goal_directed_benchmark import goal_directed_benchmark
oracle_scores = []
cumulative_auc_scores = []
def reviewer_node(state: GraphState) -> GraphState:

    # benchmark = GoalDirectedBenchmark(version_name='v1')

    print("\n==== Reviewer Node ==")
    # llm = ChatDeepSeek(model="deepseek-chat", temperature=1.0, api_key=DEEPSEEK_API_KEY)
    # TODO: Fix the below arguments for prompt
    if "molecular weight" in state["target_prop"]:
        pred_value = Chem.Descriptors.ExactMolWt(Chem.MolFromSmiles(state["improved_smiles"]))
    elif "albuterol_similarity" in state["target_prop"]:
        
        pred_value = scoring_fn.score(state["improved_smiles"])
        oracle_scores.append(pred_value)
        state["cumulative_oracle_score"].append(pred_value)

    else:
        raise NotImplementedError("Target property not implemented")

    from utils.utils import compute_auc_topk_online_torch
    if len(oracle_scores) != 0 and len(oracle_scores)%10==0:
        auc_score = compute_auc_topk_online_torch(oracle_scores)
        cumulative_auc_scores.append(auc_score)
        state["auc_topk_over_time"].append(auc_score)
        wandb.log("auc_score", auc_score)
    else:
        auc_score = 0
        state["auc_topk_over_time"].append(cumulative_auc_scores[-1] if cumulative_auc_scores else 0)

    print("AUC score:", auc_score)

    from rdkit_functional import detect_functional_groups
    functional_groups = detect_functional_groups(state['generated_smiles'])
    prompt = get_prompts.get_reviewer_prompt(state["prompt"], state['generated_smiles'], pred_value, functional_groups, state["scientist_thinking"])
    response = reviewer_llm([HumanMessage(content=prompt)]).content.strip()

    mol = Chem.MolFromSmiles(state["improved_smiles"])
    from rdkit.Chem import Draw
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        img.save(os.path.join(log_dir, f"{state['iteration']}_structure.jpg"))

    wandb.log({
        "score": pred_value,
        "smiles": state["improved_smiles"].strip(),
        "structure_image": wandb.Image(os.path.join(log_dir, f"{state['iteration']}_structure.jpg")),
        # "auc_score": state["cumulative_oracle_score"],
        # "auc_topk_over_time": state["auc_topk_over_time"],
    })

    print("Response from reviewer node:", response)
    log("\n==== Reviewer Node ==")
    log("Response from reviewer node:")
    log(response)
    log("Predicted value:")
    log(str(pred_value))
    log("Functional groups:")
    log(str(functional_groups))
    state_out = {
        **state,
        "review": response,
        "iteration": state["iteration"] + 1
    }
    return state_out

# --------------------------
# End condition checker
# --------------------------
def end_condition(state: GraphState) -> bool:

    print("\n==== End Condition Check ====")
    if state["iteration"] >= state["max_iterations"]:
        return True
    # mol = Chem.MolFromSmiles(state["improved_smiles"])
    # if mol:
    #     mw = Descriptors.MolWt(mol)
    #     print("Current iteration:", state["iteration"])
    #     print("SMILES:", state["improved_smiles"])
    #     print("Molecular weight:", mw)
    #     log("\n==== End Condition Check ====")
    #     log("Current iteration:")
    #     log(str(state["iteration"]))
    #     log("SMILES:")
    #     log(state["improved_smiles"])
    #     log("Molecular weight:")
    #     log(str(mw))
    #     if "80" in state["prompt"] and abs(mw - 80) < 1.0:
    #         return True
    return False

# --------------------------
# Main execution
# --------------------------
if __name__ == "__main__":
    builder = StateGraph(GraphState)

    # Add nodes
    builder.add_node("retrieval_agent", retrieval_agent)
    builder.add_node("generation_agent", generation_agent)
    builder.add_node("tool_select_node", tool_select_node)
    builder.add_node("tool_executor", tool_executor_node)
    builder.add_node("scientist_node", scientist_node)
    builder.add_node("double_checker_node", double_checker_node)
    builder.add_node("reviewer_node", reviewer_node)

    # Define graph transitions
    builder.set_entry_point("retrieval_agent")
    builder.add_edge("retrieval_agent", "generation_agent")
    builder.add_edge("generation_agent", "tool_select_node")
    builder.add_edge("tool_select_node", "tool_executor")
    builder.add_edge("tool_executor", "scientist_node")

    # Double-checker loop until consistent
    def route_after_checker(state: GraphState) -> str:
        if state["double_checker_feedback"].strip().lower() == "consistent":
            return "reviewer_node"
        return "scientist_node"

    builder.add_conditional_edges("scientist_node", lambda state: "double_checker_node")
    builder.add_conditional_edges("double_checker_node", route_after_checker)

    # Review → end or loop again
    def route_after_review(state: GraphState) -> str:
        return END if end_condition(state) else "scientist_node"

    builder.add_conditional_edges("reviewer_node", route_after_review)

    graph = builder.compile()
    # graph.get_graph().render_graph("langgraph_diagram", format="png")
    
    # from IPython.display import display, Image
    # display(Image(graph.get_graph().draw_mermaid_png()))

    input_state: GraphState = {
        "prompt": "THIS IS YOUR GOAL: Generate a drug-like molecule that is structurally similar to albuterol. The molecule should preserve the general scaffold and retain similar functional groups such as hydroxyl, amine, and aromatic ring.",
        "cumulative_oracle_score": [],
        "auc_topk_over_time": [],
        "target_prop": ["albuterol_similarity"],
        "retrieved_smiles": [],
        "generated_smiles": "",
        "selected_tools": [],
        "tool_outputs": [],
        "generation_thinking": "",
        "scientist_thinking": "",
        "reviewer_thinking": "",
        "improved_smiles": "",
        "iteration": 0,
        "max_iterations": 10000,
        "review": "",
        "double_checker_feedback": "",
        "smiles_history": [],
        "thinking_history": []
    }

    final_state = graph.invoke(input_state)

    # 결과 출력
    print("\n✅ Final State:")
    for key, value in final_state.items():
        print(f"{key}: {value}\n")

    # JSON 저장
    with open("final_state.json", "w") as f:
        json.dump(final_state, f, indent=2)

    # CSV 저장
    with open("smiles_log.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Prompt", "Final SMILES", "Tool Outputs", "Review", "Final Thinking"])
        writer.writerow([
            final_state["prompt"],
            final_state["improved_smiles"],
            str(final_state["tool_outputs"]),
            final_state["review"],
            final_state["thinking"]
        ])