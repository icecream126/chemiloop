from rdkit import Chem, RDLogger
import torch
import glob
import re
import os
import numpy as np
import torch.nn.functional as F
from rdkit.Chem import Draw
import wandb
from langchain.vectorstores import FAISS
from langchain.schema import Document
from rdkit.Chem import Fragments
import re
from rdkit import Chem
from rdkit.Chem import Fragments
from langchain.embeddings import HuggingFaceEmbeddings

bce_loss = torch.nn.BCEWithLogitsLoss(reduction='mean')

from rdkit import Chem
from collections import defaultdict

def format_topk_smiles(topk_smiles):
    formatted = "\n".join(
        f"({repr(smiles.strip())}, {score:.6f})"
        for smiles, score in topk_smiles
    )
    return formatted

def add_with_limit(s, item, max_len=20):
    if len(s) < max_len:
        s.add(item)
        # return s
    else:
        print(f"Cannot add '{item}': reached max size ({max_len})")

def format_set_as_text(s):
    if not s:
        return "Currently no history"
    return "\n".join(sorted(s))



def count_atoms(m):
    m = Chem.AddHs(m)

    atomic_count = defaultdict(lambda: 0)
    for atom in m.GetAtoms():
        atomic_count[atom.GetSymbol()] += 1

    # Convert to text format
    text_output = ""
    for atom, count in sorted(atomic_count.items()):
        text_output += f"- {atom}: {count}\n"

    return text_output



def describe_albuterol_features(mol):
    descriptions = []

    phenol_count = Fragments.fr_phenol(mol)
    if phenol_count:
        descriptions.append(f"- {phenol_count} phenol group(s) (aromatic hydroxyl).")

    aromatic_oh = Fragments.fr_Ar_OH(mol)
    if aromatic_oh:
        descriptions.append(f"- {aromatic_oh} aromatic hydroxyl group(s).")

    secondary_amine = Fragments.fr_NH1(mol)
    if secondary_amine:
        descriptions.append(f"- {secondary_amine} secondary amine group(s).")

    aliphatic_oh = Fragments.fr_Al_OH(mol)
    if aliphatic_oh:
        descriptions.append(f"- {aliphatic_oh} aliphatic hydroxyl group(s), possibly benzylic alcohol.")

    benzene_rings = Fragments.fr_benzene(mol)
    if benzene_rings:
        descriptions.append(f"- {benzene_rings} benzene ring(s).")

    aryl_methyl = Fragments.fr_aryl_methyl(mol)
    if aryl_methyl:
        descriptions.append(f"- {aryl_methyl} aryl methyl group(s), which may relate to ring substituents.")

    if not descriptions:
        descriptions.append("- No key albuterol-like fragments found.")

    res = "\n".join(descriptions)
    return res

import re

def get_reviewer_think_dict(response: str):
    response = response.strip()
    step_thinking = {}

    # Find all step matches
    step_matches = list(re.finditer(r"(Step\s*\d+)\s*Feedback:\s*", response))

    for i in range(len(step_matches)):
        step_key = step_matches[i].group(1).lower().replace(" ", "")  # e.g., step1
        start = step_matches[i].end()

        if i + 1 < len(step_matches):
            end = step_matches[i + 1].start()
        else:
            end = len(response)

        step_content = response[start:end].strip()
        step_thinking[step_key] = step_content

    return step_thinking


def get_scientist_think_dict(response: str):
    response = response.strip()
    step_thinking = {}

    # Find all step headers
    step_matches = list(re.finditer(r"(Step\d+):\s*(.+?)\n", response))

    # Extract each step section
    for i in range(len(step_matches)):
        step_key = step_matches[i].group(1).lower()  # e.g., step1
        start = step_matches[i].start()

        if i + 1 < len(step_matches):
            end = step_matches[i + 1].start()
        else:
            end = len(response)

        step_text = response[start:end].strip()
        step_thinking[step_key] = step_text

    # Extract SMILES string from "Final Output" section
    smiles_match = re.search(r"Final Output:\s*SMILES:\s*([^\s]+)", response)
    if smiles_match:
        step_thinking["SMILES"] = smiles_match.group(1)

    return step_thinking


import torch
from typing import List

def compute_auc_topk_online_torch(score_list: List[float], k: int = 10) -> float:
    """
    Compute the AUC of top-k average scores vs number of oracle calls using torch.

    Args:
        score_list: List of scores from oracle evaluations (one per call)
        topk: Number of top scores to average

    Returns:
        AUC value normalized to [0, 1]
    """
    if len(score_list) == 0:
        return 0.0

    scores = torch.tensor(score_list)
    topk_curve = []

    for i in range(1, len(scores) + 1):
        current_topk = torch.topk(scores[:i], min(k, i)).values
        avg_topk = current_topk.mean().item()
        topk_curve.append(avg_topk)

    curve_tensor = torch.tensor(topk_curve)
    auc = torch.trapz(curve_tensor, dx=1.0) / (len(score_list) * 1.0)

    # Normalize AUC to [0, 1] range using min-max (assuming scores ∈ [0, 1])
    return float(auc)



def get_pretty_description_str(description_list):
    # Parse into registry
    TOOL_REGISTRY = {}

    for line in description_list:
        try:
            name_match = re.match(r"^(.*?):", line)
            input_match = re.search(r"\(input:\s*(.*?)\s*,\s*output:", line)
            output_match = re.search(r"output:\s*(.*?)\)", line)
            desc_end = re.search(r"\(input:", line)

            if name_match:
                name = name_match.group(1).strip()
                description = line[len(name_match.group(0)):desc_end.start()].strip() if desc_end else line
                input_type = input_match.group(1).strip() if input_match else "Unknown"
                output_type = output_match.group(1).strip() if output_match else "Unknown"

                TOOL_REGISTRY[name] = {
                    "description": description,
                    "input_type": input_type,
                    "output_type": output_type
                }
        except Exception as e:
            print(f"Error parsing: {line} - {e}")

    # Convert to tool_list_str
    tool_list_str = "\n".join([
        f"{name}: {meta['description']} (input: {meta['input_type']}, output: {meta['output_type']})"
        for name, meta in TOOL_REGISTRY.items()
    ])

    return tool_list_str


# Step 1: Parse the tool file
def load_tool_descriptions(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    tool_blocks = re.split(r"(?=Function: )", raw_text.strip())
    documents = []

    for block in tool_blocks:
        name_match = re.search(r"Function:\s*(.*)", block)
        desc_match = re.search(r"Description:\s*(.*)", block)
        input_match = re.search(r"Input Type\s*:\s*(.*)", block)
        output_match = re.search(r"Output Type\s*:\s*(.*)", block)

        if name_match:
            name = name_match.group(1).strip()
            desc = desc_match.group(1).strip() if desc_match else ""
            input_type = input_match.group(1).strip() if input_match else ""
            output_type = output_match.group(1).strip() if output_match else ""

            full_text = f"{name}: {desc} (input: {input_type}, output: {output_type})"
            documents.append(Document(page_content=full_text, metadata={"tool_name": name}))
    return documents

# Step 2: Build vectorstore
def build_vectorstore(documents):
    # embedding_model = OpenAIEmbeddings()  # or DeepSeekEmbeddings
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    return FAISS.from_documents(documents, embedding_model)

# Step 3: Retrieve top-K relevant tools
def retrieve_top_k_tools(vectorstore, query: str, k=20):
    top_docs = vectorstore.similarity_search(query, k=k)
    top_descriptions = [doc.page_content for doc in top_docs]
    top_tool_names = [doc.metadata["tool_name"] for doc in top_docs]
    return top_descriptions, top_tool_names


def get_pretty_topk_string(topk_dict, max_smiles_length, property_name):
    res = ""
    for item in topk_dict:
        res += f"SMILES: {item['smiles']}  | {property_name}: {float(item[property_name]):.5f}\n"

    return res

def get_scientist_output_dict(scientist_output):
    step_pattern = re.compile(r"Step (\d+):.*?\n(.*?)(?=\nStep \d+:|\nFinal proposed SMILES:|\Z)", re.DOTALL)
    steps = {f"step{m.group(1)}": m.group(2).strip() for m in step_pattern.finditer(scientist_output)}

    smiles_match = re.search(r"Final proposed SMILES:\s*\n?([^\s<]+)", scientist_output)
    steps["SMILES"] = smiles_match.group(1).strip() if smiles_match else None
    return steps

def get_reviewer_output_dict(reviewer_output):
    # Extract feedback for each step using a non-greedy match until the next "Step X:" or end of string
    # Use regex to extract feedback for each step
    pattern = r"Step (\d+): (.*?)(?=(?:\n)?Step \d+:|\Z)"
    matches = re.findall(pattern, reviewer_output, re.DOTALL)

    # Build feedback dictionary
    feedback_dict = {f"step{step}": feedback.strip().replace("\n", " ") for step, feedback in matches}
    return feedback_dict


def canonicalize(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        smiles = Chem.MolToSmiles(mol)
    except:
        return None   


    if len(smiles) == 0:
        return None

    return smiles

def natural_sort(l):
    convert = lambda text: int(text) if text.isdigit() else text.lower()
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def get_files_in_dir(dir, specs=None):
    if specs is None:
        return natural_sort(glob.glob(os.path.join(dir,"*")))
    else:
        return natural_sort(glob.glob(os.path.join(dir,specs)))
    

# Metrics
def get_rmse(target, predicted):
    return torch.square(torch.subtract(target, predicted)).mean().item()

def get_mae(target, predicted):
    return torch.abs(torch.subtract(target, predicted)).item()

# log results
def save_results(logger, log_dir, iteration, smiles, molweight, diff, property_unit, property_name, min_diff):
    smiles_file = os.path.join(log_dir, f"{iteration}_smiles.txt")
    molweight_file = os.path.join(log_dir, f"{iteration}_{property_name}.txt")
    image_file = os.path.join(log_dir, f"{iteration}_structure.jpg")
    
    # Save SMILES string
    with open(smiles_file, "w") as f:
        f.write(smiles)
    
    # Save molweight
    with open(molweight_file, "w") as f:
        f.write(f"{molweight} {property_unit}")
    
    # Generate molecule image
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        img = Draw.MolToImage(mol, size=(300, 300))
        img.save(image_file)
    
    # Log results to wandb\
    logger.log({
        "iteration": iteration,
        "smiles": smiles.strip(),
        property_name: molweight,
        f"{property_name}_diff": diff,
        "min_diff": min_diff,
        "structure_image": wandb.Image(image_file)
    })