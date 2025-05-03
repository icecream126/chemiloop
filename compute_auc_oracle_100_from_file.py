import pandas as pd
import wandb
import os
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem


score_list = []
def find_log_folder(run_id):
    logs_dir = "./logs"
    for folder_name in os.listdir(logs_dir):
        if run_id in folder_name:
            return os.path.join(logs_dir, folder_name)
    return None

# ---------------------------
# AUC calculation function (same as your framework)
# ---------------------------
# https://github.com/wenhao-gao/mol-opt/blob/63382d78890e910080ef9a9b3b6d04a4552aff85/molopt/base.py#L86
def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum = 0
    prev = 0
    called = 0
    
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        
        sum += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum += (len(buffer) - called) * (top_n_now + prev) / 2
    if finish and len(buffer) < max_oracle_calls:
        sum += (max_oracle_calls - len(buffer)) * top_n_now
    return sum / max_oracle_calls

# ---------------------------
# Function to process the text file
# ---------------------------
def compute_topk_auc_from_file(file_path, top_k=10, max_oracle_calls=1000, freq_log=1):# , run_name='', project='', task=''):
    smiles_scores = []

    # Step 1: Read SMILES and score from the file
    with open(file_path, 'r') as f:
        for line in f:
            if ',' in line:
                smi, score = line.strip().split(',')
                
                smiles_scores.append((smi.strip(), float(score.strip())))

    print("number of generated smiles", len(smiles_scores))
    # Step 2: Remove invalid or duplicate SMILES
    seen = set()
    mol_buffer = {}

    for i, (smi, score) in enumerate(smiles_scores):
        mol = Chem.MolFromSmiles(smi)
        if mol:  # valid SMILES
            canonical_smi = Chem.MolToSmiles(mol)  # Canonicalize
            if canonical_smi not in seen:
                # Store as {canonical_smi: [score, generation index]}
                # mol_buffer[canonical_smi] = [score, len(mol_buffer) + 1]
                mol_buffer[canonical_smi] = [score, len(mol_buffer) + 1]
                seen.add(canonical_smi)
            else:
                continue  # duplicated canonical SMILES
        else:
            continue  # invalid SMILES
    # import pdb; pdb.set_trace()

    # Step 3: Compute AUC
    # print(f"Project: {project}, Task: {task}, Run Name: {run_name}")
    # import pdb; pdb.set_trace()
    print("mol buffer length: ", len(mol_buffer))
    if len(mol_buffer)<10:
        print("MOL buffer length is less than 10, skipping AUC calculation.")
        auc=0
    else:
        auc = top_auc(mol_buffer, top_k, finish=False, freq_log=freq_log, max_oracle_calls=max_oracle_calls)
        print(f"Top-{top_k} AUC Score: {auc:.4f}")
    # score_list.append([project, task, auc, len(mol_buffer)])

    return auc, mol_buffer


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    # smiles_history_path_list = ["/home/khm/chemiloop/logs/2025-04-29-01-37-49-6na6ezz7/smiles.txt"]
    smiles_history_path_list = ["/home/khm/chemiloop/logs/2025-04-29-01-37-49-6na6ezz7/smiles.txt"]
    
    top_k = 10
    max_oracle_calls = 1000
    freq_log = 1
    for file_path in smiles_history_path_list:
        run_name = file_path.split('/')[-2].split('-')[-1]
        auc_score, mol_buffer = compute_topk_auc_from_file(file_path=file_path, top_k=top_k, max_oracle_calls=max_oracle_calls, freq_log=freq_log)# , project, task)
        # print(mol_buffer)
        # print("mol buffer length: ", len(mol_buffer))
        # plot_auc_progress(mol_buffer, top_k, freq_log, max_oracle_calls, run_name)
        # plot_auc_progress_2(mol_buffer, top_k, freq_log, max_oracle_calls, run_name)

        print("AUC score: ", auc_score)
        print('mol buffer length: ', len(mol_buffer))