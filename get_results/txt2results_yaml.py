import os
import yaml
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def find_log_folder(run_id):
    logs_dir = "../logs"
    for folder_name in os.listdir(logs_dir):
        if run_id in folder_name:
            return os.path.join(logs_dir, folder_name)
    return None

def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum_auc = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1]))
    buffer_max_idx = ordered_results[-1][1][1] # last components' [1][1]: score
    
    for idx in range(freq_log, min(buffer_max_idx, max_oracle_calls), freq_log):
        temp_result = [item for item in ordered_results if item[1][1] <= idx]
        if len(temp_result) == 0:
            continue
        top_n_now = np.mean(
            [item[1][0] for item in sorted(temp_result, key=lambda kv: kv[1][0], reverse=True)[:top_n]]
        )
        sum_auc += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx
    
    final_result = sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True)[:top_n]
    top_n_now = np.mean([item[1][0] for item in final_result])
    sum_auc += (buffer_max_idx - called) * (top_n_now + prev) / 2

    if finish and buffer_max_idx < max_oracle_calls:
        sum_auc += (max_oracle_calls - buffer_max_idx) * top_n_now
    
    return sum_auc / max_oracle_calls

def plot_topn_avg_curve(buffer, top_n, freq_log, max_oracle_calls):
    ordered_results = sorted(buffer.items(), key=lambda kv: kv[1][1])

    x, y = [], []
    for idx in range(1, max_oracle_calls + 1, freq_log):
        current_results = [item for item in ordered_results if item[1][1] <= idx] # smiles: score, idx
        
        if len(current_results) == 0:
            x.append(0)
            continue
        top_n_now = np.mean(
            [item[1][0] for item in sorted(current_results, key=lambda kv: kv[1][0], reverse=True)[:top_n]]
        )
        x.append(idx)
        y.append(top_n_now)

    return x, y

def load_yaml_and_compute_auc(yaml_path, top_k=10, freq_log=1, max_oracle_calls=1000, log_dir=None):
    with open(yaml_path, 'r') as f:
        buffer = yaml.safe_load(f)

    finish = len(buffer) >= max_oracle_calls
    auc_score = top_auc(buffer, top_k, finish, freq_log, max_oracle_calls)
    print(f"Top-{top_k} AUC score: {auc_score:.4f}")

    return auc_score

# Complete
def convert_txt_to_results_yaml(txt_path, output_path):
    mol_buffer = {}
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines, 1):
        try:
            smi, score = line.strip().split(',')
            smi = smi.strip()
            score = float(score.strip())
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canonical_smi = Chem.MolToSmiles(mol)# ,kekuleSmiles=True)
            if canonical_smi not in mol_buffer:
                mol_buffer[canonical_smi] = [score, idx]
        except Exception:
            continue

    mol_buffer = dict(sorted(mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    
    if "tni8f7y4" not in output_path:
        with open(output_path, 'w') as f:
            yaml.dump(mol_buffer, f, sort_keys=False)


projects=["pmo_v4_no_redundancy"]
tasks = [
    "albuterol_similarity",
    "amlodipine_mpo",
    "celecoxib_rediscovery",
    "deco_hop",
    "drd2",
    "fexofenadine_mpo",
    "gsk3b",
    "isomers_c7h8n2o2",
    "isomers_c9h10n2o2pf2cl",
    "jnk3",
    "median1",
    "median2",
    "mestranol_similarity",
    "osimertinib_mpo",
    "perindopril_mpo",
    "qed",
    "ranolazine_mpo",
    "scaffold_hop",
    "sitagliptin_mpo",
    "thiothixene_rediscovery",
    "troglitazon_rediscovery",
    "valsartan_smarts",
    "zaleplon_mpo"
]
result = pd.DataFrame(index=tasks, columns=projects)

smiles_path_list = []
api = wandb.Api()
result = pd.DataFrame(index=tasks, columns=projects)
# Loop over projects and tasks
for project in projects:
    runs = api.runs(f"icecream126/{project}")  # <-- CHANGE 'your-entity-name' to your wandb username/team
    for task in tasks:
        matching_runs = [run for run in runs if run.name == task]
        if not matching_runs:
            result.at[task, project] = None
            continue
        
        run = matching_runs[0]  # Assume 1 run per task per project
        run_id = run.id

        # 1. Find corresponding folder
        log_folder = find_log_folder(run_id)
        if log_folder is None:
            print(f"Warning: Cannot find log folder for run_id {run_id}")
            result.at[task, project] = None
            continue

        smiles_path = os.path.join(log_folder, "smiles.txt")
        smiles_path_list.append([smiles_path, project, task])

for obj in smiles_path_list:
    txt_path, project, task = obj
    print(f"\n\n== Project: {project}, Task: {task} ==")
    print(f"dir path: {txt_path.split('/')[-2]}")
    log_dir = txt_path[:-10]
    yaml_path = os.path.join(log_dir, 'results.yaml')

    convert_txt_to_results_yaml(txt_path, yaml_path)
    auc_score = load_yaml_and_compute_auc(yaml_path, top_k=10, freq_log=1, max_oracle_calls=1000, log_dir=log_dir)

    result.at[task, project] = auc_score
result.to_csv(f"{projects[0]}.csv")
print(f"Saved to {projects[0]}.csv")
