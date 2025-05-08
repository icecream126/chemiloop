import os
import yaml
from rdkit import Chem
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import wandb
from rdkit import RDLogger
from tqdm import tqdm
RDLogger.DisableLog('rdApp.*')

def find_log_folder(run_id):
    logs_dir = "../logs"
    for folder_name in os.listdir(logs_dir):
        if run_id in folder_name:
            return os.path.join(logs_dir, folder_name)
    return None

def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls, buffer_max_idx=1000):
    sum_auc = 0
    prev = 0
    called = 0
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1]))
    # buffer_max_idx = ordered_results[-1][1][1] # last components' [1][1]: score
    
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

def convert_txt_to_results_yaml(txt_path, output_path):
    # smi_score_list = []
    smi_list = []
    score_list = []
    in_list = []
    mol_buffer = {}
    is_clean_smiles=True
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    
    for idx, line in enumerate(lines, 1):
        try:
            smi, score = line.strip().split(',')
            smi = smi.strip()
            score = float(score.strip())
            # smi_score_list.append([smi, score])
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                smi_list.append(smi)
                score_list.append(score)
                in_list.append("-")
                continue
            canonical_smi = Chem.MolToSmiles(mol)# ,kekuleSmiles=True)
            if canonical_smi not in mol_buffer:
                mol_buffer[canonical_smi] = [score, idx]
                smi_list.append(smi)
                score_list.append(score)
                in_list.append("O")
            else:
                smi_list.append(smi)
                score_list.append(score)
                in_list.append("-")
        except Exception:
            continue

    mol_buffer = dict(sorted(mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True))

    
    if "tni8f7y4" not in output_path:
        with open(output_path, 'w') as f:
            yaml.dump(mol_buffer, f, sort_keys=False)

    return smi_list, score_list, in_list, mol_buffer

def plot_figures(mol_buffer, log_dir):
    plot_avg_path = os.path.join(log_dir, 'top10_avg_score.png')
    plot_auc_path = os.path.join(log_dir, 'top10_auc_score.png')
    plot_combined_path = os.path.join(log_dir, 'top10_combined_scores.png')
    # Plotting logic here
    # Plot 1: TOP10_AVG
    plt.figure()
    plt.plot(range(1, len(top10_avg_score_list)+1), top10_avg_score_list, label='Top-10 Average Score')
    plt.xlabel('Number of Oracle Calls')
    plt.ylabel('Top-10 Average Score')
    plt.title('Top-10 Average Score Over Oracle Calls')
    plt.grid(True)
    plt.savefig(plot_avg_path)
    plt.close()

    # Plot 2: TOP10_AUC
    plt.figure()
    plt.plot(range(1, len(top10_auc_score_list)+1), top10_auc_score_list, label='Top-10 AUC Score', color='orange')
    plt.xlabel('Number of Oracle Calls')
    plt.ylabel('Top-10 AUC Score')
    plt.title('Top-10 AUC Score Over Oracle Calls')
    plt.grid(True)
    plt.savefig(plot_auc_path)
    plt.close()

    # Plot 3: Combined Figure
    plt.figure()
    plt.plot(range(1, len(top10_avg_score_list)+1), top10_avg_score_list, label='Top-10 Avg Score')
    plt.plot(range(1, len(top10_auc_score_list)+1), top10_auc_score_list, label='Top-10 AUC Score')
    plt.xlabel('Number of Oracle Calls')
    plt.ylabel('Score')
    plt.title('Top-10 Avg and AUC Scores Over Oracle Calls')
    plt.legend()
    plt.grid(True)
    plt.savefig(plot_combined_path)
    plt.close()

    print(f"Saved plot images:\n- {plot_avg_path}\n- {plot_auc_path}\n- {plot_combined_path}")
    pass



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

smiles_path_list = []
api = wandb.Api()
entire_top10_auc_df = pd.DataFrame(index=tasks, columns=projects)
# Loop over projects and tasks
for project in projects:
    runs = api.runs(f"icecream126/{project}")  # <-- CHANGE 'your-entity-name' to your wandb username/team
    for task in tasks:
        matching_runs = [run for run in runs if run.name == task]
        if not matching_runs:
            entire_top10_auc_df.at[task, project] = None
            continue
        
        run = matching_runs[0]  # Assume 1 run per task per project
        run_id = run.id

        # 1. Find corresponding folder
        log_folder = find_log_folder(run_id)
        if log_folder is None:
            print(f"Warning: Cannot find log folder for run_id {run_id}")
            entire_top10_auc_df.at[task, project] = None
            continue

        smiles_path = os.path.join(log_folder, "smiles.txt")
        smiles_path_list.append([smiles_path, project, task])


for obj in smiles_path_list:
    txt_path, project, task = obj
    print(f"\n\n== Project: {project}, Task: {task} ==")
    print(f"dir path: {txt_path.split('/')[-2]}")
    log_dir = txt_path[:-10]
    yaml_path = os.path.join(log_dir, 'results.yaml')

    smi_list, score_list, in_list, mol_buffer = convert_txt_to_results_yaml(txt_path, yaml_path)

    # Initialize DataFrame with columns
    df = pd.DataFrame(columns=['SMILES'])
    df['SMILES'] = smi_list
    df['SCORE'] = score_list
    df['IN'] = in_list

    top10_list = []
    top10_auc_score_list = []
    top10_avg_score_list = []
    ordered_results = list(sorted(mol_buffer.items(), key=lambda kv: kv[1][1]))

    for n_oracle in tqdm(range(1, len(smi_list)+1), desc="Calculating scores"):
        # print(f"== n_oracle: {n_oracle} ==")
        finish = n_oracle >=1000

        # Get mol_buffer until n_oracle
        temp_result = [item for item in ordered_results if item[1][1] <= n_oracle] # mol_buffer until n_oracle
        
        # Get top_10
        top10 = [item[1][0] for item in sorted(temp_result, key=lambda kv: kv[1][0], reverse=True)[:10]]
        top10_list.append(str(top10))
        
        # Get top10_avg_score
        top10_avg_score = np.mean(top10)
        top10_avg_score_list.append(round(top10_avg_score,3))
        # print(f"top10_avg_score: {top10_avg_score:.3f}")

        # Get top10_auc_score
        top10_auc_score = top_auc(mol_buffer, top_n=10, finish=finish, freq_log=1, max_oracle_calls=1000, buffer_max_idx=n_oracle)
        top10_auc_score_list.append(round(top10_auc_score,3))
        # print(f"top10_auc_score: {top10_auc_score:.3f}")
    
    entire_top10_auc_df.at[task, project] = top10_auc_score_list[-1]
    print(f"Final Top-10 AUC score: {top10_auc_score_list[-1]:.4f}")

    df['TOP10_AUC'] = top10_auc_score_list
    df['TOP10_AVG'] = top10_avg_score_list
    df['TOP10'] = top10_list
    df.to_csv(os.path.join(log_dir, 'results.csv'), index=False)
    print(f"results.csv saved in {log_dir}")

    plot_figures(mol_buffer, log_dir)

entire_top10_auc_df.to_csv(f"{projects[0]}.csv")
print(f"{projects[0]}.csv saved")