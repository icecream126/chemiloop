import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os

tasks = [
    # "albuterol_similarity",
    # "amlodipine_mpo",
    # "celecoxib_rediscovery",
    # "deco_hop",
    # "drd2",
    # "fexofenadine_mpo",
    # "gsk3b",
    # "isomers_c7h8n2o2",
    # "isomers_c9h10n2o2pf2cl",
    # "jnk3",
    "median1",
    "median2",
    # "mestranol_similarity",
    # "osimertinib_mpo",
    # "perindopril_mpo",
    # "qed",
    # "ranolazine_mpo",
    # "scaffold_hop",
    # "sitagliptin_mpo",
    # "thiothixene_rediscovery",
    # "troglitazon_rediscovery",
    # "valsartan_smarts",
    # "zaleplon_mpo"
]
projects = ["pmo_v4_no_redundancy"]

def find_log_folder(run_id):
    logs_dir = "./logs"
    for folder_name in os.listdir(logs_dir):
        if run_id in folder_name:
            return os.path.join(logs_dir, folder_name)
    return None
def plot_df(keyword, df1, df2, task, logdir):
    
    # Convert the relevant columns to numeric (in case of parsing issues)
    df1[keyword] = pd.to_numeric(df1[keyword], errors='coerce')
    df2[keyword] = pd.to_numeric(df2[keyword], errors='coerce')

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(df1[keyword], label='molleo', color='blue')
    plt.plot(df2[keyword], label='Ours', color='orange')
    plt.title(task)
    plt.xlabel('Step')
    plt.ylabel(keyword)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(f"{logdir}/{task}_{keyword}.png") 
    print(f"Saved plot for {task} with keyword {keyword} to {logdir}/{task}_{keyword}.png")




chemiloop_csv_path_dict = {}
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

        csv_path = os.path.join(log_folder, "results.csv")
        chemiloop_csv_path_dict[task] = csv_path

import pdb; pdb.set_trace()
molleo_csv_dir = "/home/khm/chemiloop/logs/molleo_deepseek"
molleo_csv_path_dict = {}
for task in tasks:
    for filename in os.listdir(molleo_csv_dir):
        if task in filename and filename.endswith('.csv'):
            molleo_csv_path_dict[task] = os.path.join(molleo_csv_dir, filename)
            break

logdir = "/home/khm/chemiloop/compare"
for task in tasks:
    molleo_csv = molleo_csv_path_dict.get(task)
    chemiloop_csv = chemiloop_csv_path_dict.get(task)

    molleo_df = pd.read_csv(molleo_csv)
    chemiloop_df = pd.read_csv(chemiloop_csv)

    plot_df(keyword='TOP10_AUC', df1=molleo_df, df2=chemiloop_df, task=task, logdir = logdir)
    plot_df(keyword='TOP10_AVG', df1=molleo_df, df2=chemiloop_df, task=task, logdir = logdir)



