import wandb
import pandas as pd
import matplotlib.pyplot as plt
import os

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
    # "valsartan_smarts",
    "zaleplon_mpo"
]


def find_log_folder(run_id):
    logs_dir = "../logs"
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
    plt.plot(df1[keyword], label='v5', color='blue')
    plt.plot(df2[keyword], label='v4', color='orange')
    # Add vertical bold red line at x = 10
    if keyword == 'TOP10_AVG':
        plt.axvline(x=10, color='red', linewidth=2.5, linestyle='--', label='x = 10')
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
projects = ["pmo_v4_no_redundancy"]
# Loop over projects and tasks
for project in projects:
    runs = api.runs(f"icecream126/{project}")  # <-- CHANGE 'your-entity-name' to your wandb username/team
    for task in tasks:
        matching_runs = [run for run in runs if run.name == task]
        if not matching_runs:
            continue
        
        run = matching_runs[0]  # Assume 1 run per task per project
        run_id = run.id

        # 1. Find corresponding folder
        log_folder = find_log_folder(run_id)
        if log_folder is None:
            print(f"Warning: Cannot find log folder for run_id {run_id}")
            continue

        csv_path = os.path.join(log_folder, "results.csv")
        chemiloop_csv_path_dict[task] = csv_path



chemiloop_csv_path_dict2 = {}
projects = ["pmo_v5"]
# Loop over projects and tasks
for project in projects:
    runs = api.runs(f"icecream126/{project}")  # <-- CHANGE 'your-entity-name' to your wandb username/team
    
    for task in tasks:
        matching_runs = [run for run in runs if run.name == task]
        
        # run = matching_runs[0]  # Assume 1 run per task per project
        # import pdb; pdb.set_trace()
        print('task', task)
        run_id = matching_runs[0].id

        # 1. Find corresponding folder
        log_folder = find_log_folder(run_id)
        if log_folder is None:
            print(f"Warning: Cannot find log folder for run_id {run_id}")
            continue

        csv_path = os.path.join(log_folder, "results.csv")
        chemiloop_csv_path_dict2[task] = csv_path


logdir = "/home/khm/chemiloop/comparison_v4_v5_plots"
os.makedirs(logdir, exist_ok=True)
for task in tasks:
    molleo_csv = chemiloop_csv_path_dict2.get(task)
    chemiloop_csv = chemiloop_csv_path_dict.get(task)

    molleo_df = pd.read_csv(molleo_csv)
    chemiloop_df = pd.read_csv(chemiloop_csv)

    plot_df(keyword='TOP10_AUC', df1=molleo_df, df2=chemiloop_df, task=task, logdir = logdir)
    plot_df(keyword='TOP10_AVG', df1=molleo_df, df2=chemiloop_df, task=task, logdir = logdir)



