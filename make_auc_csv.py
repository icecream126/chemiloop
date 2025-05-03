import wandb
import pandas as pd
import ast
import os

# Your lists
projects = ["pmo_v1", "pmo_v2", "pmo_v3", "pmo_v4"]
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

# Initialize wandb API
api = wandb.Api()

# Result table
result = pd.DataFrame(index=tasks, columns=projects)

# Helper to find correct log folder
def find_log_folder(run_id):
    logs_dir = "./logs"
    for folder_name in os.listdir(logs_dir):
        if run_id in folder_name:
            return os.path.join(logs_dir, folder_name)
    return None

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

        smiles_history_path = os.path.join(log_folder, "smiles_history.txt")

        if not os.path.exists(smiles_history_path):
            print(f"{project}: {task} not found")
            print(f"Warning: smiles_history.txt not found in {log_folder}")
            result.at[task, project] = None
            continue

        # 2. Read smiles history
        try:
            with open(smiles_history_path, "r") as f:
                lines = f.readlines()
                last_line = lines[-1]
                smiles_set = ast.literal_eval(last_line.strip())
                num_unique_smiles = len(smiles_set)
        except Exception as e:
            print(f"Error reading {smiles_history_path}: {e}")
            num_unique_smiles = 0
        
        # import pdb; pdb.set_trace()
        
        # 3. Fetch last top_10_avg_score_all
        history = run.history(keys=["top_10_avg_score_all"])
        if history.empty or "top_10_avg_score_all" not in history.columns:
            last_score = None
        else:
            top10_scores = history["top_10_avg_score_all"].dropna()
            last_score = top10_scores.iloc[-1] if not top10_scores.empty else None

        # import pdb; pdb.set_trace()
        
        # 4. Apply rule
        
        if last_score:
            if num_unique_smiles < 10:
                result.at[task, project] = "N/A"
            else:
                result.at[task, project] = round(last_score, 3)
        else:
            result.at[task, project] = "N/A"


# Save to CSV
result.to_csv("top10_avg_scores_from_smiles_history.csv")

print("Saved top10_avg_scores_from_smiles_history.csv successfully!")
