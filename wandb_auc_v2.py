import wandb
import numpy as np
from typing import List

def compute_auc_topk_online(scores: List[float], k: int) -> float:
    """
    Compute AUC of top-K average vs number of oracle calls.
    """
    sum_auc = 0
    prev_topk_avg = 0
    for step in range(1, len(scores) + 1):
        top_k_scores = sorted(scores[:step], reverse=True)[:k]
        topk_avg = sum(top_k_scores) / k
        sum_auc += (topk_avg + prev_topk_avg) / 2
        prev_topk_avg = topk_avg
    return sum_auc / len(scores)

def fetch_scores_from_wandb(project: str, run_id: str) -> List[float]:
    """
    Connect to wandb and fetch the 'score' values from a specific run.
    """
    api = wandb.Api()
    run = api.run(f"{project}/{run_id}")
    history = run.history(keys=["score"])
    scores = history["score"].dropna().tolist()
    return scores

def analyze_multiple_runs(project_runs):
    for entry in project_runs:
        project = entry["project_name"]
        run_ids = entry["run_id"]

        # Step 1: Fetch all scores first and find minimum length
        all_scores_dict = {}
        min_len = float('inf')
        for run_id in run_ids:
            scores = fetch_scores_from_wandb(project, run_id)
            all_scores_dict[run_id] = scores
            min_len = min(min_len, len(scores))

        # Step 2: Truncate all scores to minimum length and calculate AUCs
        auc_top1_all_list = []
        auc_top10_all_list = []
        auc_top1_no1_list = []
        auc_top10_no1_list = []
        min_len = 22
        for run_id, scores_all in all_scores_dict.items():
            scores_all = scores_all[:min_len]
            scores_no_1 = [s for s in scores_all if s < 1.0]

            auc_top1_all = compute_auc_topk_online(scores_all, k=1)
            auc_top10_all = compute_auc_topk_online(scores_all, k=10)

            auc_top1_no1 = compute_auc_topk_online(scores_no_1, k=1) if len(scores_no_1) >= 1 else np.nan
            auc_top10_no1 = compute_auc_topk_online(scores_no_1, k=10) if len(scores_no_1) >= 10 else np.nan

            auc_top1_all_list.append(auc_top1_all)
            auc_top10_all_list.append(auc_top10_all)
            auc_top1_no1_list.append(auc_top1_no1)
            auc_top10_no1_list.append(auc_top10_no1)

        print(f"\n=== Project: {project} ===")
        print(f"Used truncated length: {min_len}")
        # print(f"Minimum length of scores for project : {min_len}")
        print(f"Top-1 AUC (with 1.0):  {np.mean(auc_top1_all_list):.4f} ± {np.std(auc_top1_all_list):.4f}")
        print(f"Top-10 AUC (with 1.0): {np.mean(auc_top10_all_list):.4f} ± {np.std(auc_top10_all_list):.4f}")
        print(f"Top-1 AUC (no 1.0):    {np.nanmean(auc_top1_no1_list):.4f} ± {np.nanstd(auc_top1_no1_list):.4f}")
        print(f"Top-10 AUC (no 1.0):   {np.nanmean(auc_top10_no1_list):.4f} ± {np.nanstd(auc_top10_no1_list):.4f}")

# === EXAMPLE RUN LIST ===
project_runs = [
    # {"project_name": "icecream126/1000_pmo_v1_albutero_smilarity", "run_id": ["d9omae9g", "cgbrp338", "uc32dvr8"]},
    # {"project_name": "icecream126/1000_pmo_v2_albutero_smilarity", "run_id": ["txfjl22e", "937udx93", "aa9p93n3"]},
    # {"project_name": "icecream126/1000_pmo_v2_isomers_c7h8n2o2", "run_id": ["viyq8nq8", "j8g11cb4", "yqj6n583"]},
    {"project_name": "icecream126/1000_pmo_v1_isomers_c7h8n2o2", "run_id": ["mvkv1ur0", "gli844lj", "lnnfkqas"]},
    # {"project_name": "icecream126/pmo_v1_albutero_smilarity_with_fg", "run_id": ["zgprqwq9", "1irg0fvd", "2k3wmatm", "xffn2cri", "w5pzmjn0"]},
    # {"project_name": "icecream126/pmo_v2_albutero_smilarity", "run_id": ["dpnpd67u", "txuifbaa", "n8k5vxg3", "1g4073vv", "myei7kzv"]},
    # {"project_name": "icecream126/pmo_v1_isomers_c7h8n2o2", "run_id": ["5rmvrhc3", "1p6y7eec", "vc7lx41n", "v16aos82", "stx2zup4"]},
    # {"project_name": "icecream126/pmo_v2_isomers_c7h8n2o2", "run_id": ["zrcn2m25", "0h3azucn", "0jb1ate4", "0jt6ik6x", "0pcw672k"]},
]

# === RUN ANALYSIS ===
analyze_multiple_runs(project_runs)
