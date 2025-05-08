import os
from tqdm import tqdm
import json

target_list = [
    'albuterol_similarity_score', 
    'isomer_c7h8n2o2_score', 
    'isomer_c9h10n2o2pf2cl_score',
    'celecoxib_rediscovery_score',
    'amlodipine_mpo_score',
    'deco_hop_score',
    'drd2_score',
    'fexofenadine_mpo_score',
    'gsk3b_score',
    'jnk3_score',
    'median1_score',
    'median2_score',
    'mestranol_similarity_score',
    'osimertinib_mpo_score',
    'perindopril_mpo_score',
    'qed_score',
    'ranolazine_mpo_score',
    'scaffold_hop_score',
    'sitagliptin_mpo_score',
    'thiothixene_rediscovery_score',
    'troglitazon_rediscovery_score',
    'valsartan_smarts_score',
    'zaleplon_mpo_score'
    ]
output_dir = "../dataset/250k_top_100"
os.makedirs(output_dir, exist_ok=True)

# Iterate with progress bar
for target in tqdm(target_list, desc="Processing targets"):
    # Load the dataset
    with open("/home/khm/chemiloop/dataset/entire_zinc250.json", "r") as f:
        dataset = json.load(f)

    # Sort by target score in descending order and take top-5
    top_100 = sorted(dataset, key=lambda x: x.get(target, 0), reverse=True)[:100]

    # Save to new JSON file
    top_100_path = os.path.join(output_dir, f"{target}.json")
    with open(top_100_path, "w") as f:
        json.dump(top_100, f, indent=2)

