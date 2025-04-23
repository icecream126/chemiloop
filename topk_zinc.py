import json

target_list = ['albuterol_similarity_score', 'isomer_c7h8n2o2_score', 'isomer_c9h10n2o2pf2cl_score']
for target in target_list:
    # Load the dataset
    with open("/home/khm/chemiloop/dataset/entire_zinc250.json", "r") as f:
        dataset = json.load(f)

    # Sort by albuterol_similarity_score in descending order and take top-5
    top_5 = sorted(dataset, key=lambda x: x.get(target, 0), reverse=True)[:5]

    # Save to new JSON file
    top_5_path = f"./dataset/entire_top_5/{target}.json"
    with open(top_5_path, "w") as f:
        json.dump(top_5, f, indent=2)

