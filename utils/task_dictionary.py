# Mapping for retrieval_node datasets
TASK_TO_DATASET_PATH = {
    "albuterol_similarity": "/home/khm/chemiloop/dataset/entire_top_5/albuterol_similarity_score.json",
    "isomers_c7h8n2o2": "/home/khm/chemiloop/dataset/entire_top_5/isomer_c7h8n2o2_score.json",
    "isomers_c9h10n2o2pf2cl": "/home/khm/chemiloop/dataset/entire_top_5/isomer_c9h10n2o2pf2cl_score.json",
}

# Mapping for scoring functions
TASK_TO_SCORING_FUNCTION = {
    "albuterol_similarity": get_albuterol_similarity_score,
    "isomers_c7h8n2o2": get_isomer_c7h8n2o2_score,
    "isomers_c9h10n2o2pf2cl": get_isomer_c9h10n2o2pf2cl_score,
}

# Mapping for scientist prompt functions
TASK_TO_SCIENTIST_PROMPT = {
    "albuterol_similarity": prompts.v4.get_v4_json_prompts.get_scientist_prompt_isomers_c7h8n2o2,
    "isomers_c7h8n2o2": prompts.v4.get_v4_json_prompts.get_scientist_prompt_isomers_c7h8n2o2,
}

# Mapping for scientist prompt with reviewer
TASK_TO_SCIENTIST_PROMPT_WITH_REVIEW = {
    "albuterol_similarity": prompts.v4.get_v4_json_prompts.get_scientist_prompt_with_review,
    "isomers_c7h8n2o2": prompts.v4.get_v4_json_prompts.get_scientist_prompt_with_review_isomers_c7h8n2o2,
}

# Mapping for reviewer prompt
TASK_TO_REVIEWER_PROMPT = {
    "albuterol_similarity": prompts.v4.get_v4_json_prompts.get_reviewer_prompt,
    "isomers_c7h8n2o2": prompts.v4.get_v4_json_prompts.get_reviewer_prompt_isomers_c7h8n2o2,
}

# Mapping for scientist prompt with double checker
TASK_TO_SCIENTIST_PROMPT_WITH_DOUBLE_CHECKER = {
    "albuterol_similarity": prompts.v4.get_v4_json_prompts.get_scientist_prompt_with_double_checker_review,
    "isomers_c7h8n2o2": prompts.v4.get_v4_json_prompts.get_scientist_prompt_with_double_checker_review_isomers_c7g8n2o2,
}