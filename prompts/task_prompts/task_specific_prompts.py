"""Task-specific prompts for different molecular design tasks"""

from typing import Dict, Any
from .base_templates import (
    get_base_scientist_prompt,
    get_base_scientist_prompt_with_review,
    get_base_reviewer_prompt,
    get_base_scientist_prompt_with_double_checker_review,
    get_base_double_checker_prompt
)

# Task-specific conditions and constraints
TASK_CONDITIONS = {
    "albuterol_similarity": """Design a molecule that is structurally similar to albuterol but with improved metabolic stability.
Key requirements:
1. Maintain the β2-adrenergic receptor binding pharmacophore
2. Replace the catechol moiety with a more metabolically stable group
3. Keep the tert-butyl group on the amine
4. Ensure drug-like properties (MW < 500, logP < 5)""",
    
    "valsartan_smarts": """Design a molecule that matches the SMARTS pattern of valsartan's key pharmacophore.
Key requirements:
1. Must contain the tetrazole ring
2. Must have the biphenyl scaffold
3. Must include the amide linkage
4. Should maintain similar physicochemical properties""",
    
    "logp_optimization": """Design a molecule with a target logP value between 2.5 and 3.5.
Key requirements:
1. Must be drug-like (MW < 500)
2. Should have good aqueous solubility
3. Must contain at least one aromatic ring
4. Should have balanced lipophilicity""",
    
    "qed_optimization": """Design a molecule with a QED score > 0.7.
Key requirements:
1. Must follow Lipinski's Rule of Five
2. Should have good drug-likeness properties
3. Must be synthetically accessible
4. Should have balanced physicochemical properties"""
}

# Task-specific functional group requirements
TASK_FUNCTIONAL_GROUPS = {
    "albuterol_similarity": ["hydroxyl", "amine", "aromatic_ring"],
    "valsartan_smarts": ["tetrazole", "amide", "aromatic_ring"],
    "logp_optimization": ["aromatic_ring", "hydrophobic_group"],
    "qed_optimization": ["aromatic_ring", "hydrogen_bond_donor", "hydrogen_bond_acceptor"]
}

def get_task_specific_prompt(task_name: str, prompt_type: str, **kwargs) -> str:
    """Get task-specific prompt based on task name and prompt type"""
    if task_name not in TASK_CONDITIONS:
        raise ValueError(f"Unknown task: {task_name}")
    
    # Get base template
    if prompt_type == "scientist":
        return get_base_scientist_prompt(
            task_name=task_name,
            task_condition=TASK_CONDITIONS[task_name],
            **kwargs
        )
    elif prompt_type == "scientist_with_review":
        return get_base_scientist_prompt_with_review(
            task_name=task_name,
            task_condition=TASK_CONDITIONS[task_name],
            **kwargs
        )
    elif prompt_type == "reviewer":
        return get_base_reviewer_prompt(
            task_name=task_name,
            task_condition=TASK_CONDITIONS[task_name],
            **kwargs
        )
    elif prompt_type == "scientist_with_double_checker":
        return get_base_scientist_prompt_with_double_checker_review(
            task_name=task_name,
            task_condition=TASK_CONDITIONS[task_name],
            **kwargs
        )
    elif prompt_type == "double_checker":
        return get_base_double_checker_prompt(
            task_name=task_name,
            task_condition=TASK_CONDITIONS[task_name],
            **kwargs
        )
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")

def get_task_functional_groups(task_name: str) -> list:
    """Get required functional groups for a specific task"""
    if task_name not in TASK_FUNCTIONAL_GROUPS:
        raise ValueError(f"Unknown task: {task_name}")
    return TASK_FUNCTIONAL_GROUPS[task_name] 