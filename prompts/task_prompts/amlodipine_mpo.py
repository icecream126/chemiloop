import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from guacamol.utils.chemistry import canonicalize
import utils.utils

amlodipine_smiles = "Clc1ccccc1C2C(=C(/N/C(=C2/C(=O)OCC)COCCN)C)\C(=O)OC"
def get_scientist_prompt(topk_smiles):
    return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: 

Conditions:
- Achieve high structural similarity to amlodipine (SMILES: {amlodipine_smiles}).
- Preferably maintain around **3 rings** in the molecular structure to preserve desired complexity.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO AMLODIPINE: {amlodipine_smiles}.

You are provided with:
- Top-5 example molecules with high relevance to the task (you may use them as inspiration but do not copy them exactly).
- A list of previously generated SMILES (you MUST NOT repeat them).

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List the critical structural features of amlodipine (e.g., 'aryl group, ester groups, amine side chain, 3 ring system') and property goals (e.g., 'around 3 rings').",
  "step2": "Propose scaffold or substituent modifications to maintain similarity and the correct number of rings. Justify your changes chemically (e.g., 'Replace methyl group with ethyl to fine-tune lipophilicity').",
  "step3": "Describe the final designed molecule in natural language before giving the SMILES (e.g., 'A benzene ring fused to a six-membered ring with ester and amine groups attached.').",
  "SMILES": "Your valid SMILES string here"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, SMILES_HISTORY, topk_smiles):
    return f"""Task: Refine your molecule based on the reviewer's feedback.
    
ALSO, YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{SMILES_HISTORY}

Original goal:
- Design a molecule similar to amlodipine (SMILES: {amlodipine_smiles}).
- Maintain approximately 3 rings.

You must actively incorporate the reviewer's feedback into your redesign.

You are provided with:
- Top-5 relevant examples
- Previously generated SMILES (you must not repeat)
- Previous similarity score and functional groups

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

=== MOLECULE SMILES TO IMPROVE ===
MOLECULE SMILES: {previous_smiles}
- Amlodipine_mpo score: {score}
- Detected functional groups: 
{functional_groups}


=== YOUR PREVIOUS THINKING AND REVIEWER FEEDBACK ===
Step 1:
- Your thought: {scientist_think_dict["step1"]}
- Reviewer feedback: {reviewer_feedback_dict["step1"]}

Step 2:
- Your thought: {scientist_think_dict["step2"]}
- Reviewer feedback: {reviewer_feedback_dict["step2"]}

Step 3:
- Your thought: {scientist_think_dict["step3"]}
- Reviewer feedback: {reviewer_feedback_dict["step3"]}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Refined list of critical features and property targets for amlodipine rediscovery.",
  "step2": "Adjusted design strategy, ensuring high similarity and 3-ring system.",
  "step3": "Natural language description of the new structure before SMILES.",
  "SMILES": "Your corrected valid SMILES string"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s molecular design carefully for:

- Validity: Are proposed modifications chemically reasonable?
- Preservation: Are core features of amlodipine retained?
- Rediscovery goal: Is similarity high and are approximately 3 rings maintained?

Inputs for your evaluation:
- Scientist's step-by-step reasoning
- Final generated SMILES
- Tanimoto similarity score
- Detected functional groups

Amlodipine's core features include: aryl ring, ester functionalities, amine side chain, and 3-ring system.

=== SCIENTIST'S STEP-WISE THINKING ===
Step 1: {scientist_think_dict["step1"]}
Step 2: {scientist_think_dict["step2"]}
Step 3: {scientist_think_dict["step3"]}

=== SCIENTIST'S GENERATED SMILES ===
SMILES: {scientist_think_dict["smiles"]}
- Amlodipine_mpo score: {score}
- Detected functional groups:
{functional_groups}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Assess whether critical features were correctly identified. Mention missed or misinterpreted features.",
  "step2": "Evaluate if the design strategy successfully preserves similarity and ring count goals. Suggest better strategies if needed.",
  "step3": "Confirm if the final structure matches the scientist’s reasoning. Point out structural mismatches or missing elements."
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, SMILES_HISTORY):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{SMILES_HISTORY}

Improve your previous design based on double-checker feedback.

Original goal:
- Rediscover a molecule similar to Amlodipine {amlodipine_smiles}.
- Maintain approximately 3 rings and key functional features (aryl groups, ester, amine side chain).

Your previous thinking:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Previous SMILES that you need to improve:
{previous_smiles}

Double-checker feedback:
- Step1 Evaluation: {double_checker_feedback['step1']}
- Step2 Evaluation: {double_checker_feedback['step2']}
- Step3 Evaluation: {double_checker_feedback['step3']}

Now, revise your molecule thoughtfully.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List corrected and preserved features (e.g., ester group, amine group, 3-ring system).",
  "step2": "Propose improved modifications ensuring similarity and correct complexity. Justify each change chemically.",
  "step3": "Describe the final molecule in natural language, specifying rings, side chains, and functional groups.",
  "SMILES": "Your improved and chemically valid SMILES."
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_double_checker_prompt(thinking, improved_smiles):
    return f"""You will evaluate the improved molecule critically:

- Check if preserved features align with amlodipine’s scaffold.
- Check if chemical reasoning is consistent with the final SMILES.
- Confirm approximately 3 rings are maintained.

Inputs:
- User prompt
- Scientist's reasoning
- Final SMILES

Evluate each step independently.

=== SCIENTIST'S TASK ===
Scientist's task is to design a SMILES string for a molecule that satisfies the following conditions: 
- Achieve high structural similarity to amlodipine (SMILES: {amlodipine_smiles}).
- Preferably maintain around **3 rings** in the molecular structure to preserve desired complexity.

=== SCIENTIST'S THINKING ===
Step1: {thinking['step1']}
Step2: {thinking['step2']}
Step3: {thinking['step3']}

=== SCIENTIST'S SMILES ===
{improved_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Evaluate if the preserved features (e.g., 3 rings, ester groups) are correctly captured.",
  "step2": "Assess whether the design modifications are chemically sound and match the goal.",
  "step3": "Check if the SMILES structure truly reflects the intended design. Highlight any inconsistencies.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """
