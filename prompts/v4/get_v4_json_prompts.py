
import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from guacamol.utils.chemistry import canonicalize
import utils.utils

def get_user_prompt(task):
    if task == "albuterol":
        albuterol_smiles = 'CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O'
        albuterol_canonical_smiles = canonicalize(albuterol_smiles)
        albuterol_mol = Chem.MolFromSmiles(albuterol_smiles)
        albuterol_functional_group = utils.utils.describe_albuterol_features(albuterol_mol)
        return f""""Design a drug-like molecule structurally similar to albuterol (SMILES: {albuterol_smiles}, canonical: {albuterol_canonical_smiles}). Preserve the core scaffold and key functional groups. Albuterol contains: {albuterol_functional_group}."""
    else:
        print(f"No user prompt saved for this task: {task}.")
        return ""
def get_scientist_prompt(task, SMILES_HISTORY, topk_smiles):
  
  return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: {task}.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO ALBUTEROL, defined as:  
- SMILES: CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O  
- canonical SMILES: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1

You are provided with:
- Top-5 example molecules with high relevance to the task, listed below. You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.
- A list of previously generated SMILES, which YOU MUST NOT REPEAT.

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

Previously generated SMILES:
{SMILES_HISTORY}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  
```json
{{
  "step1": "List of the target’s critical structural/property features (e.g., 'Albuterol: phenyl ring, β-hydroxyamine, catechol-like substitution')\nIf property-based, specify requirements (e.g., "logP > 3: add hydrophobic groups").",
  "step2": "Propose modifications or scaffolds to meet the condition (e.g., 'Replace catechol with 3-hydroxy-4-pyridone').\n Justify each change chemically (e.g., "Maintains H-bonding but improves metabolic stability").",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., "A tert-butyl group attached to the amine (–NH–C(CH₃)₃) to mimic albuterol’s bulky substituent.")",
  "SMILES": "Your valid SMILES string here"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_scientist_prompt_isomers_c7h8n2o2(SMILES_HISTORY, topk_smiles):
  return f"""Task: Design a molecule in SMILES format that satisfies the following condition:

Objective: isomers_c7h8n2o2  
Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.  

You are provided with:
- Top-5 relevant SMILES examples with high relevance to the task, listed below. You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.
- A list of previously generated SMILES, which YOU MUST NOT REPEAT.

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

Previously generated SMILES:
{SMILES_HISTORY}

HARD CONSTRAINT (MUST follow exactly):  
The molecule must match this exact molecular formula:
- 7 Carbon atoms  
- 8 Hydrogen atoms  
- 2 Nitrogen atoms  
- 2 Oxygen atoms  
Any molecule not matching this formula is INVALID and will be discarded.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:
```json
{{
  "step1": "List possible structural motifs or fragments consistent with the formula C7H8N2O2.\n(e.g., "Common groups for C7H8N2O2: aromatic rings, nitro groups, amines, amides, phenols")",
  "step2": "Propose a valid isomer design strategy to maximize desired drug-like properties (e.g., "Maximize QED: incorporate a para-substituted aniline with hydrophilic groups").\nJustify each change chemically (e.g., "Adding a hydroxyl group improves hydrogen bonding, enhancing solubility and QED")",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES.\n(e.g., "A para-substituted phenyl ring bearing a hydroxyl and acetamide group to balance lipophilicity and polarity")",
  "SMILES": "Your valid SMILES here"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """


def get_scientist_prompt_with_review(task, scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, SMILES_HISTORY, topk_smiles):
    return f"""Task: Take reviewer's feedback actively and design a SMILES string for a molecule that satisfies the condition:\n"{task}".

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE THE SMILES IDENTICAL TO ALBUTEROL with :  
- SMILES: CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O
- canonical SMILES: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1

You are provided with:
- Top-5 example molecules with high relevance to the task, listed below. You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.
- A list of previously generated SMILES, which YOU MUST NOT REPEAT.

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

Previously generated SMILES:
{SMILES_HISTORY}

You will be provided with:
1. Previous SMILES string
2. Tanimoto similarity score (0–1) to albuterol CC(C)(C)NCC(O)c1ccc(O)c(CO)c1 based on canonical SMILES
3. Detected functional groups in your previous molecule 

--- PREVIOUS MOLECULE SMILES ---
SMILES: {previous_smiles}

--- SIMILARITY SCORE (Tanimoto) ---
Score: {score} (0–1)

--- FUNCTIONAL GROUPS DETECTED ---
{functional_groups}

--- YOUR PREVIOUS THOUGHT AND REVIEWER'S FEEDBACK ---
Step1: List Key Features

Your previous thought process:\n{scientist_think_dict["step1"]}

Accordingly, reviewer's feedback is:\n{reviewer_feedback_dict["step1"]}

Step2: Design Strategy:

Your previous thought process:\n{scientist_think_dict["step2"]}

Accordingly, reviewer's feedback is:\n{reviewer_feedback_dict["step2"]}

Step 3: Construct the Molecule:

Your previous thought process:\n{scientist_think_dict["step3"]}

Accordingly, reviewer's feedback is:\n{reviewer_feedback_dict["step3"]}

Now based on yoru prevous thoughts and the reviewer's feedback, you need to improve your design.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  
```json
{{
  "step1": "List of the target’s critical structural/property features (e.g., 'Albuterol: phenyl ring, β-hydroxyamine, catechol-like substitution')\nIf property-based, specify requirements (e.g., "logP > 3: add hydrophobic groups").",
  "step2": "Propose modifications or scaffolds to meet the condition (e.g., 'Replace catechol with 3-hydroxy-4-pyridone').\n Justify each change chemically (e.g., "Maintains H-bonding but improves metabolic stability").",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., "A tert-butyl group attached to the amine (–NH–C(CH₃)₃) to mimic albuterol’s bulky substituent.")",
  "SMILES": "Your valid SMILES string here"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """


def get_scientist_prompt_with_review_isomers_c7h8n2o2(task, scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, atom_counts, SMILES_HISTORY, topk_smiles):
    return f"""Task: Design an improved molecule in SMILES format that satisfies the following condition:

Objective: isomers_c7h8n2o2  
Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.  

You are provided with:
- Top-5 example molecules with high relevance to the task, listed below. You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.
- A list of previously generated SMILES, which YOU MUST NOT REPEAT.

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

Previously generated SMILES:
{SMILES_HISTORY}

HARD CONSTRAINT (MUST follow exactly):  
The molecule must match this exact molecular formula:
- 7 Carbon atoms  
- 8 Hydrogen atoms  
- 2 Nitrogen atoms  
- 2 Oxygen atoms  
Any molecule not matching this formula is INVALID and will be discarded.

You will be provided with:  
1. Your previous molecule and thought process  
2. The isomer score (0–1) indicating how well your molecule fits the target formula  
3. Atom counts of the target and previous molecule  
4. Feedback from a chemistry reviewer

--- PREVIOUS MOLECULE SMILES ---  
SMILES: {previous_smiles}

--- ISOMER SCORE ---  
Score: {score} (0–1)

--- ATOM COUNTS (TARGET MOLECULE) ---
- C: 7
- H: 8
- N: 2
- O: 2

--- ATOM COUNTS (PREVIOUS MOLECULE) ---  
{atom_counts}

Now you will get your previous thought and reviewer's feedback.
--- STEP 1: KEY FEATURES ---  
Your previous thought:\n{scientist_think_dict["step1"]}  

Reviewer’s feedback:\n{reviewer_feedback_dict["step1"]}  

--- STEP 2: DESIGN STRATEGY ---  
Your previous thought:\n{scientist_think_dict["step2"]}  

Reviewer’s feedback:\n{reviewer_feedback_dict["step2"]}  

--- STEP 3: CONSTRUCT THE MOLECULE ---  
Your previous thought:\n{scientist_think_dict["step3"]}  

Reviewer’s feedback:\n{reviewer_feedback_dict["step3"]}  

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  
```json
{{
  "step1": "List possible structural motifs or fragments consistent with the formula C7H8N2O2.\n(e.g., "Common groups for C7H8N2O2: aromatic rings, nitro groups, amines, amides, phenols")",
  "step2": "Propose a valid isomer design strategy to maximize desired drug-like properties (e.g., "Maximize QED: incorporate a para-substituted aniline with hydrophilic groups").\nJustify each change chemically (e.g., "Adding a hydroxyl group improves hydrogen bonding, enhancing solubility and QED")",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES.\n(e.g., "A para-substituted phenyl ring bearing a hydroxyl and acetamide group to balance lipophilicity and polarity")",
  "SMILES": "Your valid SMILES here"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition

Be constructive: Provide fixes for issues (e.g., "Replace C=O=C with O=C=O for carbon dioxide").

You are provided with:
1. The scientist's step-wise reasoning.
2. The final generated SMILES.
3. The Tanimoto similarity score to albuterol: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1
4. The detected functional groups in the generated molecule.

--- SCIENTIST'S STEP-WISE THINKING ---
Step 1: {scientist_think_dict["step1"]}

Step 2: {scientist_think_dict["step2"]}

Step 3: {scientist_think_dict["step3"]}

--- FINAL MOLECULE SMILES ---
SMILES: {scientist_think_dict["smiles"]}

--- TANIMOTO SIMILARITY SCORE ---
Score: {score} (range: 0 to 1)

--- FUNCTIONAL GROUPS DETECTED ---
{functional_groups}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  

```json
{{
  "step1": "List accurate features and functional groups identified.\nMention any critical features and functional groups that were missed or misinterpreted.",
  "step2": "Evaluate if the proposed design strategy aligns with the structural and functional similarity goal.\nComment on whether the design aligns with the initial objectives.\nSuggest improvements or alternatives if needed.",
  "step3": "Review the structural construction and positional assignments.\nCheck for missing elements or mismatches in reasoning. (e.g., "Claimed ‘para hydroxyl’ but SMILES places it meta")",
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_reviewer_prompt_isomers_c7h8n2o2(scientist_think_dict, score, atom_counts):
    
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Chemical soundness
- Valid isomer structure
- Adherence to the formula constraint: C7H8N2O2

Be constructive: Provide precise and actionable feedback  
(e.g., "Replace the nitro group with an amide to maintain the N and O count.").

You are provided with:
1. The step-wise reasoning used to design the molecule
2. The final generated SMILES string
3. The isomer score (0–1), which reflects how well the molecular formula matches C7H8N2O2
4. Atom counts for the target and generated molecule

--- SCIENTIST'S STEP-WISE THINKING ---  
Step 1: {scientist_think_dict["step1"]}

Step 2: {scientist_think_dict["step2"]}

Step 3: {scientist_think_dict["step3"]}

--- FINAL MOLECULE SMILES ---  
SMILES: {scientist_think_dict["smiles"]}

--- ISOMER SCORE ---  
Score: {score} (range: 0–1)  

--- ATOM COUNTS (TARGET) ---
- C: 7
- H: 8
- N: 2
- O: 2

--- ATOM COUNTS (GENERATED) ---  
{atom_counts}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  

```json
{{
  "step1": "List chemically plausible substructures mentioned in the reasoning.\nPoint out any inaccurate or missing motifs with respect to C7H8N2O2.",
  "step2": "Evaluate whether the design strategy aligns with the goal of optimizing a valid isomer of C7H8N2O2.\nComment on whether the chosen strategy satisfies the desired atom counts.\nSuggest structural alternatives if any atoms are misallocated.",
  "step3": "Verify that the described structure corresponds accurately to the SMILES string.\nFlag inconsistencies (e.g., "Mentioned amide linkage, but none is present in SMILES").\nEnsure that the final SMILES does not violate the atomic formula constraint."
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_double_checker_prompt_json(user_prompt, thinking, improved_smiles):
    return f"""You will be given:
- A user prompt describing the target objective,
- The scientist’s reasoning broken into Step1 through Step4,
- The SMILES string proposed by the scientist.

Evaluate each step **independently**, comparing the described logic to the molecular structure in the SMILES. Provide a reasoning assessment for each step.

If all four steps are logically consistent with the final SMILES, mark "Consistency" as "Consistent".  
If **any** step is inconsistent, mark "Consistency" as "Inconsistent" and provide specific suggestions for improvement.

=== USER PROMPT === 
{user_prompt}

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

Use the following format:  
```json
{{
  "step1": "Your analysis of whether scientist's Step1 thinking is chemically valid and  reflected in the SMILES.",
  "step2": "Your analysis of whether scientist's Step2 thinking is chemically valid and  reflected in the SMILES.",
  "step3": "Your analysis of whether scientist's Step3 thinking is chemically valid and reflected in the SMILES.",
  "consistency": "Consistent" or "Inconsistent",
}}

```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

# def get_scientist_prompt_with_double_checker_review(task, scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, SMILES_HISTORY, topk_smiles):
  
def get_scientist_prompt_with_double_checker_review(user_prompt, previous_thinking, previous_smiles, double_checker_feedback):
    return f"""Improve your previous generated SMILES based on detailed double-checker feedback.
Your original task:
{user_prompt}

Your previous reasoning steps were:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Your previously generated SMILES:
{previous_smiles}

The double-checker reviewed each of your steps and gave the following evaluations:

- Step1_Evaluation: {double_checker_feedback['step1']}
- Step2_Evaluation: {double_checker_feedback['step2']}
- Step3_Evaluation: {double_checker_feedback['step3']}

Now, based on your original reasoning and the above feedback, revise your thinking and generate an improved SMILES string that better aligns with your design logic.


You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  
```json
{{
  "step1": "List of the target’s critical structural/property features (e.g., 'Albuterol: phenyl ring, β-hydroxyamine, catechol-like substitution')\nIf property-based, specify requirements (e.g., "logP > 3: add hydrophobic groups").",
  "step2": "Propose modifications or scaffolds to meet the condition (e.g., 'Replace catechol with 3-hydroxy-4-pyridone').\n Justify each change chemically (e.g., "Maintains H-bonding but improves metabolic stability").",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., "A tert-butyl group attached to the amine (–NH–C(CH₃)₃) to mimic albuterol’s bulky substituent.")",
  "SMILES": "Your improved and valid SMILES string here"
}}
```
IF YOU DO NOT FOLLOW THIS EXACT FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_scientist_prompt_with_double_checker_review_isomers_c7g8n2o2(user_prompt, previous_thinking, previous_smiles, double_checker_feedback):
    return f"""Improve your previous generated SMILES based on detailed double-checker feedback.
Your original task:
{user_prompt}

Your previous reasoning steps were:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Your previously generated SMILES:
{previous_smiles}

The double-checker reviewed each of your steps and gave the following evaluations:

- Step1_Evaluation: {double_checker_feedback['step1']}
- Step2_Evaluation: {double_checker_feedback['step2']}
- Step3_Evaluation: {double_checker_feedback['step3']}

Now, based on your original reasoning and the above feedback, revise your thinking and generate an improved SMILES string that better aligns with your design logic.


You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  
```json
{{
  "step1": "List possible structural motifs or fragments consistent with the formula C7H8N2O2.\n(e.g., "Common groups for C7H8N2O2: aromatic rings, nitro groups, amines, amides, phenols")",
  "step2": "Propose a valid isomer design strategy to maximize desired drug-like properties (e.g., "Maximize QED: incorporate a para-substituted aniline with hydrophilic groups").\nJustify each change chemically (e.g., "Adding a hydroxyl group improves hydrogen bonding, enhancing solubility and QED")",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES.\n(e.g., "A para-substituted phenyl ring bearing a hydroxyl and acetamide group to balance lipophilicity and polarity")",
  "SMILES": "Your valid SMILES here"
}}
```
IF YOU DO NOT FOLLOW THIS EXACT FORMAT, INNOCENT PEOPLE WILL DIE. """