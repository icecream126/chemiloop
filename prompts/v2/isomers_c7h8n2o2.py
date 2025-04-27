
import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from guacamol.utils.chemistry import canonicalize
import utils.utils

    
def get_scientist_prompt(SMILES_HISTORY, topk_smiles):
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

def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, atom_counts, SMILES_HISTORY, topk_smiles):
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

def get_reviewer_prompt(scientist_think_dict, score, atom_counts):
    
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

def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback):
    return f"""Improve your previous generated SMILES based on detailed double-checker feedback.
Your original task:
Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.

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

def get_double_checker_prompt(thinking, improved_smiles):
    return f"""You will be given:
- A user prompt describing the target objective,
- The scientist’s reasoning broken into Step1 through Step4,
- The SMILES string proposed by the scientist.

Evaluate each step **independently**, comparing the described logic to the molecular structure in the SMILES. Provide a reasoning assessment for each step.

If all four steps are logically consistent with the final SMILES, mark "Consistency" as "Consistent".  
If **any** step is inconsistent, mark "Consistency" as "Inconsistent" and provide specific suggestions for improvement.

=== USER PROMPT === 
Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.
HARD CONSTRAINT (MUST follow exactly):  
The molecule must match this exact molecular formula:
- 7 Carbon atoms  
- 8 Hydrogen atoms  
- 2 Nitrogen atoms  
- 2 Oxygen atoms  
Any molecule not matching this formula is INVALID and will be discarded.

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
