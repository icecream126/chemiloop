def get_scientist_prompt(task):
    return f"""You are a skilled chemist.

Task: Design a SMILES string for a molecule that satisfies the condition:\n"{task}".

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE THE SMILES IDENTICAL TO ALBUTEROL with :  
- SMILES: CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O
- canonical SMILES: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1

FOLLOW THIS EXACT FORMAT. DO NOT ADD ANYTHING ELSE. IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE:

Step1: List Key Features

List the target’s critical structural/property features (e.g., "Albuterol: phenyl ring, β-hydroxyamine, catechol-like substitution").

If property-based, specify requirements (e.g., "logP > 3: add hydrophobic groups").

Step2: Design Strategy:

Propose modifications or scaffolds to meet the condition (e.g., "Replace catechol with bioisostere 3-hydroxy-4-pyridone").

Justify each change chemically (e.g., "Maintains H-bonding but improves metabolic stability").

Step3: Construct the Molecule:

Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., "A tert-butyl group attached to the amine (–NH–C(CH₃)₃) to mimic albuterol’s bulky substituent.")

Final Output:
SMILES: [Your valid SMILES here]"""

def get_scientist_prompt_isomers_c7h8n2o2(task):
    return f"""You are a skilled chemist.
Task: Design a molecule in SMILES format that satisfies the following condition:

Objective: isomers_c7h8n2o2
Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.
Generate the SMILES that include H atoms.

HARD CONSTRAINT (MUST follow exactly):  
The molecule must match this exact molecular formula:
- 7 Carbon atoms  
- 8 Hydrogen atoms  
- 2 Nitrogen atoms  
- 2 Oxygen atoms  
Any molecule not matching this formula is INVALID and will be discarded.

FOLLOW THIS EXACT FORMAT. DO NOT ADD ANYTHING ELSE. IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE:

Step1: List Key Features

List possible structural motifs or fragments consistent with the formula C7H8N2O2.  
(e.g., "Common groups for C7H8N2O2: aromatic rings, nitro groups, amines, amides, phenols")

Step2: Design Strategy:

Propose a valid isomer design strategy to maximize desired drug-like properties (e.g., "Maximize QED: incorporate a para-substituted aniline with hydrophilic groups")

Justify each change chemically (e.g., "Adding a hydroxyl group improves hydrogen bonding, enhancing solubility and QED")

Step3: Construct the Molecule:

Describe the full structure of your designed molecule in natural language before writing the SMILES.  
(e.g., "A para-substituted phenyl ring bearing a hydroxyl and acetamide group to balance lipophilicity and polarity")

Final Output:
SMILES: [Your valid SMILES here including H atoms]"""


def get_scientist_prompt_with_review(task, scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups):
    return f"""You are a skilled chemist.
Task: Take reviewer's feedback actively and design a SMILES string for a molecule that satisfies the condition:\n"{task}".

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE THE SMILES IDENTICAL TO ALBUTEROL with :  
- SMILES: CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O
- canonical SMILES: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1

You will be provided with:
1. Previous SMILES string
2. Tanimoto similarity score (0–1) to albuterol CC(C)(C)NCC(O)c1ccc(O)c(CO)c1 based on canonical SMILES
2. Detected functional groups in your previous molecule 

--- PREVIOUS MOLECULE SMILES ---
SMILES: {previous_smiles}

--- SIMILARITY SCORE (Tanimoto) ---
Score: {score} (0–1)

--- FUNCTIONAL GROUPS DETECTED ---
{functional_groups}

FOLLOW THIS EXACT FORMAT. DO NOT ADD ANYTHING ELSE. IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE:

Step1: List Key Features

Your previous thought process:\n{scientist_think_dict["step1"]}

Accordingly, reviewer's feedback is:\n{reviewer_feedback_dict["step1"]}

Write your improved thought by actively taking the reviewer's feedback into account.

Step2: Design Strategy:

Your previous thought process:\n{scientist_think_dict["step2"]}

Accordingly, reviewer's feedback is:\n{reviewer_feedback_dict["step2"]}

Write your improved thought by actively taking the reviewer's feedback into account.

Step 3: Construct the Molecule:

Your previous thought process:\n{scientist_think_dict["step3"]}

Accordingly, reviewer's feedback is:\n{reviewer_feedback_dict["step3"]}

Write your improved thought by actively taking the reviewer's feedback into account.

Final Output:
SMILES: [Your valid and improved SMILES here]"""


def get_scientist_prompt_with_review_isomers_c7h8n2o2(task, scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, atom_counts):
    return f"""You are a skilled chemist.  
Task: Design a improved molecule in SMILES format that satisfies the following condition:

Objective: isomers_c7h8n2o2
Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.
Generate the SMILES that include H atoms.

HARD CONSTRAINT (MUST follow exactly):  
The molecule must match this exact molecular formula:
- 7 Carbon atoms  
- 8 Hydrogen atoms  
- 2 Nitrogen atoms  
- 2 Oxygen atoms  
Any molecule not matching this formula is INVALID and will be discarded.

You will be provided with:  
1. Previous SMILES string  
2. Isomer score (0–1) based on closeness to molecular formula  
3. Atom counts of the target molecule C7H8N2O2
4. Atom counts in your previous molecule  

--- PREVIOUS MOLECULE SMILES ---  
SMILES: {previous_smiles}

--- ISOMERS SCORE ---  
Score: {score} (0–1)

--- ATOM COUNTS (TARGET MOLECULE) ---
- C: 7
- H: 8
- N: 2
- O: 2

--- ATOM COUNTS (PREVIOUS MOLECULE) ---  
{atom_counts}

FOLLOW THIS EXACT FORMAT. DO NOT ADD ANYTHING ELSE. IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE:

Step1: List Key Features

Your previous thought process:\n{scientist_think_dict["step1"]}

Accordingly, reviewer's feedback is:\n{reviewer_feedback_dict["step1"]}

Write your improved thought by actively taking the reviewer's feedback into account.

Step2: Design Strategy:

Your previous thought process:\n{scientist_think_dict["step2"]}

Accordingly, reviewer's feedback is:\n{reviewer_feedback_dict["step2"]}

Write your improved thought by actively taking the reviewer's feedback into account.

Step 3: Construct the Molecule:

Your previous thought process:\n{scientist_think_dict["step3"]}

Accordingly, reviewer's feedback is:\n{reviewer_feedback_dict["step3"]}

Write your improved thought by actively taking the reviewer's feedback into account.

Final Output:
SMILES: [Your valid and improved SMILES here]"""

def get_reviewer_prompt(task, scientist_think_dict, score, functional_groups):
    return f"""Role: You are a rigorous chemistry reviewer. Evaluate the Scientist LLM’s reasoning steps and  final SMILES(molecule) for validity, chemical soundness, and adherence to the design condition.

Rules:
Be constructive: Provide fixes for errors (e.g., "Replace C=O=C with O=C=O for carbon dioxide").

You will be provided with:
1. The step-wise reasoning used to design the molecule.
2. The final generated SMILES string.
3. The Tanimoto similarity score (0–1) to albuterol CC(C)(C)NCC(O)c1ccc(O)c(CO)c1 based on canonical SMILES.
4. The detected functional groups in the generated molecule.

--- SCIENTIST'S STEP-WISE THINKING ---
{scientist_think_dict["step1"]}

{scientist_think_dict["step2"]}

{scientist_think_dict["step3"]}

--- FINAL MOLECULE SMILES ---
SMILES: {scientist_think_dict["SMILES"]}

--- SIMILARITY SCORE (Tanimoto) ---
Score: {score} (0–1)

--- FUNCTIONAL GROUPS DETECTED ---
{functional_groups}

--- REVIEW INSTRUCTIONS ---
FOLLOW THIS EXACT FORMAT. DO NOT ADD ANYTHING ELSE. IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE:

Step 1 Feedback:  
- [List accurate features and functional groups identified.]  
- [Mention any critical features and functional groups that were missed or misinterpreted.]

Step 2 Feedback:  
- [Evaluate if the proposed design strategy aligns with the structural and functional similarity goal.  ]
- [Comment on whether the design aligns with the initial objectives.]  
- [Suggest improvements or alternatives if needed.]

Step 3 Feedback:  
- [Review the structural construction and positional assignments.]  
- [Check for missing elements or mismatches in reasoning. (e.g., "Claimed ‘para hydroxyl’ but SMILES places it meta")]"""

def get_reviewer_prompt_isomers_c7h8n2o2(task, scientist_think_dict, score, atom_counts):
    return f"""Role: You are a rigorous chemistry reviewer. Evaluate the Scientist LLM’s reasoning steps and  final SMILES(molecule) for validity, chemical soundness, and adherence to the design condition.

Rules:  
Be constructive: Provide precise feedback with actionable fixes  
(e.g., "Replace the nitro group with an amide to maintain the N and O count.").

You will be provided with:  
1. The step-wise reasoning used to design the molecule
2. The final generated SMILES string
3. The isomer score (0–1), which measures how well the molecular formula matches C7H8N2O2
4. Atom counts of the target molecule (C7H8N2O2)
4. Atom counts of the generated molecule

--- SCIENTIST'S STEP-WISE THINKING ---  
{scientist_think_dict["step1"]}

{scientist_think_dict["step2"]}

{scientist_think_dict["step3"]}

--- FINAL MOLECULE SMILES ---  
SMILES: {scientist_think_dict["SMILES"]}

--- ISOMER SCORE ---  
Score: {score} (0–1)  


--- ATOM COUNTS (TARGET MOLECULE) ---
- C: 7
- H: 8
- N: 2
- O: 2

--- ATOM COUNTS (GENERATED MOLECULE) ---  
{atom_counts}

--- REVIEW INSTRUCTIONS ---  
FOLLOW THIS EXACT FORMAT. DO NOT ADD ANYTHING ELSE. IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE:

Step 1 Feedback:  
- [List chemically plausible substructures mentioned in the reasoning.]  
- [Point out any inaccurate or missing motifs with respect to C7H8N2O2.]

Step 2 Feedback:  
- [Evaluate whether the design strategy aligns with the goal of optimizing a valid isomer of C7H8N2O2.]  
- [Comment on whether the chosen strategy satisfies the desired atom counts.]  
- [Suggest structural alternatives if any atoms are misallocated.]

Step 3 Feedback:  
- [Verify that the described structure corresponds accurately to the SMILES string.]  
- [Flag inconsistencies (e.g., "Mentioned amide linkage, but none is present in SMILES").]  
- [Ensure that the final SMILES does not violate the atomic formula constraint.]"""