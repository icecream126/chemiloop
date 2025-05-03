def get_scientist_prompt(SMILES_HISTORY, topk_smiles):
    return f"""Previously generated SMILES. YOU MUST NOT REPEAT ANY OF THEM:
{SMILES_HISTORY}

Your task is to design a SMILES string for a molecule that satisfies the following conditions:
- High structural similarity to osimertinib (SMILES: "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34").
- Achieve a Topological Polar Surface Area (TPSA) close to **100**.
- Maintain a low-to-moderate lipophilicity (LogP ≈ **1**).

IMPORTANT CONSTRAINT:  
YOU MUST NOT generate a molecule identical to osimertinib.

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "List key pharmacophoric and structural features of osimertinib (e.g., quinazoline core, methoxy group, tertiary amines, multiple aromatic systems).",
  "step2": "Propose molecular modifications or scaffold tuning strategies to balance similarity, TPSA (~100), and LogP (~1). Provide chemical justifications.",
  "step3": "Describe your designed structure naturally before providing the SMILES (e.g., 'A quinazoline-based scaffold linked to multiple amine substituents for reduced lipophilicity.').",
  "SMILES": "Your valid SMILES string here"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, SMILES_HISTORY, topk_smiles):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{SMILES_HISTORY}

Your task is to design a SMILES string for a molecule that satisfies the following conditions:
- High structural similarity to osimertinib (SMILES: "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34").
- Achieve a Topological Polar Surface Area (TPSA) close to **100**.
- Maintain a low-to-moderate lipophilicity (LogP ≈ **1**).

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You will also see:
1. Molecule SMILES to improve
2. Its osimertinib_mpo score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---  
MOLECULE SMILES: {previous_smiles}
- osimertinib_mpo score: {score}
- Detected functional groups:
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

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Updated structural features and property targets (TPSA ≈ 100, LogP ≈ 1).",
  "step2": "Refined molecular design strategy balancing similarity, polarity, and hydrophobicity.",
  "step3": "Natural description of the improved structure before the SMILES.",
  "SMILES": "Your corrected SMILES string"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
Design a SMILES string for a molecule that satisfies the following conditions:
- High structural similarity to osimertinib (SMILES: "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34").
- Achieve a Topological Polar Surface Area (TPSA) close to **100**.
- Maintain a low-to-moderate lipophilicity (LogP ≈ **1**).

IMPORTANT CONSTRAINT:  
MUST NOT generate a molecule identical to osimertinib.

Provided:
1. Scientist's step-wise thinking
2. Scientist-generated SMILES
3. osimertinib_mpo score
4. Detected Atom counts

--- SCIENTIST'S STEP-WISE THINKING ---
Step1: {scientist_think_dict["step1"]}
Step2: {scientist_think_dict["step2"]}
Step3: {scientist_think_dict["step3"]}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict["smiles"]}
- osimertinib_mpo score: {score}
- Functional groups detected:
{functional_groups}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Check if key structural features and properties were correctly identified.",
  "step2": "Evaluate if the design strategy is optimal for TPSA and LogP goals.",
  "step3": "Confirm consistency between reasoning and final SMILES structure."
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, SMILES_HISTORY):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{SMILES_HISTORY}

Your task is to design a SMILES string for a molecule that satisfies the following conditions:
- High structural similarity to osimertinib (SMILES: "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34").
- Achieve a Topological Polar Surface Area (TPSA) close to **100**.
- Maintain a low-to-moderate lipophilicity (LogP ≈ **1**).

Your previous steps:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Previous SMILES to improve:
{previous_smiles}

Double-checker evaluations:
- Step1 Evaluation: {double_checker_feedback['step1']}
- Step2 Evaluation: {double_checker_feedback['step2']}
- Step3 Evaluation: {double_checker_feedback['step3']}

Now, based on your original reasoning and the above feedback, revise your thinking and generate an improved SMILES string that better aligns with your design logic.

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format.
Take a deep breath and think carefully before writing your answer.
```json
{{
  "step1": "Correctly list osimertinib’s key pharmacophores and target TPSA, LogP values.",
  "step2": "Propose chemically sound modifications improving polarity and hydrophobicity balance.",
  "step3": "Describe your final improved structure naturally before SMILES.",
  "SMILES": "Your corrected SMILES string."
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_double_checker_prompt(thinking, improved_smiles):
    return f"""You will be given:
- A user prompt describing the target objective,
- The scientist’s reasoning broken into Step1 through Step3,
- The SMILES string proposed by the scientist.

Evaluate each step **independently**, comparing the described logic to the molecular structure in the SMILES. Provide a reasoning assessment for each step.

If all three steps are logically consistent with the final SMILES, mark "Consistency" as "Consistent".  
If any step is inconsistent, mark "Consistency" as "Inconsistent" and provide specific suggestions for improvement.


--- SCIENTIST'S TASK ---
Design a SMILES string for a molecule that satisfies the following conditions:
- High structural similarity to osimertinib (SMILES: "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc2nccc(n2)c3cn(C)c4ccccc34").
- Achieve a Topological Polar Surface Area (TPSA) close to **100**.
- Maintain a low-to-moderate lipophilicity (LogP ≈ **1**).

IMPORTANT CONSTRAINT:  
YOU MUST NOT generate a molecule identical to osimertinib.

--- SCIENTIST'S THINKING ---
Step1: {thinking['step1']}
Step2: {thinking['step2']}
Step3: {thinking['step3']}

--- SCIENTIST'S SMILES ---
{improved_smiles}

You must return your response in the following json format.
The text inside each key explains what kind of answer is expected — it is a **guideline, not the answer**.

DO NOT repeat the example text or instructions.  
Instead, write your own scientifically reasoned content based on the task.

Use the following format:  
```json
{{
  "step1": "Evaluate whether key osimertinib features were identified and preserved.",
  "step2": "Analyze the chemical soundness of the proposed design modifications.",
  "step3": "Confirm logical consistency between design steps and SMILES.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """


