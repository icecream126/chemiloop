

zaleplon_mpo_smiles="O=C(C)N(CC)C1=CC=CC(C2=CC=NC3=C(C=NN23)C#N)=C1"

def get_scientist_prompt(SMILES_HISTORY, topk_smiles):
    return f"""Previously generated SMILES. YOU MUST NOT REPEAT ANY OF THEM:
{SMILES_HISTORY}

Your task is to design a SMILES string for a molecule that satisfies the following conditions: 
- Achieve high structural similarity to zaleplon (SMILES: {zaleplon_mpo_smiles}).
- Match the molecular formula **C19H17N3O2** exactly (correct atom counts).

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO ZALEPLON (SMILES: {zaleplon_mpo_smiles}).

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
  "step1": "List zaleplon's key structural and functional features (e.g., pyrazolopyrimidine core, amide side chain, aromatic rings). Mention the target molecular formula C19H17N3O2.",
  "step2": "Propose scaffold modifications or side chain adjustments that maintain similarity and formula. Justify each change chemically.",
  "step3": "Describe the designed molecule naturally, including key cores and substituents, before writing the SMILES.",
  "SMILES": "Your valid SMILES string here"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_scientist_prompt_with_review(scientist_think_dict, reviewer_feedback_dict, previous_smiles, score, functional_groups, SMILES_HISTORY, topk_smiles):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{SMILES_HISTORY}

Task: Refine your molecule based on reviewer's feedback.

Original goal:
- Design a molecule highly similar to zaleplon (SMILES: {zaleplon_mpo_smiles}).
- Match the exact molecular formula C19H17N3O2.

Top-5 Relevant SMILES Examples (SMILES, score) are as below.
You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.:
{topk_smiles}

You will also see:
1. Molecule SMILES to improve
2. Its zaleplon_mpo score
3. Its functional groups

--- MOLECULE SMILES TO IMPROVE ---  
MOLECULE SMILES: {previous_smiles}
- zaleplon_mpo score: {score}
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
  "step1": "Updated list of zaleplon’s key features and formula requirements.",
  "step2": "Refined design strategy to boost similarity and maintain molecular formula C19H17N3O2.",
  "step3": "Natural description of the designed structure.",
  "SMILES": "Your corrected valid SMILES string"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_reviewer_prompt(scientist_think_dict, score, functional_groups):
    return f"""Evaluate the Scientist LLM’s reasoning steps and final SMILES molecule for:
- Validity
- Chemical soundness
- Adherence to the design condition:
Design a SMILES string for a molecule that satisfies the following conditions: 
- Achieve high structural similarity to zaleplon (SMILES: {zaleplon_mpo_smiles}).
- Match the molecular formula **C19H17N3O2** exactly (correct atom counts).

IMPORTANT CONSTRAINT:  
MUST NOT GENERATE A MOLECULE IDENTICAL TO ZALEPLON (SMILES: {zaleplon_mpo_smiles}).

You are provided with:
- Scientist's thinking.
- Scientist-generated SMILES.
- zaleplon_mpo score.
- Detected functional groups.

--- SCIENTIST'S THINKING ---
Step 1: {scientist_think_dict['step1']}
Step 2: {scientist_think_dict['step2']}
Step 3: {scientist_think_dict['step3']}

--- SCIENTIST-GENERATED MOLECULE SMILES ---
MOLECULE SMILES: {scientist_think_dict['smiles']}
- zaleplon_mpo score: {score}
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
  "step1": "Analyze whether key zaleplon features and formula constraint were correctly captured. Point out missing or wrong features.",
  "step2": "Assess whether the design maintains high similarity and matches C19H17N3O2 formula. Suggest improvements if needed.",
  "step3": "Confirm whether the SMILES structure matches the stepwise logic. Highlight inconsistencies."
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """

def get_scientist_prompt_with_double_checker_review(previous_thinking, previous_smiles, double_checker_feedback, SMILES_HISTORY):
    return f"""YOU MUST NOT REPEAT ANY OF THE PREVIOUSLY GENERATED SMILES:
{SMILES_HISTORY}

Improve your previously designed molecule based on double-checker feedback.

Original Task:
- Design a molecule highly similar to zaleplon.
- Match the molecular formula C19H17N3O2.

Your previous reasoning:
- Step1: {previous_thinking['step1']}
- Step2: {previous_thinking['step2']}
- Step3: {previous_thinking['step3']}

Previous SMILES to improve:
{previous_smiles}

Double-checker feedback:
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
  "step1": "List correctly captured zaleplon features and molecular formula constraints.",
  "step2": "Propose chemically rational modifications ensuring high similarity and formula matching.",
  "step3": "Describe the new designed molecule before presenting the SMILES.",
  "SMILES": "Your improved and valid SMILES string"
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
- Achieve high structural similarity to zaleplon (SMILES: {zaleplon_mpo_smiles}).
- Match the molecular formula **C19H17N3O2** exactly (correct atom counts).

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO ZALEPLON (SMILES: {zaleplon_mpo_smiles}).

--- SCIENTIST'S REASONING ---
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
  "step1": "Analyze whether the critical structural features and molecular formula are preserved.",
  "step2": "Evaluate if the proposed modifications are chemically sensible and fulfill the rediscovery task.",
  "step3": "Check if the final SMILES logically follows the thought process.",
  "consistency": "Consistent" or "Inconsistent"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """
