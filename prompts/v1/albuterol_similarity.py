
import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from guacamol.utils.chemistry import canonicalize
import utils.utils

albuterol_smiles = 'CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O'
albuterol_canonical_smiles = canonicalize(albuterol_smiles)
albuterol_mol = Chem.MolFromSmiles(albuterol_smiles)
albuterol_functional_group = utils.utils.describe_albuterol_features(albuterol_mol)


def get_scientist_prompt(SMILES_HISTORY, topk_smiles):
  
  return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: 
Design a drug-like molecule structurally similar to albuterol (SMILES: {albuterol_smiles}, canonical: {albuterol_canonical_smiles}). 
Preserve the core scaffold and key functional groups. Albuterol contains: {albuterol_functional_group}.
  
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