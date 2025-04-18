import json

def get_scientist_prompt(task, SMILES_HISTORY):
  return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: {task}.

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO ALBUTEROL, defined as:  
- SMILES: CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O  
- canonical SMILES: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1

DO NOT REPRODUCE ANY OF THE PREVIOUSLY GENREATED SMILES LISTED BELOW:
{SMILES_HISTORY}

YOU MUST RETURN YOUR RESPONSE STRICTLY IN THE FOLLOWING JSON FORMAT AND NOTHING ELSE:  
```json
{{
  "Step1": "List of the target’s critical structural/property features (e.g., 'Albuterol: phenyl ring, β-hydroxyamine, catechol-like substitution')\nIf property-based, specify requirements (e.g., "logP > 3: add hydrophobic groups").",
  "Step2": "Propose modifications or scaffolds to meet the condition (e.g., 'Replace catechol with 3-hydroxy-4-pyridone').\n Justify each change chemically (e.g., "Maintains H-bonding but improves metabolic stability").",
  "Step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., "A tert-butyl group attached to the amine (–NH–C(CH₃)₃) to mimic albuterol’s bulky substituent.")",
  "SMILES": "Your valid SMILES string here"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """



def get_scientist_prompt_isomers_c7h8no2(SMILES_HISTORY):
  return f"""Task: Design a molecule in SMILES format that satisfies the following condition:

Objective: isomers_c7h8n2o2  
Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.  
Generate the SMILES including explicit hydrogen atoms.

DO NOT REPRODUCE ANY OF THE PREVIOUSLY GENREATED SMILES LISTED BELOW:
{SMILES_HISTORY}

HARD CONSTRAINT (MUST follow exactly):  
The molecule must match this exact molecular formula:
- 7 Carbon atoms  
- 8 Hydrogen atoms  
- 2 Nitrogen atoms  
- 2 Oxygen atoms  
Any molecule not matching this formula is INVALID and will be discarded.

YOU MUST RETURN YOUR RESPONSE STRICTLY IN THE FOLLOWING JSON FORMAT AND NOTHING ELSE:  
```json
{{
  "Step1": "List possible structural motifs or fragments consistent with the formula C7H8N2O2.\n(e.g., "Common groups for C7H8N2O2: aromatic rings, nitro groups, amines, amides, phenols")",
  "Step2": "Propose a valid isomer design strategy to maximize desired drug-like properties (e.g., "Maximize QED: incorporate a para-substituted aniline with hydrophilic groups").\nJustify each change chemically (e.g., "Adding a hydroxyl group improves hydrogen bonding, enhancing solubility and QED")",
  "Step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES.\n(e.g., "A para-substituted phenyl ring bearing a hydroxyl and acetamide group to balance lipophilicity and polarity")",
  "SMILES": "Your valid SMILES here"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """
