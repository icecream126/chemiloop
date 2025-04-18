def get_scientist_prompt(task):
    
    return f"""You are a skilled chemist.

Task: Design a SMILES string for a molecule that satisfies the condition: "{task}".

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
SMILES: [Your valid SMILES here]  
"""

def get_scientist_prompt_isomers_c7h8no2():
    return f"""You are a skilled chemist.  
Task: Design a improved molecule in SMILES format that satisfies the following condition:

Objective: isomers_c7h8n2o2
Create a valid chemical structure in SMILES format that is an isomer of the molecular formula C7H8N2O2.

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
SMILES: [Your valid SMILES here]"""