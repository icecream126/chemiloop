import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from guacamol.utils.chemistry import canonicalize
import utils.utils


celecoxib_smiles = 'CC1=CC=C(C=C1)C2=CC(=NN2C3=CC=C(C=C3)S(=O)(=O)N)C(F)(F)F'
celecoxib_canonical_smiles = canonicalize(celecoxib_smiles)
celecoxib_mol = Chem.MolFromSmiles(celecoxib_smiles)
celecoxib_functional_gorups = utils.utils.describe_celecoxib_features(celecoxib_mol)


def get_scientist_prompt(SMILES_HISTORY, topk_smiles):
  
  return f"""Your task is to design a SMILES string for a molecule that satisfies the following condition: 
Design a drug-like molecule structurally similar to celecoxib (SMILES: {celecoxib_smiles}, canonical: {celecoxib_canonical_smiles}). 
Preserve the core scaffold and important pharmacophores (e.g., sulfonamide, pyrazole ring).

IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO CELECOXIB, defined as:  
- SMILES: {celecoxib_smiles}
- canonical SMILES: {celecoxib_canonical_smiles}

You are provided with:
- Top-5 example molecules with high relevance to the task, listed below. You may use these as inspiration, but YOU MUST NOT COPY THEM EXACTLY.
- A list of previously generated SMILES, which YOU MUST NOT REPEAT.

Top-5 Relevant SMILES Examples (SMILES, score):
{topk_smiles}

Previously generated SMILES:
{SMILES_HISTORY}

You must return your response in the following json format.

```json
{{
  "step1": "List of the target’s critical structural/property features (e.g., 'Celecoxib: pyrazole core, sulfonamide group, phenyl rings for hydrophobicity.')",
  "step2": "Propose modifications or scaffolds to meet the condition (e.g., 'Replace one phenyl ring with a thiophene to modulate hydrophobicity.')",
  "step3": "Describe the full structure of your designed molecule in natural language before writing the SMILES. (e.g., 'A sulfonamide-substituted pyrazole ring connected to a thiophene and phenyl group.')",
  "SMILES": "Your valid SMILES string here"
}}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE. """