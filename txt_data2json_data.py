import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
import torch
from utils.metrics import get_albuterol_similarity_score, get_isomer_c7h8n2o2_score, get_isomer_c9h10n2o2pf2cl_score
# Load CSV file
csv_path = "/home/khm/chemiloop/dataset/subset_250k_rndm_zinc_drugs_clean_3.csv"
df = pd.read_csv(csv_path)
train_len = int(len(df) * 0.9)
df = df[:train_len]

output_data = []

for _, row in df.iterrows():
    smiles = row['smiles']
    mol = Chem.MolFromSmiles(smiles)
    # import pdb; pdb.set_trace()
    if mol:
        try:
            # ecfp4 = str(torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits = 1024)).tolist()),
            # ecfp4_list = list(ecfp4)
            logP = row['logP'] if 'logP' in row else Descriptors.MolLogP(mol)
            qed = row['qed'] if 'qed' in row else QED.qed(mol)
            sas = row['SAS'] if 'SAS' in row else None
            albuterol_score = get_albuterol_similarity_score(smiles)
            isomer_c7h8n2o2_score = get_isomer_c7h8n2o2_score(smiles)
            isomer_c9h10n2o2pf2cl_score = get_isomer_c9h10n2o2pf2cl_score(smiles)
            # molwt = Descriptors.MolWt(mol)
            # import pdb; pdb.set_trace()
            output_data.append({
                "smiles": smiles,
                "ecfp4": str(torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits = 1024)).tolist()),
                # "molecular_weight": round(molwt, 2),
                "logP": logP,
                "QED": qed,
                "SAS": sas if sas is not None else None,
                "albuterol_similarity_score": albuterol_score,
                "isomer_c7h8n2o2_score": isomer_c7h8n2o2_score,
                "isomer_c9h10n2o2pf2cl_score": isomer_c9h10n2o2pf2cl_score
            })
        except:
            continue

# Save to JSON
json_path = "./dataset/subset_zinc250.json"
with open(json_path, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"Saved {len(output_data)} entries to {json_path}")
