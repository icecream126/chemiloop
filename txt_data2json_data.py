import pandas as pd
import json
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
import torch
from utils.metrics import * 
# Load CSV file
csv_path = "/home/khm/chemiloop/dataset/250k_rndm_zinc_drugs_clean_3.csv"
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
            albuterol_similarity_score = get_albuterol_similarity_score(smiles)
            isomers_c7h8n2o2_score = get_isomers_c7h8n2o2_score(smiles)
            isomers_c9h10n2o2pf2cl_score = get_isomers_c9h10n2o2pf2cl_score(smiles)
            celecoxib_rediscovery_score = get_celecoxib_rediscovery_score(smiles)
            amlodipine_mpo_score = get_amlodipine_mpo_score(smiles)
            deco_hop_score = get_deco_hop_score(smiles)
            drd2_score = get_drd2_score(smiles)
            fexofenadine_mpo_score = get_fexofenadine_mpo_score(smiles)
            gsk3b_score = get_gsk3b_score(smiles)
            jnk3_score = get_jnk3_score(smiles)
            median1_score = get_median1_score(smiles)
            median2_score = get_median2_score(smiles)
            mestranol_similarity_score = get_mestranol_similarity_score(smiles)
            osimertinib_mpo_score = get_osimertinib_mpo_score(smiles)
            perindopril_score = get_perindopril_mpo_score(smiles)
            qed_score = get_qed_score(smiles)
            ranolazine_mpo_score = get_ranolazine_mpo_score(smiles)
            scaffold_hop_score = get_scaffold_hop_score(smiles)
            sitagliptin_mpo_score = get_sitagliptin_mpo_score(smiles)
            thiothixene_rediscovery_score = get_thiothixene_rediscovery_score(smiles)
            troglitazone_rediscovery_score = get_troglitazone_rediscovery_score(smiles)
            valsartan_smarts_score = get_valsartan_smarts_score(smiles)
            zaleplon_mpo_score = get_zaleplon_mpo_score(smiles)

            # molwt = Descriptors.MolWt(mol)
            # import pdb; pdb.set_trace()
            output_data.append({
                "smiles": smiles,
                "ecfp4": str(torch.tensor(AllChem.GetMorganFingerprintAsBitVect(mol, useChirality=True, radius=2, nBits = 1024)).tolist()),
                # "molecular_weight": round(molwt, 2),
                "logP": logP,
                "SAS": sas if sas is not None else None,
                "albuterol_similarity_score": albuterol_similarity_score,
                "isomer_c7h8n2o2_score": isomers_c7h8n2o2_score,
                "isomer_c9h10n2o2pf2cl_score": isomers_c9h10n2o2pf2cl_score,
                "celecoxib_rediscovery_score": celecoxib_rediscovery_score,
                "amlodipine_mpo_score": amlodipine_mpo_score,
                "deco_hop_score": deco_hop_score,
                "drd2_score": drd2_score,
                "fexofenadine_mpo_score": fexofenadine_mpo_score,
                "gsk3b_score": gsk3b_score,
                "jnk3_score": jnk3_score,
                "median1_score": median1_score,
                "median2_score": median2_score,
                "mestranol_similarity_score": mestranol_similarity_score,
                "osimertinib_mpo_score": osimertinib_mpo_score,
                "perindopril_mpo_score": perindopril_score,
                "qed_score": qed_score,
                "ranolazine_mpo_score": ranolazine_mpo_score,
                "scaffold_hop_score": scaffold_hop_score,
                "sitagliptin_mpo_score": sitagliptin_mpo_score,
                "thiothixene_rediscovery_score": thiothixene_rediscovery_score,
                "troglitazone_rediscovery_score": troglitazone_rediscovery_score,
                "valsartan_smarts_score": valsartan_smarts_score,
                "zaleplon_mpo_score": zaleplon_mpo_score

            })
        except:
            continue

# Save to JSON
json_path = "./dataset/entire_zinc250.json"
with open(json_path, "w") as f:
    json.dump(output_data, f, indent=2)

print(f"Saved {len(output_data)} entries to {json_path}")
