import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem

# ---------------------------
# AUC calculation function (same as your framework)
# ---------------------------
def top_auc(buffer, top_n, finish, freq_log, max_oracle_calls):
    sum_auc = 0
    prev = 0
    called = 0

    # Sort by generation order (value[1] = call index)
    ordered_results = list(sorted(buffer.items(), key=lambda kv: kv[1][1], reverse=False))

    for idx in range(freq_log, min(len(buffer), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        # Sort by score, take top_n
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_n]
        top_n_now = np.mean([item[1][0] for item in temp_result])
        sum_auc += freq_log * (top_n_now + prev) / 2
        prev = top_n_now
        called = idx

    # Handle the last segment
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_n]
    top_n_now = np.mean([item[1][0] for item in temp_result])
    sum_auc += (len(buffer) - called) * (top_n_now + prev) / 2

    if finish and len(buffer) < max_oracle_calls:
        sum_auc += (max_oracle_calls - len(buffer)) * top_n_now

    return sum_auc / max_oracle_calls

# ---------------------------
# Function to process the text file
# ---------------------------
def compute_topk_auc(smiles_scores, top_k=5, max_oracle_calls=1000, freq_log=1):
    # smiles_scores = []

    print("number of generated smiles", len(smiles_scores))
    # Step 2: Remove invalid or duplicate SMILES
    seen = set()
    mol_buffer = {}
    for i, (smi, score) in enumerate(smiles_scores):
        mol = Chem.MolFromSmiles(smi)
        if mol:  # valid SMILES
            canonical_smi = Chem.MolToSmiles(mol)  # Canonicalize
            if canonical_smi not in seen:
                # Store as {canonical_smi: [score, generation index]}
                mol_buffer[canonical_smi] = [score, i + 1]
                seen.add(canonical_smi)
            else:
                continue  # duplicated canonical SMILES
        else:
            continue  # invalid SMILES

    # Step 3: Compute AUC
    auc = top_auc(mol_buffer, top_k, finish=False, freq_log=freq_log, max_oracle_calls=max_oracle_calls)
    print(f"Top-{top_k} AUC Score: {auc:.4f}")
    return auc, mol_buffer