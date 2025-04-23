
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem

def compute_topk_auc_from_file(file_path, top_k=10):
    # Step 1: Read SMILES and scores
    smiles_scores = []
    with open(file_path, 'r') as f:
        for line in f:
            if ',' in line:
                smi, score = line.strip().split(',')
                smiles_scores.append((smi.strip(), float(score.strip())))

    # Step 2: Remove duplicates while keeping first appearance
    seen = set()
    unique_smiles_scores = []
    for smi, score in smiles_scores:
        mol = Chem.MolFromSmiles(smi)
        # Check if the SMILES is valid
        if mol:
            smi = Chem.MolToSmiles(mol) # Canonicalize the SMILES
            if smi not in seen:
                unique_smiles_scores.append((smi, score))
                seen.add(smi)
        else:
            print(f"Invalid SMILES: {smi}")

    print('unique smiles score: ', unique_smiles_scores)
    # Step 3: Compute top-k AUC over steps
    topk_avgs = []
    for i in range(1, len(unique_smiles_scores) + 1):
        topk = sorted([s for _, s in unique_smiles_scores[:i]], reverse=True)[:top_k]
        topk_avg = sum(topk) / top_k
        topk_avgs.append(topk_avg)

    # Step 4: Normalize AUC
    auc = np.trapz(topk_avgs, dx=1) / len(topk_avgs)

    # Step 5: Plot
    plt.figure(figsize=(8, 5))
    plt.plot(topk_avgs, label=f'Top-{top_k} AUC = {auc:.4f}')
    plt.xlabel('Step')
    plt.ylabel(f'Average Top-{top_k} Score')
    plt.title(f'Top-{top_k} AUC over Time')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('./topk_auc_plot.png')

# Example usage:
text_file_path = "/home/khm/chemiloop/logs/2025-04-17-16-16-29-0rs86qzq/smiles.txt"
text_file_path2 = "/home/khm/chemiloop/logs/2025-04-17-16-16-35-dqe56h9n/smiles.txt"
compute_topk_auc_from_file(text_file_path, top_k=10)
