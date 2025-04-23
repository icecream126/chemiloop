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
def compute_topk_auc_from_file(file_path, top_k=5, max_oracle_calls=1000, freq_log=1, run_name=''):
    smiles_scores = []

    # Step 1: Read SMILES and score from the file
    with open(file_path, 'r') as f:
        for line in f:
            if ',' in line:
                smi, score = line.strip().split(',')
                smiles_scores.append((smi.strip(), float(score.strip())))
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
    auc = top_auc(mol_buffer, top_k, finish=True, freq_log=freq_log, max_oracle_calls=max_oracle_calls)
    print(f"Top-{top_k} AUC Score: {auc:.4f}")

    return auc, mol_buffer

# ---------------------------
# Visualization function
# ---------------------------
def plot_auc_progress(mol_buffer, top_k=5, freq_log=1, max_oracle_calls=1000, run_name=''):
    ordered_results = list(sorted(mol_buffer.items(), key=lambda kv: kv[1][1], reverse=False))
    x_vals = []
    y_vals = []
    prev = 0

    for idx in range(freq_log, min(len(ordered_results), max_oracle_calls), freq_log):
        temp_result = ordered_results[:idx]
        temp_result = list(sorted(temp_result, key=lambda kv: kv[1][0], reverse=True))[:top_k]
        top_k_avg = np.mean([item[1][0] for item in temp_result])
        import pdb; pdb.set_trace()
        x_vals.append(idx)
        y_vals.append(top_k_avg)

    # plot final segment
    temp_result = list(sorted(ordered_results, key=lambda kv: kv[1][0], reverse=True))[:top_k]
    final_top_k = np.mean([item[1][0] for item in temp_result])
    x_vals.append(len(ordered_results))
    y_vals.append(final_top_k)

    x_vals = x_vals[9:]
    y_vals = y_vals[9:]

    plt.plot(x_vals, y_vals, label=f"sci_albutero")
    plt.xlabel("Oracle Calls")
    plt.ylabel(f"Top-{top_k} Average Score")
    plt.title("Top-10 Avg score vs Oracle call")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f'./top_10_auc_{run_name}.png')

def plot_auc_progress_2(mol_buffer, top_k=5, freq_log=1, max_oracle_calls=1000, run_name=''):
    # Sort by generation step
    ordered_results = list(sorted(mol_buffer.items(), key=lambda kv: kv[1][1]))

    # Create a dict of {step: score} to track when each molecule was added
    step_to_topk_avg = {}

    # Iterate over every valid generation step
    for i in range(len(ordered_results)):
        up_to_step = ordered_results[:i+1]

        # Sort by score and pick top_k
        top_k_scores = sorted([item[1][0] for item in up_to_step], reverse=True)[:top_k]
        avg_top_k = np.mean(top_k_scores)

        gen_step = ordered_results[i][1][1]
        step_to_topk_avg[gen_step] = avg_top_k

    # Fill missing steps by holding previous value
    min_step = min(step_to_topk_avg.keys())
    max_step = max(step_to_topk_avg.keys())
    x_vals = list(range(min_step, max_step + 1))
    y_vals = []

    last_val = None
    for step in x_vals:
        if step in step_to_topk_avg:
            last_val = step_to_topk_avg[step]
        y_vals.append(last_val)

    # Trim x/y based on freq_log (if needed)
    x_vals = x_vals[::freq_log]
    y_vals = y_vals[::freq_log]

    # Plot
    plt.plot(x_vals, y_vals, label=f"{run_name}")
    plt.xlabel("Oracle Call (Generation Step)")
    plt.ylabel(f"Top-{top_k} Average Score")
    plt.title(f"Top-{top_k} Avg vs Oracle Call ({run_name})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'./top_10_auc_{run_name}_2.png')
    plt.show()


# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    file_path = "/home/khm/chemiloop/logs/2025-04-17-16-16-29-0rs86qzq/smiles.txt" # v2_isomer : 0.9960 (mol buffer length 78) (258 calls)
    # file_path = "/home/khm/chemiloop/logs/2025-04-17-16-16-22-oa8pfog1/smiles.txt" # v2_albutero : 0.9993 (mol buffer length 39) (268 calls)
    # file_path = "/home/khm/chemiloop/logs/2025-04-17-16-16-35-dqe56h9n/smiles.txt" # v1_isomer : 0.9968 (mol buffer length 23) (534 calls)
    # file_path = "/home/khm/chemiloop/logs/2025-04-17-16-16-40-1zh2um3d/smiles.txt" # v1_albutero : 0.9513 (mol buffer length 7) (534 calls)
    top_k = 10
    max_oracle_calls = 1000
    freq_log = 1

    run_name = file_path.split('/')[-2].split('-')[-1]
    auc_score, mol_buffer = compute_topk_auc_from_file(file_path, top_k, max_oracle_calls, freq_log)
    # print(mol_buffer)
    # print("mol buffer length: ", len(mol_buffer))
    # plot_auc_progress(mol_buffer, top_k, freq_log, max_oracle_calls, run_name)
    plot_auc_progress_2(mol_buffer, top_k, freq_log, max_oracle_calls, run_name)