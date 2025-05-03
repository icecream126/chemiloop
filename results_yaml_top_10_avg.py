import yaml
import matplotlib.pyplot as plt
import numpy as np

# Load the YAML file
results_yaml_path = "/home/khm/chemiloop/logs/2025-05-01-03-08-15-usvk9yhc/results.yaml" # scaffold_hop
with open(results_yaml_path, "r") as f:
    buffer = yaml.safe_load(f)

# Sort molecules by oracle call index
ordered_results = sorted(buffer.items(), key=lambda kv: kv[1][1])

# Set parameters
top_n = 10
freq_log = 10
max_oracle_calls = 1000

# Prepare lists for plotting
x, y = [], []

# Compute top-n average score at each oracle step
for idx in range(freq_log, max_oracle_calls + 1, freq_log):
    current_results = [item for item in ordered_results if item[1][1] <= idx]
    if len(current_results) == 0:
        continue
    top_n_scores = [item[1][0] for item in sorted(current_results, key=lambda kv: kv[1][0], reverse=True)[:top_n]]
    top_n_avg = np.mean(top_n_scores)
    x.append(idx)
    y.append(top_n_avg)

# Plot the curve
plt.figure(figsize=(10, 6))
plt.plot(x, y, marker='o', linestyle='-', color='blue')
plt.title(f"Top-{top_n} Average Score by Oracle Call")
plt.xlabel("Oracle Call")
plt.ylabel(f"Top-{top_n} Average Score")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.savefig(f"{results_yaml_path[:-12]}top_10_avg.png")