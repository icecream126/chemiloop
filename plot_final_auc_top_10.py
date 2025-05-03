import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths to the CSV files
csv_paths = [
    "./csv/fexofenadine_mpo.csv",
    "./csv/median2.csv",
    "./csv/mestranol_similarity.csv",
    "./csv/valsatran_smarts.csv"
]

# Generate plots
for path in csv_paths:
    df = pd.read_csv(path)
    
    # Identify relevant columns
    import pdb; pdb.set_trace()
    run_set_cols = [col for col in df.columns if "(Run set) - auc_top10_all" in col]
    run_set2_cols = [col for col in df.columns if "(Run set2) - auc_top10_all" in col]

    if not run_set_cols or not run_set2_cols:
        continue  # Skip if required columns are missing

    run_set_col = run_set_cols[0]
    run_set2_col = run_set2_cols[0]

    # Starting from the 10th step
    x_values = list(range(10, len(df)))
    y_run_set = df[run_set_col].iloc[10:]
    y_run_set2 = df[run_set2_col].iloc[10:]

    # Plot
    plt.figure()
    plt.plot(x_values, y_run_set, label=run_set_col, color='blue')
    plt.plot(x_values, y_run_set2, label=run_set2_col, color='orange')
    plt.xlabel("Step")
    plt.ylabel("AUC Top 10 All")
    plt.title(f"AUC Comparison - {os.path.basename(path)}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig('./plots/' + os.path.basename(path) + '.png')
    print("Saved at ./plots/" + os.path.basename(path) + ".png")
