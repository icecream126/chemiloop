import pandas as pd
import json

# Load the new CSV file
csv_path = "/home/khm/chemiloop/dataset/emitters.csv"
df = pd.read_csv(csv_path)

# Drop rows with missing values
df_clean = df.dropna(subset=['multi-objective value', 'oscillator strength'])

# Filter out dim molecules
df_filtered = df_clean[df_clean['oscillator strength'] >= 0.05]

# Sort by multi-objective value (higher is better)
top5_emitters = df_filtered.sort_values(by='multi-objective value', ascending=False).head(5)

# Create output dictionary
output = {
    "top5_emitters": top5_emitters.to_dict(orient='records')
}

# Save to JSON file
output_path = "/home/khm/chemiloop/dataset/tartarus_top_5/emitters.json"
with open(output_path, "w") as f:
    json.dump(output, f, indent=4)
