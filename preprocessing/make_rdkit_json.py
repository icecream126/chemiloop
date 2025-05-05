import json
import re
from collections import defaultdict

# Load the raw text
tool_txt_path = "/home/khm/chemiloop/dataset/rdkit_fragments.txt"
with open(tool_txt_path, "r") as f:
    raw_text = f.read()

# Split the text by double newlines (each block contains one function + description)
blocks = raw_text.strip().split("\n\n")

# Prepare output dictionary
tool_dict = {}
formatted_functions = []
for block in blocks:
    lines = block.strip().split("\n")
    if len(lines) < 2:
        continue
    func_line = lines[0].strip()
    desc_line = lines[1].strip()

    # Extract function name from the full function line
    match = re.search(r'rdkit\.Chem\.Fragments\.(\w+)\(', func_line)
    if not match:
        continue
    func_name = match.group(1)
    func_obj = {
        "type": "function",
        "function": {
            "name": func_name,
            "description": desc_line,
            "parameters": {
                "type": "object",
                "properties": {
                    "mol": {
                        "type": "string",
                        "description": "Mol object"
                    }
                },
                "required": ["mol"]
            }
        }
    }
    formatted_functions.append(func_obj)
import pdb; pdb.set_trace()

# Save to JSON
output_path = "./rdkit_fragments.json"
with open(output_path, "w") as f:
    json.dump(formatted_functions, f, indent=2)
print(f"JSON file saved to {output_path}")

output_path
