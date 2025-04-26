import re
import json
import pandas as pd

# Simulated contents of your txt file
with open('/home/khm/chemiloop/dataset/rdkit_tool_registry_summary.txt', 'r') as file:
    txt_data = file.read()

# Split blocks by "Function:"
blocks = re.split(r"\bFunction:\s*", txt_data.strip())[1:]

tools = []

for block in blocks:
    lines = block.strip().splitlines()
    name = lines[0].strip()
    desc = ""
    input_props = {}

    for line in lines[1:]:
        line = line.strip()
        if line.startswith("Description:"):
            desc = line.replace("Description:", "").strip()
        elif line.startswith("Input Type"):
            match = re.findall(r"\((\w+)\)(\w+)", line)
            for typ, param in match:
                input_props[param] = {
                    "type": "string",
                    "description": f"{typ} object"
                }

    tool = {
        "type": "function",
        "function": {
            "name": name.lower(),
            "description": desc,
            "parameters": {
                "type": "object",
                "properties": input_props,
                "required": list(input_props.keys())
            }
        }
    }
    tools.append(tool)

tool_path = "./dataset/rdkit_tool_json.txt"
with open(tool_path, "w") as f:
    json.dump(tools, f, indent=2)

# Step 3: Function to load the tool definition for use
def load_tools(path):
    with open(path, "r") as f:
        return json.load(f)

# Load tools for example use
loaded_tools = load_tools(tool_path)
# loaded_tools[:1]  # show a sample for confirmation