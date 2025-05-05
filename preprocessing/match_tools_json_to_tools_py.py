import ast
import re
import json

function_definition_path = "/home/khm/chemiloop/utils/tools.py"
# Load raw Python source file that contains the function definitions
with open(function_definition_path, "r") as f:
    source_code = f.read()

# Use AST to extract all function names
parsed_ast = ast.parse(source_code)
defined_function_names = {node.name.replace("_","") for node in parsed_ast.body if isinstance(node, ast.FunctionDef)}

# Load the rdkit tool JSON
with open("/home/khm/chemiloop/dataset/rdkit_tool.json", "r") as f:
    tool_data = json.load(f)


filtered_tool_data = []
for entry in tool_data:
    if entry.get("type") == "function":
        temp_entry_name = entry["function"]["name"]
        temp_entry_name = temp_entry_name.lower().replace("_", "")
        function_name = temp_entry_name
        if function_name in defined_function_names:
            filtered_tool_data.append(entry)

# Save the filtered JSON
final_path = "./filtered_function_names.json"
with open(final_path, "w") as f:
    json.dump(filtered_tool_data, f, indent=2)

