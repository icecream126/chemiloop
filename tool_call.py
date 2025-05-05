from openai import OpenAI
import json

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
from guacamol.utils.chemistry import canonicalize
import utils.utils
import utils.tools as tools
import json

albuterol_smiles = 'CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O'
albuterol_canonical_smiles = canonicalize(albuterol_smiles)
albuterol_mol = Chem.MolFromSmiles(albuterol_smiles)
albuterol_functional_group = utils.utils.describe_albuterol_features(albuterol_mol)
tool_path = "/home/khm/chemiloop/dataset/filtered_rdkit_tool.json"

with open(tool_path, "r") as tool_json:
    tool_specs = json.load(tool_json)



system_prompt = """You are a professional AI chemistry assistant.

You are given a molecule design condition and a set of available chemical tools (functions). Your goal is to:

1. Analyze the condition and identify key molecular features (e.g., functional groups, properties).
2. Choose **as many tools as necessary** from the toolset that are relevant to solving the task.
   - The number of selected tools is **not limited**.
3. Explain why each tool is useful for this task."""

user_prompt = """Condition for molecule design:
Design a drug-like molecule structurally similar to albuterol (SMILES: {albuterol_smiles}, canonical: {albuterol_canonical_smiles}). 
Preserve the core scaffold and key functional groups. Albuterol contains: {albuterol_functional_group}.
  
IMPORTANT CONSTRAINT:  
YOU MUST NOT GENERATE A MOLECULE IDENTICAL TO ALBUTEROL, defined as:  
- SMILES: CC(C)(C)NCC(C1=CC(=C(C=C1)O)CO)O  
- canonical SMILES: CC(C)(C)NCC(O)c1ccc(O)c(CO)c1
             
Now output the tools to use by using the following JSON format.
Take a deep breath and think carefully before writing your answer.
```json
{
  "tools_to_use": [
    {"name": "function_name_1", "reason": "Why this function is useful."},
    {"name": "function_name_2", "reason": "Why this function is useful."},
  ]
}
```
IF YOU DO NOT FOLLOW THIS FORMAT, INNOCENT PEOPLE WILL DIE."""


client = OpenAI(api_key="sk-577e021a148d45e4b5e842fc43a6a07c", base_url="https://api.deepseek.com")




messages = [{"role": "system", "content": system_prompt}]
messages = [{"role": "user", "content": user_prompt}]


response = client.chat.completions.create(
    model="deepseek-chat",
    messages=messages,
    tools=tool_specs,
    response_format={'type': 'json_object'},
    temperature=1.5,
)
    
# Parse LLM output
llm_reply = response.choices[0].message.content
print("\n🔎 LLM Reasoning and Tool Selection:\n", llm_reply)

# Try to extract tool names from the JSON string
try:
    tool_json = json.loads(llm_reply)
    tools_to_use = tool_json.get("tools_to_use", [])
except Exception as e:
    print("❌ Failed to parse JSON:", e)
    tools_to_use = []

# Execute each selected function if available
print("\n🚀 Executing selected tools on albuterol...\n")

report_lines = []

for tool in tools_to_use:
    func_name = tool["name"].lower()
    reason = tool["reason"]
    
    try:
        func = getattr(tools, func_name)
        result = func(albuterol_mol)

        report = f"""🔧 Tool: `{func_name}`
📌 Reason: {reason}
📊 Output: `{result}`

"""
        report_lines.append(report)

    except Exception as e:
        report = f"""🔧 Tool: `{func_name}`
📌 Reason: {reason}
❌ Error: Could not execute `{func_name}` — {str(e)}

"""
        report_lines.append(report)

# Print full report
final_report = "\n".join(report_lines)
print("📝 Tool Execution Report:\n")
print(final_report)
