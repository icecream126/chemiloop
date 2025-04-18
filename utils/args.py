from dotenv import load_dotenv
import os
import argparse

# load .env file info
load_dotenv()

# API Keys
def return_API_keys():
    return {
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY", "your_default_api_key_here"),
        "SERPER_API_KEY": os.getenv("SERPER_API_KEY", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY_YUNHUI": os.getenv("OPENAI_API_KEY_YUNHUI", "your_default_api_key_here"),
        "DEEPSEEK_API_KEY": os.getenv("DEEPSEEK_API_KEY", "your_default_api_key_here")
    }

# argparse
# https://stackoverflow.com/questions/46719811/best-practices-for-writing-argparse-parsers
def parse_args(*args):
    parser = argparse.ArgumentParser()
    parser.add_argument('--molecule_type', type=str, default='chromophore', help='Type of molecule to generate')
    parser.add_argument('--project-name', type=str, default='guacamol_logP', help='wandb project name')
    parser.add_argument('--topk', type=int, default=3, help='TOP k molecules')
    parser.add_argument('--property-name', type=str, default='logP', help='Conditional property')
    parser.add_argument('--property-value', type=float, default=2.0, help='Conditional property value')
    parser.add_argument('--property-unit', type=str, default='', help='Unit of conditional property')
    parser.add_argument('--property-threshold', type=int, default=0, help='Molweight threshold parameter')
    parser.add_argument('--max-iter', type=int, default=30, help='Maximum number of iterations')
    
    parser.add_argument('--scientist-temperature', type=float, default=1.0, help='Temperature setting for scientist LLM')
    parser.add_argument('--reviewer-temperature', type=float, default=1.0, help='Temperature setting for reviewer LLM')
    parser.add_argument('--scientist-model-name', type=str, default="gpt-4o", help='Scientist LLM model')
    parser.add_argument('--reviewer-model-name', type=str, default="gpt-4o", help='Reviewer LLM model')
    parser.add_argument('--doc-batch-size', type=int, default=50, help='Batch size for document processing')
    
    return parser.parse_args(*args)