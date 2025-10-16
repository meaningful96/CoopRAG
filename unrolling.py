import os
import re
import json
import copy
import time
import openai
import argparse
from logger_config import logger
from utils_llm import ask_gpt4

openai.api_key = os.environ["OPENAI_API_KEY"]
def parse_subq_list(output_str):
    # Parse list of sub-questions from output
    pattern = r'\[\s*(.*?)\s*\]'
    m = re.search(pattern, output_str, flags=re.DOTALL)
    if m:
        inner = m.group(1)
        items = re.findall(r'"(.*?)"', inner)
        return items
    return []

def parse_triple_list(output_str):
    pattern = r"Triple Reasoning Chain:\s*(\[[\s\S]+\])"
    m = re.search(pattern, output_str)
    if m:
        json_str = m.group(1)
        try:
            triple_list = json.loads(json_str)
            return triple_list
        except Exception as e:
            triple_pattern = r'\[\s*"(.*?)"\s*,\s*"(.*?)"\s*,\s*"(.*?)"\s*\]'
            triples = re.findall(triple_pattern, json_str)
            return [list(triple) for triple in triples]
    return []

def process_example(example):
    question = example['question']
    prompt1 = f"""
{system_prompt1}

---

Example 6
Original Question: {question}

Outputs:
    """
    attempt = 1
    subq_list = []
    triple_list = []
    while attempt <= 5:
        print(f"Attempt {attempt}...")
        output1 = ask_gpt4(prompt1)
        output1 = re.sub(r'\\', '', output1).strip()
        subq_list = parse_subq_list(output1)
        triple_list = parse_triple_list(output1)
        if subq_list and triple_list:
            break
        attempt += 1

    example['sub_questions'] = subq_list
    example['sub_triples'] = triple_list
    print(f"Question: {example['question']}")
    print(f"Sub Questions: {example['sub_questions']}")
    print(f"Sub Triples: {example['sub_triples']}")
    print("=" * 100)
    return example

# Reading Prompt
with open('./prompt/unrolling.txt', 'r', encoding='utf-8') as f:
    system_prompt1 = f.read()

parser = argparse.ArgumentParser(description="Inference with GPT")
parser.add_argument("--dataset", type=str, default='hotpotqa')
args = parser.parse_args()

# Load Dataset
dataset = args.dataset
base_dir = f'./Datasets/{dataset}'
file_path = f"{base_dir}/test.json"
out_path = f"{base_dir}/final_test.json"
logger.info(f"Data Path: {file_path}")
logger.info(f"Save Path: {out_path}")

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)

preprocessed = []
cnt = 0
for ex in data:
    print("="*100)
    print(f"Sample {cnt}")
    tmp = copy.deepcopy(ex)
    processed = process_example(tmp)
    preprocessed.append(processed)
    cnt += 1

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(preprocessed, f, ensure_ascii=False, indent=4)

print(f"Processing completed. Output saved to: {out_path}")

