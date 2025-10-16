import os
import json
import time
import openai
import argparse
from tqdm import tqdm
from utils_llm import factchecker
from logger_config import logger
openai.api_key = os.environ["OPENAI_API_KEY"]

def parse_fill(reply):
    if reply.count("<<FILL>>") >= 2:
        return reply.split("<<FILL>>")[1].strip()
    return reply.strip()

with open("./prompt/fact_checking.txt", "r", encoding="utf-8") as f:
    fact_check_prompt = f.read()

parser = argparse.ArgumentParser(description="Inference with GPT")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., freshllm, musique, 2wikimultihop)")
args = parser.parse_args()
dataset = args.dataset

in_path = f'./results/result_{dataset}.json'
out_path = f"./results/factchecking_{dataset}.json"
with open(in_path, "r", encoding="utf-8") as f:
    data = json.load(f)
    logger.info("Data Loading for Fact Checking Done!!")

correct_cnt = 0
incorrect_cnt = 0
total = len(data)

checking_result = []
for item in tqdm(data, desc="Fact checking"):
    question = item["Question"]
    generated_answer = item["Final Answer"]
    ground_truth = item["Ground_Truth"]
    prompt = f"""{fact_check_prompt}

---
Example4
QUESTION:
{question}

GROUND TRUTH:
{ground_truth}

GENERATED ANSWER:
{generated_answer}

OUTPUT:
"""
    reply = factchecker(prompt)
    result = parse_fill(reply)
    print(result)
    if result.lower() == "correct":
        correct_cnt += 1
    elif result.lower() == "incorrect":
        incorrect_cnt += 1
    else:
        print(f"Unrecognized response: {reply}")
    item['factchecking'] = result
    checking_result.append(item)

print("="*50)
print(f"Total samples: {total}")
print(f"Correct: {correct_cnt}")
print(f"Incorrect: {incorrect_cnt}")
print(f"Accuracy: {correct_cnt/total:.4f}")
print("="*50)

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(checking_result, f, indent=4, ensure_ascii=False)
    logger.info(f"Fact Checking for {dataset} Done!!")
