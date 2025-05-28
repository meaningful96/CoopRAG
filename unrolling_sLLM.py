import os
import json
import re
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_name = "google/gemma-2-9b-it"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype="bfloat16",
)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open('./prompt/unrolling_subq.txt', 'r', encoding='utf-8') as f:
    system_prompt1 = f.read()

with open('./prompt/unrolling_rc.txt', 'r', encoding='utf-8') as f:
    system_prompt2 = f.read()

def ask_gemma(prompt):
    inputs = tokenizer.encode(prompt, return_tensors='pt', max_length=4096, truncation=True).to(device)
    input_length = inputs.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=1024,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
    generated_tokens = outputs[0][input_length:]
    output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return output_text

def clean_para(text):
    pattern = r"[^A-Za-z0-9\s\",.\-:;~]"
    cleaned = re.sub(pattern, "", text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

def cleaning_contexts(docs):
    strings = []
    for i, doc in enumerate(docs):
        title = doc['title']
        text = doc['paragraph_text']
        text = clean_para(text)
        tmp = f"Document {i}: (Title: {title}). {text}\n"
        strings.append(tmp)
    return ''.join(strings)

def parse_subq_list(output_str):
    pattern = r'\[\s*(.*?)\s*\]'
    m = re.search(pattern, output_str, flags=re.DOTALL)
    if m:
        inner = m.group(1)
        items = re.findall(r'"(.*?)"', inner)
        return items
    return []

def parse_triple_list(output_str):
    try:
        keyword = "Triple Reasoning Chain:"
        idx = output_str.find(keyword)
        if idx != -1:
            json_part = output_str[idx+len(keyword):].strip()
        else:
            json_part = output_str.strip()
        triple_list = json.loads(json_part)
        return triple_list
    except Exception as e:
        pattern = r'\[\s*(?P<triples>.+?)\s*\]'
        m = re.search(pattern, output_str, flags=re.DOTALL)
        if m:
            triples_str = m.group("triples")
            triple_pattern = r'\[\s*"(.*?)"\s*,\s*"(.*?)"\s*,\s*"(.*?)"\s*\]'
            triples = re.findall(triple_pattern, triples_str)
            return [list(triple) for triple in triples]
        return []

def process_example(example):
    question = example['question']
    pos = [d for d in example['paragraphs'] if d['is_supporting'] == True]
    docs_string = cleaning_contexts(pos)
    prompt1 = f"""
{system_prompt1}

Example 6

Input:
Original Question: {question}

Outputs:
    """
    output1 = ask_gemma(prompt1)
    output1 = re.sub(r'\\', '', output1).strip()
    subq_list = parse_subq_list(output1)
    example['sub_questions'] = subq_list

    prompt2 = f"""
{system_prompt2}

Example 6
Original Question: {question}
Sub-questions: {subq_list}

Outputs:
    """
    output2 = ask_gemma(prompt2)
    output2 = re.sub(r'\\', '', output2).strip()
    triple_list = parse_triple_list(output2)
    example['sub_triples'] = triple_list

    print(f"Question: {example['question']}")
    print(f"Sub Questions: {example['sub_questions']}")
    print(f"Sub Triples: {example['sub_triples']}")
    print("=" * 100)
    return example

base_dir = './Datasets/Test'
if not os.path.exists(f"{base_dir}/Departed"):
    os.makedirs(f"{base_dir}/Departed")
    print(f"Build a directory: {base_dir}/Departed")
file_path = f"{base_dir}/test_2wikimultihop.json"
out_path = f"{base_dir}/Departed/test_2wikimultihop.json"
print(f"Data Path: {file_path}")
print(f"Save Path: {out_path}")

with open(file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)
if os.path.exists(out_path):
    with open(out_path, 'r', encoding='utf-8') as f:
        preprocessed = json.load(f)
    data = data[len(preprocessed):]
else:
    preprocessed = []

import copy
cnt = 1
cnt += len(preprocessed)
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

