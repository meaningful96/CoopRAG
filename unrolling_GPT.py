import os
import json
import re
import time
import openai
from openai.error import Timeout, RateLimitError, APIConnectionError, APIError

openai.api_key = os.getenv("OPENAI_API_KEY")
with open('./prompt/unrolling_subq.txt', 'r', encoding='utf-8') as f:
    system_prompt1 = f.read()

with open('./prompt/unrolling_rc.txt', 'r', encoding='utf-8') as f:
    system_prompt2 = f.read()

def ask_gpt4(prompt):
    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1024,
                top_p=0.9,
                request_timeout=1200,
                timeout=1200
            )
            return response.choices[0].message["content"]
        except Timeout as e:
            print(f"[OpenAI Timeout] Retry {attempt+1}/3: {e}")
            if attempt == 2:
                print("OpenAI request timed out 3 times. Returning empty string.")
                return ""
            else:
                time.sleep(10)
        except RateLimitError as e:
            print(f"[OpenAI RateLimitError] Retry {attempt+1}/3: {e}")
            if attempt == 2:
                print("Rate limit exceeded 3 times. Returning empty string.")
                return ""
            else:
                time.sleep(10)
        except APIConnectionError as e:
            print(f"[OpenAI APIConnectionError] Retry {attempt+1}/3: {e}")
            if attempt == 2:
                print("API connection failed 3 times. Returning empty string.")
                return ""
            else:
                time.sleep(10)
        except APIError as e:
            print(f"[OpenAI APIError] Retry {attempt+1}/3: {e}")
            if attempt == 2:
                print("API error occurred 3 times. Returning empty string.")
                return ""
            else:
                time.sleep(10)
        except Exception as e:
            print(f"[Other Error] Retry {attempt+1}/3: {e}")
            if attempt == 2:
                print("Unknown error occurred 3 times. Returning empty string.")
                return ""
            else:
                time.sleep(10)

def clean_para(text):
    # remove special characters
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
    # parse list of sub-questions from output
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
    output1 = ask_gpt4(prompt1)
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
    output2 = ask_gpt4(prompt2)
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
    cnt+=1

with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(preprocessed, f, ensure_ascii=False, indent=4)

print(f"Processing completed. Output saved to: {out_path}")

