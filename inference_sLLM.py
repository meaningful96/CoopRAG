import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re
from sentence_transformers import SentenceTransformer, util
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import os
import numpy as np
import random
import string
from collections import Counter
from prompt_inputs import initial_input
import pdb
import time
import torch._dynamo
import ast

torch._dynamo.config.suppress_errors = False

def read_file(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return f"File '{file_name}' does not exist in the current directory."

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        text = text.replace('-', ' ')
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def exact_match_score_single(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    
    if (normalized_prediction == "true" and normalized_ground_truth == "yes") or \
       (normalized_prediction == "false" and normalized_ground_truth == "no"):
        return 1
    
    return int(normalized_prediction == normalized_ground_truth)

def f1_score_single(prediction, ground_truth):
    normalized_prediction = normalize_answer(prediction)
    normalized_ground_truth = normalize_answer(ground_truth)
    ZERO_METRIC = (0, 0, 0)
    
    if (normalized_prediction == "true" and normalized_ground_truth == "yes") or \
       (normalized_prediction == "false" and normalized_ground_truth == "no"):
        return 1.0, 1.0, 1.0
    
    if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC
    if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
        return ZERO_METRIC

    prediction_tokens = normalized_prediction.split()
    ground_truth_tokens = normalized_ground_truth.split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return ZERO_METRIC

    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1, precision, recall

def exact_match_score_multi(prediction, ground_truths):
    normalized_prediction = normalize_answer(prediction)

    best_score = 0
    for gt in ground_truths:
        normalized_gt = normalize_answer(gt)
        if (normalized_prediction == "true" and normalized_gt == "yes") or \
           (normalized_prediction == "false" and normalized_gt == "no"):
            return 1 
        if normalized_prediction == normalized_gt:
            best_score = 1 
    return best_score

def f1_score_multi(prediction, ground_truths):
    normalized_prediction = normalize_answer(prediction)
    ZERO_METRIC = (0, 0, 0)
    best_f1 = 0
    best_precision = 0
    best_recall = 0

    for gt in ground_truths:
        normalized_gt = normalize_answer(gt)

        if (normalized_prediction == "true" and normalized_gt == "yes") or \
           (normalized_prediction == "false" and normalized_gt == "no"):
            return 1.0, 1.0, 1.0

        if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_gt:
            continue
        if normalized_gt in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_gt:
            continue

        pred_tokens = normalized_prediction.split()
        gt_tokens = normalized_gt.split()
        common = Counter(pred_tokens) & Counter(gt_tokens)
        num_same = sum(common.values())

        if num_same == 0:
            continue

        precision = num_same / len(pred_tokens)
        recall = num_same / len(gt_tokens)
        f1 = (2 * precision * recall) / (precision + recall)

        if f1 > best_f1:
            best_f1 = f1
            best_precision = precision
            best_recall = recall

    if best_f1 == 0:
        return ZERO_METRIC

    return best_f1, best_precision, best_recall

def cleaning_contexts(contexts):
    contexts_string = []
    for i, context in enumerate(contexts):
        tmp = f"Document [{i+1}] {context}\n"
        contexts_string.append(tmp)
    contexts_string = ''.join(contexts_string)
    return contexts_string

def cleaning_subqs(sub_questions):
    subqs = []
    for i, subq in enumerate(sub_questions):
        temp = f"SUB_Q{i+1}: {subq}\n"
        subqs.append(temp)
    subqs_string = ''.join(subqs)
    return subqs_string

def parse_triple_list(output_str):
    start = output_str.find('[')
    end = output_str.rfind(']')
    if start != -1 and end != -1 and end > start:
        substring = output_str[start:end+1]
        substring = substring.replace('\n', '')
        return substring
    return output_str

#######################
dataset = 'hotpotqa'
tag = 'with_sub_reason2' 
#######################

prompt_instructions = ['triple_verification.txt', 'inference.txt']
prompt_examples = ['examples_triple_verification.txt', 'examples_inference.txt']

prompt_base_dir = './prompt/Verification'

prompt_instructions = [f"{prompt_base_dir}/{file}" for file in prompt_instructions]
prompt_examples = [f"{prompt_base_dir}/{file}" for file in prompt_examples]

instruction1 = read_file(prompt_instructions[0])
instruction2 = read_file(prompt_instructions[1])
example1 = read_file(prompt_examples[0])
example2 = read_file(prompt_examples[1])

nltk.download('punkt')
model_name =  "/home/huggingface/models--google--gemma-2-9b-it/snapshots/checkpoint"  

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",        
    torch_dtype="bfloat16",
)

data_path = f'./Weight_Union/results/sLLM/retrieved_{dataset}.json'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(data_path, 'r') as file:
    data = json.load(file)
    print(f"Loading the data from {data_path}")
    print(f"Total Test Sample: {len(data)}")

new_data_list = []
names = f'1000 sampled_hippo({tag})'
path = f"results/res_verification_{dataset}.json"
os.makedirs(os.path.dirname(path), exist_ok=True)

print()
print("*"*50)
print(f"Preparing Done for {names}")
print(f"Total Sample: {len(data)}")
print("*"*50)
print()

tok_err = 0
all_em = []
all_f1 = []
all_time = 0
for cnt, item in enumerate(data):
    start_time = time.time()
    question = item['question']
    ground_truth = item['answer']
    sub_questions = item['sub_questions']
    triples = item['sub_triples']
    contexts = item['retrieved_docs']
    context = cleaning_contexts(contexts)
    sub_questions = cleaning_subqs(sub_questions)
    f1_score_list = []
    subq_set_list = []

    max_retry_complete = 3 
    retry_count_complete = 0
    complete_triples = ""
    while retry_count_complete < max_retry_complete:
        model_input1 = initial_input(instruction1, example1, context, question, sub_questions, triples)
        inputs1 = tokenizer.encode(model_input1, return_tensors='pt', max_length=4096, truncation=True).to(device)
        input_length1 = inputs1.shape[1]
        with torch.no_grad():
            output1 = model.generate(
                inputs1,
                max_new_tokens=1024,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated_tokens = output1[0][input_length1:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        output_text = re.sub(r'\\', '', output_text).strip()
        complete_triples = parse_triple_list(output_text)
        # Check if valid triple string is generated
        if complete_triples.startswith('[') and complete_triples.endswith(']') and complete_triples != "[]":
            break
        else:
            retry_count_complete += 1
            print(f"Retry complete_triples generation {retry_count_complete}...")
    if retry_count_complete == max_retry_complete:
        print("Warning: complete_triples still not valid after retries.")

    model_input = initial_input(instruction2, example2, context, question, sub_questions, complete_triples)
    inputs = tokenizer.encode(model_input, return_tensors='pt', max_length=4096, truncation=True).to(device)
    input_length = inputs.shape[1]
    with torch.no_grad():
        outputs = model.generate(
            inputs,
            max_new_tokens=256,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    max_retry = 3
    retry_count = 0
    answer_word = None
    generated_answer = ""
    
    while retry_count < max_retry:
        generated_tokens = outputs[0][input_length:]
        output_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        generated_answer = output_text.strip()
        
        if generated_answer.count("<<ANS>>") >= 2:
            answer_word = generated_answer.split("<<ANS>>")[1].strip()
        elif generated_answer.count("Generated Answer:") > 0:
            try:
                answer_word = generated_answer.split("Generated Answer:")[1].split("\n")[0].strip()
            except IndexError:
                answer_word = None
        else:
            answer_word = None
        
        if answer_word:
            break
        else:
            retry_count += 1
            print(f"Retry answer extraction {retry_count}...")
    
    if not answer_word:
        answer_word = generated_answer  
        tok_err += 1

    end_time = time.time()
    single_time = end_time - start_time

    if dataset == 'musique':
        true_answer_list = [ground_truth]
        true_answer_list = true_answer_list + item['answer_aliases']
        
        em_score = exact_match_score_multi(answer_word, true_answer_list)
        f1_score = f1_score_multi(answer_word, true_answer_list)[0]
    elif dataset == '2wikimultihop':
        true_answer_list = [ground_truth]
        true_answer_list = true_answer_list + item['aliases'] + item['demonyms']

        em_score = exact_match_score_multi(answer_word, true_answer_list)
        f1_score = f1_score_multi(answer_word, true_answer_list)[0]
    else:
        em_score = exact_match_score_single(answer_word, ground_truth)
        f1_score = f1_score_single(answer_word, ground_truth)[0]

    new_item = {
        "Question": question,
        'Documents': [doc.cpu() if torch.is_tensor(doc) else doc for doc in contexts],
        "Initial Response": generated_answer,
        "Final Answer": answer_word,
        "Ground_Truth": ground_truth,
        "Exact Match Score": em_score,
        "F1 Score": f1_score 
    }
    
    new_data_list.append(new_item)

    print()
    print("*"*50)
    print(f"Sample {cnt+1}")
    print()
    print(f"Complete Reasoning Chain:")
    print(f"{complete_triples}")
    print(f"Example Result:")
    print(f"{generated_answer}")
    print(f"Final Answer: {answer_word}")
    print(f"Ground Truth: {ground_truth}")
    print(f"Exact Match Score: {em_score}")
    print(f"F1 Score: {f1_score}")
    print(f"Inference Time: {single_time}")
    print("*" * 50)
    all_em.append(em_score)
    all_f1.append(f1_score)
    all_time += single_time
    if cnt % 1 == 0:
        with open(path, 'w') as file:
            json.dump(new_data_list, file, ensure_ascii=False, indent=4)
            print(f"Saving Done: {path}")
            print(f"Sample {cnt}/{len(data)}")
        torch.cuda.empty_cache()

print()
print("*"*50)
print(f"Token Error: {tok_err}")
print("Average of Exact Match & F1 Score")
print(f"Average EM: {sum(all_em)/len(all_em)}")
print(f"Average F1: {sum(all_f1)/len(all_f1)}")
print(f"Average Time: {all_time/len(data)}")
print("*"*50)
print()
with open(path, 'w') as file:
    json.dump(new_data_list, file, ensure_ascii=False, indent=4)
print(f"Saving Done: {path}")

