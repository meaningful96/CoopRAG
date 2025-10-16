import argparse
import os
import json
import re
import time
import torch
import openai
from prompt_inputs import *
from utils_llm import (
    read_file, cleaning_contexts, cleaning_subqs, parse_triple_list,
    exact_match_score_single, f1_score_single,
    exact_match_score_multi, f1_score_multi,
    ask_gpt4
)

openai.api_key = os.environ["OPENAI_API_KEY"]

parser = argparse.ArgumentParser(description="Inference with GPT")
parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., freshllm, musique, 2wikimultihop)")
args = parser.parse_args()
dataset = args.dataset

prompt_instructions = ['completion.txt', 'inference.txt']
prompt_examples = ['examples_completion.txt', 'examples_inference.txt']

prompt_base_dir = './prompt/Completion'
prompt_instructions = [f"{prompt_base_dir}/{file}" for file in prompt_instructions]
prompt_examples = [f"{prompt_base_dir}/{file}" for file in prompt_examples]

instruction1 = read_file(prompt_instructions[0])
instruction2 = read_file(prompt_instructions[1])
example1 = read_file(prompt_examples[0])
example2 = read_file(prompt_examples[1])

data_path = f'./RaLa/results/retrieved_{dataset}.json'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

with open(data_path, 'r') as file:
    data = json.load(file)
    print(f"Loading the data from {data_path}")
    print(f"Total Test Sample: {len(data)}")

new_data_list = []

# Result Path
path = f"results/result_{dataset}.json"
os.makedirs(os.path.dirname(path), exist_ok=True)

print()
print("*"*50)
print(f"Preparing Done")
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
        model_input1 = verification_input(instruction1, example1, context, question, sub_questions, triples)
        output_text = ask_gpt4(model_input1)
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

    model_input = inference_input(instruction2, example2, context, question, sub_questions, complete_triples)
    generated_answer = ask_gpt4(model_input)
   
    retry_count = 0
    max_retry = 3
    while retry_count < max_retry:
        if generated_answer.count("<<FILL>>") >= 2:
            answer_word = generated_answer.split("<<FILL>>")[1].strip()
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
    if cnt % 100 == 0:
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

