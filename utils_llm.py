import re
import ast
import json
import time
import string
import random
from collections import Counter
import openai
from openai.error import Timeout, RateLimitError, APIConnectionError, APIError

def read_file(file_name):
    try:
        with open(file_name, "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return f"File '{file_name}' does not exist in the current directory."

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

def factchecker(prompt):
    for attempt in range(3):
        try:
            response = openai.ChatCompletion.create(
                model="gpt-4.1-mini",
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
