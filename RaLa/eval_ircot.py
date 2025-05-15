import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import re
import json
import faiss
import torch
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
from config import args
from logger_config import logger
from doc import get_tokenizer
from predictor import BasePredictor
from utils import calculate_retrieval_metrics
import openai
import time
from openai.error import Timeout, RateLimitError, APIConnectionError, APIError
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

def ask_gpt4(prompt):
    # Function to query GPT-4 with retries
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

def preprocess_doc(title, paragraph):
    # Preprocess document by combining title and paragraph
    title = title.strip()
    paragraph = paragraph.strip()
    if title:
        return f"(Title: {title}) {paragraph}"
    return paragraph

def parse_llm_output(text):
    # Regex pattern to match "(3, "Key Sentence"). So the answer is: SomeAnswer"
    pattern = r'^\(\s*(\d+)\s*,\s*"(.*)"\)\.\s*So the answer is:\s*(.+)$'
    match = re.search(pattern, text.strip())
    if match:
        doc_idx = int(match.group(1))
        key_sentence = match.group(2).strip()
        final_answer = match.group(3).strip()
        return doc_idx, key_sentence, final_answer
    else:
        return None, "", ""

def build_faiss_index(doc_embs: np.ndarray) -> faiss.IndexFlatIP:
    # Build FAISS index using inner product
    index = faiss.IndexFlatIP(doc_embs.shape[1])
    index.add(doc_embs)
    return index

def numbering_docs(data):
    # Placeholder for numbering_docs implementation if needed
    pass

def compute_doc_embeddings(docs: list, model: BasePredictor, tokenizer, device='cuda') -> np.ndarray:
    # Compute document embeddings in batches
    model.eval()
    all_embs = []
    batch_size = 512
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        enc = tokenizer(batch, padding="max_length", truncation=True,
                        max_length=args.max_seq_length, return_tensors="pt")
        enc = enc.to(device)
        with torch.no_grad():
            emb = model.encode_doc(enc["input_ids"], enc["attention_mask"])
            emb = emb.cpu().numpy()
        all_embs.append(emb)
    return np.concatenate(all_embs, axis=0)

def retrieve_top_k_queries(q_texts: list, q_model: BasePredictor, tokenizer,
                           faiss_index: faiss.IndexFlatIP, top_k: int, device='cuda'):
    # Retrieve top_k document indices based on query embeddings
    q_model.eval()
    batch_size = 512
    all_indices = []
    for i in range(0, len(q_texts), batch_size):
        sub_qs = q_texts[i:i+batch_size]
        enc = tokenizer(sub_qs, padding="max_length", truncation=True,
                        max_length=args.max_seq_length, return_tensors="pt")
        enc = enc.to(device)
        with torch.no_grad():
            q_emb = q_model.encode_query(enc["input_ids"], enc["attention_mask"])
            q_emb_np = q_emb.cpu().numpy()
        _, I = faiss_index.search(q_emb_np, top_k)
        all_indices.append(I)
    return np.concatenate(all_indices, axis=0)

def format_llm_documents(documents):
    """
    Format documents for LLM input.
    Each document is expected in the format: (Title: {title}) {paragraph}
    This function prepends "Document {i}: " for each document.
    """
    formatted_docs = []
    for i, doc in enumerate(documents, start=1):
        formatted_docs.append(f"Document {i}: {doc}\n")
    return ''.join(formatted_docs)

def iterative_ircot(query: str, predictor: BasePredictor, tokenizer, full_faiss_index: faiss.IndexFlatIP,
                    doc_map: dict, doc_embs: np.ndarray, device='cuda', max_selections: int = 5):
    """
    Perform iterative IRCoT process.
    Returns:
        final_selected_docs (list): top-5 documents selected
        iteration_count (int): how many iterations were performed
    """
    selected_doc_set = set()        # set of indices for selected documents
    accumulated_key_sentences = []  # list of key sentences for query update
    final_selected_docs = []        # final top-5 documents
    current_query = query
    candidate_indices = None
    iteration_count = 0             # track how many iterations

    while len(final_selected_docs) < max_selections:
        iteration_count += 1  # increment iteration
        indices = retrieve_top_k_queries([current_query], predictor, tokenizer, full_faiss_index, top_k=100, device=device)
        candidate_indices = [idx for idx in indices[0].tolist() if idx not in selected_doc_set]
        
        if len(candidate_indices) < 5:
            top5_indices = candidate_indices
        else:
            top5_indices = candidate_indices[:5]
        
        candidate_docs = [doc_map[idx] for idx in top5_indices]
        formatted_docs = format_llm_documents(candidate_docs)
        
        prompt_template_path = os.path.join("..", "prompt", "ircot.txt")
        if prompt_template_path and os.path.exists(prompt_template_path):
            with open(prompt_template_path, 'r', encoding='utf-8') as f:
                prompt_template = f.read()
        else:
            prompt_template = (
                "You are an assistant specialized in multi-hop question answering using a Beam Retrieval framework. "
                "Your task is to iteratively select the most helpful document among the provided documents. "
                "Follow these steps and output in the exact format:\n"
                "(Document {i}, \"Key Sentence\"). So the answer is: {final answer}\n\n"
                "If the final answer cannot be deduced, output False for final answer."
            )
        
        full_prompt = f"""
{prompt_template}

Question: {current_query}

{formatted_docs}
        """
        print("="*100)
        print(f"[Iteration {iteration_count}] Query: {current_query}")
        inference_text = ask_gpt4(full_prompt)
        print("LLM Output:")
        print(inference_text)
        print("="*100)
        
        doc_number, key_sentence, final_answer = parse_llm_output(inference_text)
        if doc_number is None:
            print("LLM output parsing failed. Stopping iteration.")
            break
        
        try:
            selected_doc_idx = top5_indices[doc_number - 1]
        except IndexError:
            print("Selected document number is out of candidate range. Stopping iteration.")
            break
        
        final_selected_docs.append(doc_map[selected_doc_idx])
        selected_doc_set.add(selected_doc_idx)
        
        if final_answer != "False":
            print(f"Final answer deduced: {final_answer}")
            break
        else:
            if key_sentence not in accumulated_key_sentences:
                accumulated_key_sentences.append(key_sentence)
            current_query = query + " " + " ".join(accumulated_key_sentences)
            print(f"Updated query: {current_query}")
    
    # Fill remaining documents if fewer than max_selections
    if len(final_selected_docs) < max_selections:
        remaining_needed = max_selections - len(final_selected_docs)
        for idx in top5_indices:
            if idx not in selected_doc_set and remaining_needed > 0:
                final_selected_docs.append(doc_map[idx])
                selected_doc_set.add(idx)
                remaining_needed -= 1
            if remaining_needed == 0:
                break
    
    return final_selected_docs, iteration_count

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    logger.info("Creating BasePredictor...")
    predictor = BasePredictor(args.pretrained_model)
    if getattr(args, 'checkpoint_path', None) and os.path.exists(args.checkpoint_path):
        logger.info(f"Loading checkpoint from {args.checkpoint_path} ...")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        if 'state_dict' in checkpoint:
            predictor.load_state_dict(checkpoint['state_dict'], strict=False)
        else:
            predictor.load_state_dict(checkpoint, strict=False)
        logger.info("Checkpoint loaded.")
    else:
        logger.info("No valid checkpoint_path found. Using random init model.")

    predictor.to(device)
    predictor.eval()

    logger.info("Loading test data...")
    with open(args.test_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)

    logger.info("Preparing docs...")
    temp_list = []
    for ex in test_data:
        if "paragraphs" in ex:
            for p in ex["paragraphs"]:
                title = p.get("title", "")
                paragraph = p["paragraph_text"]
                temp_list.append({"title": title, "paragraph_text": paragraph})
    unique_map = {}
    for doc in temp_list:
        pt = doc["paragraph_text"]
        if pt not in unique_map:
            unique_map[pt] = doc["title"]
    final_docs = []
    for para_t, ttl in unique_map.items():
        final_docs.append({"title": ttl, "paragraph_text": para_t})
    logger.info(f"Unique documents count: {len(final_docs)}")
    docs_for_embedding = []
    for fd in final_docs:
        doc_str = preprocess_doc(fd["title"], fd["paragraph_text"])
        docs_for_embedding.append(doc_str)
    logger.info("Computing embeddings...")
    doc_embs = compute_doc_embeddings(docs_for_embedding, predictor, predictor.tokenizer, device=device)
    logger.info("Building FAISS index...")
    full_faiss_index = build_faiss_index(doc_embs)
    doc_map = {i: docs_for_embedding[i] for i in range(len(docs_for_embedding))}
    str_to_index = {docs_for_embedding[i]: i for i in range(len(docs_for_embedding))}
    def doc_str_to_index(d_str: str):
        return str_to_index[d_str] if d_str in str_to_index else None

    # Variables to track statistics
    total_samples = 0
    r2_total = 0.0
    r5_total = 0.0
    total_iterations = 0
    iteration_dict = {}  # iteration_count -> number of samples

    for sample_idx, ex in enumerate(tqdm(test_data, desc="Processing"), 1):
        main_question = ex.get("question", "").strip()
        if not main_question:
            continue

        query = main_question
        if "sub_questions" in ex and ex["sub_questions"]:
            query += " " + " ".join([sq.strip() for sq in ex["sub_questions"] if sq.strip()])

        if 'sub_triples' in ex and ex['sub_triples']:
            sub_triples = ex['sub_triples']
            subT_strings = [' '.join(triple) for triple in sub_triples]
            subT = '. '.join(subT_strings)
            query += subT
        
        gold_indices = []
        if "paragraphs" in ex:
            for p in ex["paragraphs"]:
                if p.get("is_supporting", False):
                    raw_para = p["paragraph_text"]
                    raw_title = p.get("title", "")
                    gold_str = preprocess_doc(raw_title, raw_para)
                    idx_ = doc_str_to_index(gold_str)
                    if idx_ is not None:
                        gold_indices.append(idx_)

        if not gold_indices:
            continue

        total_samples += 1
        print(f"Sample {sample_idx} - Question: {query}")

        final_docs_selected, iteration_count = iterative_ircot(
            query, predictor, predictor.tokenizer,
            full_faiss_index, doc_map, doc_embs,
            device=device, max_selections=5
        )
        
        # Convert final_docs_selected to their indices
        final_indices = []
        for doc_str in final_docs_selected:
            idx_ = doc_str_to_index(doc_str)
            if idx_ is not None:
                final_indices.append(idx_)

        ex["retrieved_docs"] = final_docs_selected
        # Compute R@2, R@5
        _, r2, r5, _ = calculate_retrieval_metrics(final_indices, gold_indices)
        r2_total += r2
        r5_total += r5
        print(f"Recall@2 = {r2}, Recall@5 = {r5}")
        print("==================================================")

        # Track iteration count distribution
        if iteration_count not in iteration_dict:
            iteration_dict[iteration_count] = 0
        iteration_dict[iteration_count] += 1
        total_iterations += iteration_count

    # Print overall statistics
    if total_samples > 0:
        avg_r2 = r2_total / total_samples
        avg_r5 = r5_total / total_samples
        avg_iteration = total_iterations / total_samples
        print(f"Total samples: {total_samples}")
        print(f"Average R@2: {avg_r2:.4f}, Average R@5: {avg_r5:.4f}")
        print(f"Average iteration count: {avg_iteration:.2f}")
        print(f"Iteration distribution: {iteration_dict}")

        logger.info(f"IRCoT Evaluation: #samples={total_samples}")
        logger.info(f"R@2={avg_r2:.4f}, R@5={avg_r5:.4f}")
        logger.info(f"Average Iterations: {avg_iteration:.2f}")
        iteration_dict = dict(sorted(iteration_dict.items()))
        logger.info(f"Iteration Distribution: {iteration_dict}")
    else:
        logger.info("No valid samples found in test_data")

    result_path = args.results_path if args.results_path is not None else "ircot_result.json"
    logger.info(f"Saving result to {result_path}")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

