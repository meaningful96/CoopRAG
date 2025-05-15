import os
import json
import faiss
import torch
import numpy as np
from typing import List
from predictor import BasePredictor
from doc import get_tokenizer
from utils import calculate_retrieval_metrics
from config import args
from logger_config import logger
from transformers import AutoTokenizer
from tqdm import tqdm

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

def build_faiss_index(doc_embs: np.ndarray) -> faiss.IndexFlatIP:
    # minimal English comment
    index = faiss.IndexFlatIP(doc_embs.shape[1])
    index.add(doc_embs)
    return index

def compute_doc_embeddings(docs: List[str], model: BasePredictor, tokenizer, device='cuda') -> np.ndarray:
    model.eval()
    all_embs = []
    batch_size = 512
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        enc = tokenizer(batch, padding=True, truncation=True, max_length=args.max_seq_length, return_tensors="pt")
        enc = enc.to(device)
        with torch.no_grad():
            emb = model.encode_doc(enc["input_ids"], enc["attention_mask"])
            emb = emb.cpu().numpy()
        all_embs.append(emb)
    return np.concatenate(all_embs, axis=0)

def retrieve_top_k_queries(q_texts: List[str], q_model: BasePredictor, tokenizer,
                           faiss_index: faiss.IndexFlatIP, top_k: int = 10, device='cuda'):
    # minimal English comment
    q_model.eval()
    batch_size = 512
    all_indices = []
    for i in range(0, len(q_texts), batch_size):
        sub_qs = q_texts[i:i+batch_size]
        enc = tokenizer(sub_qs, padding=True, truncation=True, max_length=args.max_seq_length, return_tensors="pt")
        enc = enc.to(device)
        with torch.no_grad():
            q_emb = q_model.encode_query(enc["input_ids"], enc["attention_mask"])
            q_emb_np = q_emb.cpu().numpy()
        D, I = faiss_index.search(q_emb_np, top_k)
        all_indices.append(I)
    return np.concatenate(all_indices, axis=0)

def rerank_top_docs_avgmaxsim(query: str, doc_list: List[str],
                              predictor: BasePredictor, tokenizer,
                              device='cuda', topn=5):
    # minimal English comment
    sims = []
    batch_size = 100
    total_docs = len(doc_list)

    for i in range(0, total_docs, batch_size):
        sub_docs = doc_list[i:i+batch_size]
        enc_q = tokenizer([query]*len(sub_docs), padding=True, truncation=True,
                          max_length=args.max_seq_length, return_tensors="pt")
        enc_d = tokenizer(sub_docs, padding=True, truncation=True,
                          max_length=args.max_seq_length, return_tensors="pt")
        enc_q = enc_q.to(device)
        enc_d = enc_d.to(device)
        with torch.no_grad():
            score_mat = predictor.calc_avg_maxsim(
                enc_q["input_ids"], enc_q["attention_mask"],
                enc_d["input_ids"], enc_d["attention_mask"]
            )
            s_ = score_mat.squeeze(0).cpu().tolist()
        sims.extend(s_)

    scored_list = sorted(zip(doc_list, sims), key=lambda x: x[1], reverse=True)
    return [d for d, _ in scored_list[:topn]]


def rerank_top_docs_avgmaxsim_mmr(query: str, doc_list: List[str],
                              predictor: BasePredictor, tokenizer,
                              device='cuda', topn=5):
    lambda_val = 0.5
    total_docs = len(doc_list)
    
    # Compute query-document similarity for all docs at once
    enc_q = tokenizer([query] * total_docs, padding=True, truncation=True,
                      max_length=args.max_seq_length, return_tensors="pt").to(device)
    enc_d = tokenizer(doc_list, padding=True, truncation=True,
                      max_length=args.max_seq_length, return_tensors="pt").to(device)
    with torch.no_grad():
        qd_score_mat = predictor.calc_avg_maxsim(enc_q["input_ids"], enc_q["attention_mask"],
                                                 enc_d["input_ids"], enc_d["attention_mask"])
    # Assume output shape (total_docs, 1)
    q_sims = [s[0] for s in qd_score_mat.cpu().tolist()]
    
    # Compute document-document similarity matrix in one call
    enc_docs = tokenizer(doc_list, padding=True, truncation=True,
                         max_length=args.max_seq_length, return_tensors="pt")
    for k in enc_docs:
        enc_docs[k] = enc_docs[k].to(device)
    with torch.no_grad():
        doc_sim_matrix = predictor.calc_avg_maxsim(enc_docs["input_ids"], enc_docs["attention_mask"],
                                                   enc_docs["input_ids"], enc_docs["attention_mask"])
    # Assume output shape (total_docs, total_docs)
    sim_matrix = doc_sim_matrix.cpu().tolist()
    
    # MMR selection
    selected_indices = []
    candidate_indices = list(range(total_docs))
    while len(selected_indices) < topn and candidate_indices:
        mmr_scores = []
        for idx in candidate_indices:
            score_query = q_sims[idx]
            if selected_indices:
                max_sim = max(sim_matrix[idx][s_idx] for s_idx in selected_indices)
            else:
                max_sim = 0.0
            mmr = lambda_val * score_query - (1 - lambda_val) * max_sim
            mmr_scores.append((idx, mmr))
        best_idx, _ = max(mmr_scores, key=lambda x: x[1])
        selected_indices.append(best_idx)
        candidate_indices.remove(best_idx)
    
    return [doc_list[idx] for idx in selected_indices]

def preprocess_doc(title_str: str, para_str: str) -> str:
    title_str = title_str.strip()
    para_str = para_str.strip()
    if title_str:
        return f"(Title: {title_str}) {para_str}"
    else:
        return para_str

def recall_at_100(top100_indices: List[int], gold_indices: List[int]) -> float:
    if not gold_indices:
        return 0.0
    top100_set = set(top100_indices)
    gold_set = set(gold_indices)
    inter = len(top100_set & gold_set)
    return inter / len(gold_set)

def main():

    dir_path = './results/Departed'
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    logger.info(f"Save Path: {args.results_path}")
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

    logger.info("Preparing docs (text-based dedup)...")
    temp_list = []
    for ex in test_data:
        # collect paragraphs
        if "paragraphs" in ex:
            for p_ in ex["paragraphs"]:
                title_ = p_.get("title", "")
                paragraph_ = p_["paragraph_text"]
                temp_list.append({"title": title_, "paragraph_text": paragraph_})

    unique_map = {}
    for doc_ in temp_list:
        pt = doc_["paragraph_text"]
        if pt not in unique_map:
            unique_map[pt] = doc_["title"]

    final_docs = []
    for para_t, ttl in unique_map.items():
        final_docs.append({"title": ttl, "paragraph_text": para_t})

    logger.info("Unique documents count: %d" % len(final_docs))
    docs_for_embedding = []
    for fd in final_docs:
        doc_str = preprocess_doc(fd["title"], fd["paragraph_text"])
        docs_for_embedding.append(doc_str)

    logger.info("Computing embeddings...")
    doc_embs = compute_doc_embeddings(docs_for_embedding, predictor, predictor.tokenizer, device=device)
    logger.info("Building Faiss index...")
    faiss_index = build_faiss_index(doc_embs)

    doc_map = {i: docs_for_embedding[i] for i in range(len(docs_for_embedding))}
    str_to_index = {docs_for_embedding[i]: i for i in range(len(docs_for_embedding))}

    def doc_str_to_index(d_str: str):
        return str_to_index[d_str] if d_str in str_to_index else None

    total_samples = 0
    # metrics
    h1_case4 = r2_case4 = r3_case4 = r5_case4 = 0.0
    r100_case4 = 0.0

    for ex in tqdm(test_data, desc="Processing"):
        main_q = ex.get("question", "").strip()
        if not main_q:
            continue

        # gather gold docs
        gold_indices = []
        if "paragraphs" in ex:
            for p_ in ex["paragraphs"]:
                if p_.get("is_supporting", False):
                    raw_para = p_["paragraph_text"]
                    raw_title = p_.get("title", "")
                    gold_str = preprocess_doc(raw_title, raw_para)
                    idx_ = doc_str_to_index(gold_str)
                    if idx_ is not None:
                        gold_indices.append(idx_)

        if not gold_indices:
            continue

        total_samples += 1

        # build concat_q
        concat_q = main_q
        
        if 'sub_questions' in ex:
            for subq in ex['sub_questions']:
                concat_q += " " + subq 
        
        if 'sub_triples' in ex:
            sub_triples = ex['sub_triples']
            subT_strings = [' '.join(triple) for triple in sub_triples]
            subT = '. '.join(subT_strings)    
            concat_q += subT

        # retrieve top-100
        I_c4 = retrieve_top_k_queries([concat_q], predictor, predictor.tokenizer,
                                      faiss_index, top_k=100, device=device)
        idx_list_c4 = I_c4[0]
        # recall@100
        rec100_4 = recall_at_100(idx_list_c4, gold_indices)
        r100_case4 += rec100_4

        # rerank top-5
        cand_texts_c4 = [doc_map[idx] for idx in idx_list_c4]
        top5_c4 = rerank_top_docs_avgmaxsim(concat_q, cand_texts_c4, predictor, predictor.tokenizer,
                                            device=device, topn=5)

        final_idx_c4 = [doc_str_to_index(d_) for d_ in top5_c4 if doc_str_to_index(d_) is not None]

        # store retrieved docs in ex
        ex["retrieved_docs"] = top5_c4

        # calculate metrics
        h1_4, rr2_4, rr5_4, rr10_4 = calculate_retrieval_metrics(final_idx_c4, gold_indices)

        # custom r@3
        r3_4 = 0
        if len(final_idx_c4) >= 3:
            gold_set = set(gold_indices)
            top3_set = set(final_idx_c4[:3])
            inter = len(gold_set & top3_set)
            if len(gold_set) > 0:
                r3_4 = float(inter) / len(gold_set)

        h1_case4 += h1_4
        r2_case4 += rr2_4
        r5_case4 += rr5_4
        r3_case4 += r3_4

    if total_samples > 0:
        logger.info("[Case4] #samples=%d" % total_samples)
        logger.info(" Hit@1=%.4f, R@2=%.4f, R@3=%.4f, R@5=%.4f, R@100=%.4f" % (
            h1_case4/total_samples,
            r2_case4/total_samples,
            r3_case4/total_samples,
            r5_case4/total_samples,
            r100_case4/total_samples
        ))
    else:
        logger.info("No valid samples found in test_data")

    # save result
    result_path = args.results_path
    if result_path is None:
        result_path = "case4_result.json"
    logger.info(f"Saving result to {result_path}")
    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(test_data, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()

