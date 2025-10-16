import torch
import torch.utils.data
from transformers import AutoTokenizer
from typing import List, Dict

def get_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    special_tokens_dict = {'additional_special_tokens': ['[Question]','<FILL>', '<UNCERTAIN>', '[Passage]']}
    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, data: List[Dict], tokenizer_name: str, max_seq_length: int = 256):
        self.data = data
        self.tokenizer = get_tokenizer(tokenizer_name)
        self.max_seq_length = max_seq_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def collate_fn_inbatch_neg(self, batch: List[Dict]):
        B = len(batch)
        questions = []
        doc_texts = []
        subq_count_list = []
        for ex in batch:
            q_text = "[Question] " + ex["question"]
            questions.append(q_text)
            subq_count_list.append(len(ex["sub_questions"])) 
            for doc in ex["paragraphs"]:
                passage = doc["paragraph_text"].strip()
                title = doc["title"]
                d_text = "[Passage] " + title + ". " + passage
                doc_texts.append(d_text)

        q_enc = self.tokenizer(
            questions, padding="max_length", truncation=True,
            max_length=self.max_seq_length, return_tensors="pt"
        )
        d_enc = self.tokenizer(
            doc_texts, padding="max_length", truncation=True,
            max_length=self.max_seq_length, return_tensors="pt"
        )
        return {
            "q_input_ids": q_enc["input_ids"],
            "q_attention_mask": q_enc["attention_mask"],
            "d_input_ids": d_enc["input_ids"],
            "d_attention_mask": d_enc["attention_mask"],
            "subq_count": subq_count_list 
        }

