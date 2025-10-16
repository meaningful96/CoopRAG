import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from doc import get_tokenizer

class BasePredictor(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.config = AutoConfig.from_pretrained(model_name, output_hidden_states=True)
        self.tokenizer = get_tokenizer(model_name)
        self.encoder = AutoModel.from_pretrained(model_name, config=self.config)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.pad_id = self.tokenizer.pad_token_id

    def encode_doc(self, input_ids, attention_mask):
        o = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                         return_dict=True, output_hidden_states=True)
        d_tok = o.hidden_states[-1]
        cls = d_tok[:, 0, :]
        norm_cls = F.normalize(cls, p=2, dim=1)  
        return norm_cls

    def encode_query(self, input_ids, attention_mask):
        o = self.encoder(input_ids=input_ids, attention_mask=attention_mask,
                         return_dict=True, output_hidden_states=True)
        q_tok = o.hidden_states[-1]
        cls = q_tok[:, 0, :]
        norm_cls = F.normalize(cls, p=2, dim=1)
        return norm_cls

    def calc_avg_maxsim(self, q_input_ids, q_attention_mask, d_input_ids, d_attention_mask):
        q_enc = self.encoder(q_input_ids, attention_mask=q_attention_mask,
                             return_dict=True, output_hidden_states=True)
        d_enc = self.encoder(d_input_ids, attention_mask=d_attention_mask,
                             return_dict=True, output_hidden_states=True)
        q_tok = q_enc.hidden_states[-1]
        d_tok = d_enc.hidden_states[-1]
        
        q_tok = F.normalize(q_tok, p=2, dim=-1)
        d_tok = F.normalize(d_tok, p=2, dim=-1)
        sim4d = q_tok.unsqueeze(1) @ d_tok.unsqueeze(0).transpose(-1, -2)

        qm = (q_input_ids == self.pad_id).unsqueeze(1).unsqueeze(-1)
        dm = (d_input_ids == self.pad_id).unsqueeze(0).unsqueeze(-2)
        sim4d = sim4d.masked_fill(qm | dm, float('-inf'))

        max_sim = sim4d.max(dim=-1)[0]  
        qm2 = (q_input_ids != self.pad_id).float().unsqueeze(1)
        sum_sim = (max_sim * qm2).sum(dim=-1)

        num_valid_tokens = (q_input_ids != self.pad_id).sum(dim=-1).unsqueeze(-1).float()
        avg_sim = sum_sim / (num_valid_tokens + 1e-10)
        return avg_sim



