import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from dataclasses import dataclass
from typing import Dict, List
from doc import get_tokenizer

def L2_norm(matrix):
    return F.normalize(matrix, p=2, dim=-1)

@dataclass
class ModelOutput:
    logits: torch.Tensor
    labels: torch.Tensor
    logits_pos: torch.Tensor
    logits_neg: torch.Tensor
    logits_weighted: torch.Tensor
    inv_t: torch.Tensor
    Ws: torch.Tensor
    weight: torch.Tensor

class CustomRetrieverModel(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.pretrained_model, output_hidden_states=True)
        self.tokenizer = get_tokenizer(args.pretrained_model)
        self.pad_id = self.tokenizer.pad_token_id
        self.encoder = AutoModel.from_pretrained(args.pretrained_model, config=self.config)
        self.encoder.resize_token_embeddings(len(self.tokenizer))
        self.log_inv_t = nn.Parameter(torch.tensor(1 / args.t).log(), requires_grad=True)
        self.cls = True

    def _encode(self, input_ids, attention_mask):
        o = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True
        )
        return o.hidden_states

    def filter_punctuation(self, embeddings, input_ids):
        p = [',', '.', '?', '!', ':', ';', '-', '(', ')', '"', "'"]
        p_ids = set(self.tokenizer.convert_tokens_to_ids(p))
        m = torch.ones_like(input_ids, dtype=torch.bool)
        for pid in p_ids:
            m &= (input_ids != pid)
        m = m.unsqueeze(-1)
        return embeddings * m.float()

    def forward(self, q_input_ids, q_attention_mask, d_input_ids, d_attention_mask):
        q_h = self._encode(q_input_ids, q_attention_mask)
        d_h = self._encode(d_input_ids, d_attention_mask)
        q_tok = q_h[-1]
        d_tok = d_h[-1]
        d_tok = self.filter_punctuation(d_tok, d_input_ids)
        
        q_tok = F.normalize(q_tok, p=2, dim=-1)
        d_tok = F.normalize(d_tok, p=2, dim=-1)
       
        return {
            "q_tok": q_tok,
            "d_tok": d_tok,
            "q_h": q_h,
            "d_h": d_h
        }

    def fixed_weight_query(self, q_hidden_states, d_hidden_states, layers: list):
        sel_q = [q_hidden_states[i] for i in layers]
        sel_d = [d_hidden_states[i] for i in layers]
        if self.cls:
            sel_q = [x[:, 0, :] for x in sel_q]
            sel_d = [x[:, 0, :] for x in sel_d]
        else:
            sel_q = [x.mean(dim=1) for x in sel_q]
            sel_d = [x.mean(dim=1) for x in sel_d]
        q_last = L2_norm(sel_q[-1])
        d_last = L2_norm(sel_d[-1])
        sel_q = [L2_norm(i) for i in sel_q]
        sel_d = [L2_norm(j) for j in sel_d]

        center = q_last.mm(d_last.t())
        cand_sims = [q_last.mm(d.t()).unsqueeze(-1) for d in sel_d[:-1]]
        cand_sims = torch.cat(cand_sims, dim=-1)
        min_doc = cand_sims.min(dim=-1)[0]
        W_query = (center - min_doc)/2
        return W_query

    def compute_logits(self, output_dict: dict, batch_dict: dict) -> ModelOutput:
        q_tok = output_dict["q_tok"] 
        d_tok = output_dict["d_tok"]
        q_h = output_dict["q_h"]
        d_h = output_dict["d_h"]
        q_ids = batch_dict["q_input_ids"]
        d_ids = batch_dict["d_input_ids"]
        B, Lq, H = q_tok.size()
        M = d_tok.size(0)  # M = 2B

        sim4d = q_tok.unsqueeze(1) @ d_tok.unsqueeze(0).transpose(-1, -2)
        # sim4d: (B, M, Lq, Ld)
        qm = (q_ids == self.pad_id).unsqueeze(1).unsqueeze(-1)  # (B, 1, Lq-1, 1)
        dm = (d_ids == self.pad_id).unsqueeze(0).unsqueeze(-2)  # (1, M, 1, Ld-1)

        sim4d = sim4d.masked_fill(qm | dm, -1e-9)

        max_sim = sim4d.max(dim=-1)[0]
        qm2 = (q_ids != self.pad_id).float().unsqueeze(1)
        sum_sim = (max_sim * qm2).sum(dim=-1)

        num_valid_tokens = (q_ids != self.pad_id).sum(dim=-1)
        num_valid_tokens = num_valid_tokens.unsqueeze(-1).float()
        avg_sim = sum_sim / (num_valid_tokens + 1e-10) 
        logits_all = avg_sim

        del sum_sim

        logits_pos = logits_all[:, 0::2]
        logits_neg = logits_all[:, 1::2]
        logits = torch.cat([logits_pos, logits_neg], dim=1)
        Wq = self.fixed_weight_query(q_h, d_h, self.args.layers)
        pos_wq = Wq[:, 0::2]
        neg_wq = Wq[:, 1::2]
        wq_all = torch.cat([pos_wq, neg_wq], dim=1)
        logits_weighted = logits * wq_all
        inv_t = self.log_inv_t.exp()
        logits_final = logits_weighted * inv_t
        labels = torch.arange(B, device=logits.device)
      
        subq_count = batch_dict['subq_count']
        ws = torch.tensor(subq_count, device=logits.device, dtype=torch.float) 
        ws = (ws + 1.0).log() 
        return ModelOutput(
            logits=logits_final,
            labels=labels,
            logits_pos=logits_pos,
            logits_neg=logits_neg,
            logits_weighted = logits_weighted, 
            inv_t=inv_t.detach(),
            Ws = ws,
            weight = wq_all
        )


