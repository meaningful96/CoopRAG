import os
import glob
import torch
import shutil
import numpy as np
import torch.nn as nn
from logger_config import logger
import copy
import random
from config import args

class AttrDict:
    pass

def save_checkpoint(state: dict, is_best: bool, filename: str):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, os.path.dirname(filename) + '/model_best.mdl')
    shutil.copyfile(filename, os.path.dirname(filename) + '/model_last.mdl')

def delete_old_ckt(path_pattern: str, keep=5):
    files = sorted(glob.glob(path_pattern), key=os.path.getmtime, reverse=True)
    for f in files[keep:]:
        logger.info('Delete old checkpoint {}'.format(f))
        os.system('rm -f {}'.format(f))

def report_num_trainable_parameters(model: torch.nn.Module) -> int:
    num_parameters = 0
    for name, p in model.named_parameters():
        if p.requires_grad:
            num_parameters += np.prod(list(p.size()))
    logger.info('Number of parameters: {}M'.format(num_parameters // 1000000))
    return num_parameters

def get_model_obj(model: nn.Module):
    return model.module if hasattr(model, "module") else model

def move_to_cuda(sample):
    if len(sample) == 0:
        return {}
    def _move_to_cuda(maybe_tensor):
        if torch.is_tensor(maybe_tensor):
            return maybe_tensor.cuda(non_blocking=True)
        elif isinstance(maybe_tensor, dict):
            return {key: _move_to_cuda(value) for key, value in maybe_tensor.items()}
        elif isinstance(maybe_tensor, list):
            return [_move_to_cuda(x) for x in maybe_tensor]
        elif isinstance(maybe_tensor, tuple):
            return tuple(_move_to_cuda(x) for x in maybe_tensor)
        else:
            return maybe_tensor
    return _move_to_cuda(sample)

class AverageMeter:
    def __init__(self, name):
        self.name = name
        self.reset()
    def reset(self):
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0
    def update(self, val, n=1):
        self.val = float(val)
        self.sum += self.val * n
        self.count += n
        if self.count != 0:
            self.avg = self.sum / self.count
        else:
            self.avg = 0.0
    def __str__(self):
        if self.name != "InvT":
            return f"{self.name} {self.val:.4f} ({self.avg:.4f})"
        return f"{self.name} {self.val:.2f} ({self.avg:.2f})"

class ProgressMeter:
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
    def display(self, batch: int):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logger.info('\t'.join(entries))
    def _get_batch_fmtstr(self, num_batches: int) -> str:
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

def get_all_docs(input_data):
    res_dict = {}
    for ex in input_data:
        for doc in ex["paragraphs"]:
            t = doc["title"]
            txt = doc["paragraph_text"]
            if t not in res_dict:
                res_dict[t] = set()
            res_dict[t].add(txt)
    result = []
    for t, sset in res_dict.items():
        for st in sset:
            result.append({"title": t, "paragraph_text": st})
    return result

def Making_datasets(input_data, all_docs):
    data = copy.deepcopy(input_data)

    res = []
    cnt = 0
    for ex in data:
        center = copy.deepcopy(ex)
        docs = ex["paragraphs"]
        pos = [d for d in docs if d.get("is_supporting", False)]
        neg = [d for d in docs if not d.get("is_supporting", False)]
        
        sub_questions = ex['sub_questions']
        main_q = center['question']

        if "sub_triples" in ex:
            sub_triples = ex['sub_triples']
            if len(sub_triples) >= 1:
                sub_triples = [' '.join(map(str, triple)) for triple in sub_triples]
            else:
                sub_triples = ["<UNKNOWN>"]
            subt_string = '. '.join(sub_triples)
            question = main_q + ' ' + ' '.join(sub_questions) + ' ' + subt_string
        else:
            question = main_q + ' ' + ' '.join(sub_questions)
        if len(pos) > len(neg):
            neg_cand = random.sample(all_docs, len(pos))
            for i, pair in enumerate(pos):
                tmp = {}
                tmp['question'] = question
                para_pair = [pair, neg_cand[i]]
                tmp['paragraphs'] = para_pair
                tmp['sub_questions'] = sub_questions
                res.append(tmp)           
        else:
            neg_cand = random.sample(neg, len(pos))
            for i, pair in enumerate(pos):
                tmp = {}
                tmp['question'] = question
                para_pair = [pair, neg_cand[i]]
                tmp['paragraphs'] = para_pair
                tmp['sub_questions'] = sub_questions
                res.append(tmp)
    return res


def calculate_metrics_train(predicted, gold):
    if not gold:
        return 1.0, 1.0, 1.0, 1.0
    hit1 = 1.0 if predicted and predicted[0] in gold else 0.0
    top2 = predicted[:2]
    recall2 = len(set(top2) & set(gold)) / len(gold)
    top5 = predicted[:5]
    recall5 = len(set(top5) & set(gold)) / len(gold)
    top10 = predicted[:10]
    recall10 = len(set(top10) & set(gold)) / len(gold)
    return hit1, recall2, recall5, recall10

def calculate_metrics(predicted, gold):
    if not gold:
        return 1.0, 1.0, 1.0, 1.0, 1.0
    hit1 = 1.0 if predicted and predicted[0] in gold else 0.0
    top2 = predicted[:2]
    recall2 = len(set(top2) & set(gold)) / len(gold)
    top5 = predicted[:5]
    recall5 = len(set(top5) & set(gold)) / len(gold)
    k = min(5, len(gold))
    topk = predicted[:k]
    ps = set(topk)
    gs = set(gold)
    tp = len(ps & gs)
    fp = len(ps - gs)
    fn = len(gs - ps)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2*precision*rec / (precision+rec) if (precision+rec) > 0 else 0.0
    em = 1.0 if ps == gs else 0.0
    return hit1, recall2, recall5, f1, em

def calculate_retrieval_metrics(predicted, gold):

    if not gold:
        return 1.0, 1.0, 1.0, 1.0

    hit1 = 1.0 if (predicted and predicted[0] in gold) else 0.0
    top2 = predicted[:2]
    r2 = len(set(top2) & set(gold)) / len(gold)
    top5 = predicted[:5]
    r5 = len(set(top5) & set(gold)) / len(gold)
    top10 = predicted[:10]
    r10 = len(set(top10) & set(gold)) / len(gold)

    return hit1, r2, r5, r10
    
