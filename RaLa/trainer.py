import os
import torch
import torch.nn as nn
import torch.utils.data
import json
import random
import time
from torch.cuda.amp import autocast, GradScaler
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup, AutoModelForSequenceClassification, AutoTokenizer
from logger_config import logger
from utils import AverageMeter, ProgressMeter, save_checkpoint, delete_old_ckt, report_num_trainable_parameters, move_to_cuda, get_model_obj
from utils import calculate_metrics_train, get_all_docs, Making_datasets
from doc import CustomDataset
from models import CustomRetrieverModel, ModelOutput
from config import args
import copy
import datetime
import numpy as np

def mean_tensor(matrix):
    return torch.mean(matrix, dim=0)

class Trainer:
    def __init__(self, args):
        self.args = args
        logger.info("=> creating model")
        self.model = CustomRetrieverModel(args)
        if torch.cuda.is_available():
            self.model.cuda()
        if torch.cuda.device_count() > 1:
            logger.info(f"Using DataParallel on {torch.cuda.device_count()} GPUs.")
            self.model = nn.DataParallel(self.model)

        report_num_trainable_parameters(self.model)
        self.optimizer = torch.optim.AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=args.weight_decay
        )
        self.scaler = GradScaler(enabled=args.use_amp)
        self.train_data = json.load(open(args.train_path, 'r', encoding='utf-8'))
        self.valid_data = json.load(open(args.valid_path, 'r', encoding='utf-8'))
        self.all_doc_list = get_all_docs(self.train_data)
        self.all_doc_valid = get_all_docs(self.valid_data)
        self.train_loss = []
        self.valid_loss = []
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.best_metric = None

        steps_per_epoch = len(self.train_data) // args.batch_size
        total_steps = steps_per_epoch * args.epochs
        if args.lr_scheduler == 'linear':
            self.scheduler = get_linear_schedule_with_warmup(self.optimizer, min(args.warmup, total_steps), total_steps)
        elif args.lr_scheduler == 'cosine':
            self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, min(args.warmup, total_steps), total_steps)
        else:
            raise ValueError("Unknown scheduler")

    def train_loop(self):
        for epoch in range(self.args.epochs):
            s1 = time.time()
            new_data = Making_datasets(self.train_data, self.all_doc_list)
            random.shuffle(new_data)
            e1 = time.time()
            logger.info(f"Done building training datasets: {len(new_data)}")
            logger.info(f"Time: {datetime.timedelta(seconds=e1 - s1)}")
            train_dataset = CustomDataset(data=new_data, tokenizer_name=self.args.pretrained_model, max_seq_length=self.args.max_seq_length)
            train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=self.args.batch_size,
                collate_fn=train_dataset.collate_fn_inbatch_neg,
                shuffle=False, drop_last=True, num_workers=4
            )
            s2 = time.time()
            new_data_valid = Making_datasets(self.valid_data, self.all_doc_valid)
            random.shuffle(new_data_valid)
            e2 = time.time()
            logger.info(f"Done building validation datasets: {len(new_data_valid)}")
            logger.info(f"Time: {datetime.timedelta(seconds=e2 - s2)}")
            valid_dataset = CustomDataset(data=new_data_valid, tokenizer_name=self.args.pretrained_model, max_seq_length=self.args.max_seq_length)
            valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=self.args.batch_size,
                collate_fn=valid_dataset.collate_fn_inbatch_neg,
                shuffle=False, drop_last=True, num_workers=4
            )
            self.train_epoch(epoch, train_loader)
            self._run_eval(epoch, valid_loader)

    def train_epoch(self, epoch, train_loader):
        self.model.train()
        start_time = time.time()
        losses = AverageMeter('Loss')
        t_meter = AverageMeter('InvT')
        r1_meter = AverageMeter('R@1')
        r2_meter = AverageMeter('R@2')
        r5_meter = AverageMeter('R@5')
        progress = ProgressMeter(len(train_loader), [losses, t_meter, r1_meter, r2_meter, r5_meter],
                                 prefix=f"Epoch: [{epoch}]")

        # tmp = {"logits_pos":[], "logits_neg": [], 'weight': [], 'logits_fin': []}
        for i, batch in enumerate(train_loader):
            if torch.cuda.is_available():
                batch = move_to_cuda(batch)
            with autocast(enabled=self.args.use_amp, dtype=torch.float16):
                fwd_out = self.model(
                    q_input_ids=batch["q_input_ids"],
                    q_attention_mask=batch["q_attention_mask"],
                    d_input_ids=batch["d_input_ids"],
                    d_attention_mask=batch["d_attention_mask"]
                )
                outs = get_model_obj(self.model).compute_logits(fwd_out, batch)
                logits = outs.logits
                labels = outs.labels

                Ws = outs.Ws
                loss_fwd = self.criterion(logits, labels) * Ws
                loss_main = mean_tensor(loss_fwd).to(logits.device)
                loss_bwd = self.criterion(logits[:, :self.args.batch_size].t(), labels)
                loss_infonce = mean_tensor(loss_bwd).to(logits.device)
                loss_total = loss_main + loss_infonce

            t_meter.update(outs.inv_t, 1)
    
            """
            g1, g2 = outs.logits_pos.cpu(), outs.logits_neg.cpu()
            g3, g4 = outs.weight.cpu(), outs.logits_weighted.cpu()

            # g1, g2 = (B, B) tensor
            # g3, g4 = (B, M) tensor

            tmp['logits_pos'].append(g1.detach().numpy())
            tmp['logits_neg'].append(g2.detach().numpy())
            tmp['weight'].append(g3.detach().numpy())
            tmp['logits_fin'].append(g4.detach().numpy())

            if len(tmp['logits_pos']) == 100:
                np.savez(f"mpnet_{self.args.task}.npz", **tmp)
                print("Saving Done")
            """    
            self.optimizer.zero_grad()
            if self.args.use_amp:
                self.scaler.scale(loss_total).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss_total.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_clip)
                self.optimizer.step()

            self.scheduler.step()
            losses.update(loss_total.item(), logits.size(0))
            preds_sorted = torch.argsort(logits, dim=-1, descending=True)
            b_ = logits.size(0)
            for b_idx in range(b_):
                gold_indices = [b_idx]
                pred_ = preds_sorted[b_idx].tolist()
                r1v, r2v, r5v, _ = calculate_metrics_train(pred_, gold_indices)
                r1_meter.update(r1v, 1)
                r2_meter.update(r2v, 1)
                r5_meter.update(r5v, 1)
            if i % self.args.print_freq == 0 and i < len(train_loader) - 1:
                progress.display(i)

        end_time = time.time()
        logger.info(f"Epoch {epoch} done in {datetime.timedelta(seconds=end_time - start_time)}")
        logger.info(f"Epoch {epoch} final step: Loss={losses.avg:.4f}, R@1={r1_meter.avg:.4f}, R@2={r2_meter.avg:.4f}, R@5={r5_meter.avg:.4f}")
        self.train_loss.append(round(losses.avg, 3))
        print("Train_loss = {}".format(self.train_loss))

    @torch.no_grad()
    def _run_eval(self, epoch, valid_loader):
        if not valid_loader:
            logger.info("No valid_loader. Skipping evaluation.")
            return
        metric_dict = self.eval_epoch(epoch, valid_loader)
        if (self.best_metric is None) or (metric_dict['r5'] >= self.best_metric['r5']):
            self.best_metric = metric_dict
        c = {
            'epoch': epoch,
            'args': self.args.__dict__,
            'state_dict': get_model_obj(self.model).state_dict(),
        }
        fn = f"{self.args.model_dir}/checkpoint_epoch{epoch}.mdl"
        torch.save(c, fn)
        delete_old_ckt(path_pattern=f"{self.args.model_dir}/checkpoint_*.mdl", keep=self.args.max_to_keep)
        if (self.best_metric is None) or (metric_dict['r5'] >= self.best_metric['r5']):
            self.best_metric = metric_dict
            bn = f"{self.args.model_dir}/model_best.mdl"
            torch.save(c, bn)
            logger.info(f"Saving model_best at epoch {epoch}")
        if epoch == self.args.epochs - 1:
            ln = f"{self.args.model_dir}/model_last.mdl"
            torch.save(c, ln)
            logger.info("Saving model_last.")

    @torch.no_grad()
    def eval_epoch(self, epoch, valid_loader):
        self.model.eval()
        total_loss = 0.0
        total_count = 0
        total_samples = 0
        rec1_sum = 0.0
        rec2_sum = 0.0
        rec5_sum = 0.0
        for batch in valid_loader:
            if torch.cuda.is_available():
                batch = move_to_cuda(batch)
            with autocast(enabled=self.args.use_amp, dtype=torch.float16):
                fwd_out = self.model(
                    q_input_ids=batch["q_input_ids"], q_attention_mask=batch["q_attention_mask"],
                    d_input_ids=batch["d_input_ids"], d_attention_mask=batch["d_attention_mask"]
                )
                outs = get_model_obj(self.model).compute_logits(fwd_out, batch)
                lg = outs.logits
                lb = outs.labels
                loss = self.criterion(lg, lb)
                loss = mean_tensor(loss).to(lg.device)
                bs = lg.size(0)
                total_loss += loss.item() * bs
                total_count += bs
                prd = torch.argsort(lg, dim=-1, descending=True)
                for b_idx in range(bs):
                    gold_indices = [b_idx]
                    p_ = prd[b_idx].tolist()
                    r1v, r2v, r5v, _ = calculate_metrics_train(p_, gold_indices)
                    rec1_sum += r1v
                    rec2_sum += r2v
                    rec5_sum += r5v
                    total_samples += 1
        r1a = rec1_sum / total_samples if total_samples else 0.0
        r2a = rec2_sum / total_samples if total_samples else 0.0
        r5a = rec5_sum / total_samples if total_samples else 0.0
        vl = total_loss / total_count if total_count else 0.0
        logger.info(f"Eval epoch {epoch}: R@1={r1a:.4f}, R@2={r2a:.4f}, R@5={r5a:.4f}, Val_Loss={vl:.4f}")
        self.valid_loss.append(round(vl, 3))
        print("Valid_loss = {}".format(self.valid_loss))
        return {'r@1': r1a, 'r2': r2a, 'r5': r5a}


