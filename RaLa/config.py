import os
import random
import torch
import argparse
import warnings
import torch.backends.cudnn as cudnn

parser = argparse.ArgumentParser(description='MHQA training arguments')
parser.add_argument('--pretrained-model', default='', type=str)
parser.add_argument('--reranker-model', default='jinaai/jina-reranker-v2-base-multilingual', type=str)
parser.add_argument('--train-path', default='', type=str)
parser.add_argument('--valid-path', default='', type=str)
parser.add_argument('--model-dir', default='', type=str)
parser.add_argument('--warmup', default=400, type=int)
parser.add_argument('--max-to-keep', default=5, type=int)
parser.add_argument('--grad-clip', default=10.0, type=float)
parser.add_argument('--t', default=0.05, type=float)
parser.add_argument('--use-amp', default=True, action='store_true')
parser.add_argument('--use-posFactor', default=False, action='store_true')
parser.add_argument('-j', '--workers', default=1, type=int)
parser.add_argument('--epochs', default=10, type=int)
parser.add_argument('-b', '--batch-size', default=6, type=int)
parser.add_argument('--lr', '--learning-rate', default=2e-5, type=float, dest='lr')
parser.add_argument('--lr-scheduler', default='linear', type=str)
parser.add_argument('--weight-decay', '--wd', default=1e-5, type=float, dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=50, type=int)
parser.add_argument('--seed', default=None, type=int)
parser.add_argument('--is-test', action='store_true')
parser.add_argument('--eval-model-path', default='', type=str)
parser.add_argument('--finetune-t', action='store_true')
parser.add_argument('--max-seq-length', default=512, type=int)
parser.add_argument('--checkpoint-path', default='', type=str)
parser.add_argument('--test-path', default='', type=str)
parser.add_argument('--out-path', default='', type=str)
parser.add_argument('--results-path', default='', type=str)
parser.add_argument('--task', default='hotpotqa', type=str)
parser.add_argument('--layers', default=[0, 4, 7, 11], type=list)

args = parser.parse_args()
if args.model_dir:
    os.makedirs(args.model_dir, exist_ok=True)

if args.seed is not None:
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    cudnn.deterministic = True

try:
    if args.use_amp:
        import torch.cuda.amp
except Exception:
    args.use_amp = False
    warnings.warn('AMP training is not available, set use_amp=False')

if not torch.cuda.is_available():
    args.use_amp = False
    args.print_freq = 1
    warnings.warn('GPU is not available, set use_amp=False and print_freq=1')

