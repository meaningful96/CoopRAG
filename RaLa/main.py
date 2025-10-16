import torch
import json
import torch.backends.cudnn as cudnn
import os
import warnings

from config import args
from trainer import Trainer
from logger_config import logger

warnings.filterwarnings("ignore", category=FutureWarning)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    # Remove any reference to local_rank or torch.distributed
    # We'll just detect how many GPUs are available and use DataParallel if > 1
    ngpus_per_node = torch.cuda.device_count()
    cudnn.benchmark = True
    logger.info(f"Use {ngpus_per_node} GPUs for training.")

    # Create trainer (DP version)
    trainer = Trainer(args)

    logger.info(f"Args={json.dumps(args.__dict__, ensure_ascii=False, indent=4)}")
    trainer.train_loop()

if __name__ == '__main__':
    main()

