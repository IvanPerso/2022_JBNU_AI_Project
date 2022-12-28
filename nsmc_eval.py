import math
from argparse import ArgumentParser

import numpy as np
import torch
from torch.utils.data import DataLoader, RandomSampler
from tqdm import tqdm
from transformers import BertForSequenceClassification

from configs import nsmcConfig
from dataload.semti_data import dynamic_padding_collate_fn
from dataload.semti_data import semti_dataload

import logging
from utils import TqdmLoggingHandler
import os

parser = ArgumentParser()
parser.add_argument("-m", "--model-path", type=str, required=True)
parser.add_argument("-b", "--batch-size", type=int, default=50)
parser.add_argument('-tokenizer_name', "--tokenizer_name", type=str, default='')
parser.add_argument('-model_name', "--model_name", type=str, default='')
parser.add_argument('-output_dir', "--output-dir", type=str, default="google_bert/", help="모델과 학습 로그를 저장할 경로")

def main():
    config = nsmcConfig()
    args = parser.parse_args()
    
    logger = _create_logger(output_dir=args.output_dir)
    print(f"model_name: {args.model_name}")
    model = BertForSequenceClassification.from_pretrained(args.model_name, num_labels=2)
    model.load_state_dict(torch.load(args.model_path, map_location="cpu"))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)


    # tokenizer = SentencePieceBPETokenizer.from_file(
    #     vocab_filename="tokenizer/vocab.json", merges_filename="tokenizer/merges.txt", add_prefix_space=False
    # )
    
    # tokenizer = KoBertTokenizer.from_pretrained("monologg/kobert")
    if args.tokenizer_name == 'bert-base-multilingual-cased' or args.tokenizer_name == 'beomi/kcbert-base':
        from transformers import BertTokenizer 
        tokenizer = BertTokenizer.from_pretrained(args.tokenizer_name, do_lower_case=False)
    elif args.tokenizer_name == 'monologg/kobert':
        from kobert_tokenizer.KoBERTTokenizer import KoBertTokenizer
        tokenizer = KoBertTokenizer.from_pretrained(args.tokenizer_name)
        
        
    test_dataset = semti_dataload(tokenizer=tokenizer, is_train=False)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.test_batch_size, collate_fn=dynamic_padding_collate_fn
        )

    model = model.to(device)
    model.eval()

    model.eval()
    logits_len = 0
    acc_ = 0
    for batch_data in tqdm(test_dataloader, desc="[EVAL]"):
        with torch.no_grad():
            input_ids, labels, attention_mask = tuple(value.to(device) for value in batch_data)
            model_outputs =  model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)
            
            logits = model_outputs.logits # [50, 2]
            logits_max = torch.argmax(logits, dim=-1) # [50], [0, 1, 1 ... , 1, 0]
            logits_len += len(logits_max) 
            ture_false = torch.eq(logits_max, labels) # max, labels 같은지 확인 [50], [Ture, False, ... , True]
            acc_ += torch.sum(ture_false) 

    print(logits_len)
    logger.info(f"[EVAL] acc:{acc_ / logits_len * 100:.4f}")
    

def _create_logger(output_dir: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")

    file_handler = logging.FileHandler(os.path.join(output_dir, "valid.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    return logger

if __name__ == "__main__":
    main()
