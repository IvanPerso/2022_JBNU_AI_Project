from dataload.semti_data import semti_dataload
from torch.utils.data import DataLoader
import torch
from argparse import ArgumentParser
from datetime import datetime
import os
import logging
from transformers import BertForSequenceClassification
from torch.utils.data import DistributedSampler
from dataload.semti_data import dynamic_padding_collate_fn
from torch.optim import Adam
from tqdm import tqdm
import numpy as np
from utils import TqdmLoggingHandler
from configs import nsmcConfig
from transformers import get_linear_schedule_with_warmup
import math

parser = ArgumentParser()
parser.add_argument("--train-dataset", type=str, help="학습 데이터 경로")
parser.add_argument("--dev-dataset", type=str, help="평가 데이터 경로")

parser.add_argument("--epochs", type=int, help="학습 전체를 반복할 횟수")
parser.add_argument("--lr", type=float, help="learning rate")

parser.add_argument("--train-batch-size", type=int, help="학습에 사용할 배치 크기")
parser.add_argument("--test-batch-size", type=int, help="평가에 사용할 배치 크기")
parser.add_argument("--validation-interval", type=int, help="dev 셋에 대해서 validation 을 수행할 steps")
parser.add_argument("--save-interval", type=int, help="모델을 저장할 steps")

parser.add_argument("--output-dir", type=str, default="google_bert/", help="모델과 학습 로그를 저장할 경로")
parser.add_argument("--tokenizer_name", type=str, default='')
parser.add_argument("--model_name", type=str, default='')

def main_(gpu, ngpus_per_node):
    kwargs = {key: value for key, value in vars(parser.parse_args()).items() if value is not None}
    
    timestamp = datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    artifacts_dir = os.path.join(kwargs["output_dir"], f"bert_{timestamp}")
    os.makedirs(artifacts_dir, exist_ok=True)
    kwargs["output_dir"] = artifacts_dir
    
    config=nsmcConfig(**kwargs)
        
    logger = _create_logger(output_dir=config.output_dir)
    logger.info("============================")
    for key, value in config._asdict().items():
        logger.info(f"{key:30}:{value}")
    logger.info("============================")
    torch.manual_seed(config.random_seed)
    
    torch.distributed.init_process_group(
            backend='nccl',
            init_method='tcp://127.0.0.1:3456',
            world_size=ngpus_per_node,
            rank=gpu)
    
    torch.cuda.set_device(gpu)
    
    # tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased', do_lower_case=False)
    
    if config.tokenizer_name == 'bert-base-multilingual-cased' or config.tokenizer_name == 'beomi/kcbert-base':
        from transformers import BertTokenizer 
        tokenizer = BertTokenizer.from_pretrained(config.tokenizer_name, do_lower_case=False)
    elif config.tokenizer_name == 'monologg/kobert':
        from kobert_tokenizer.KoBERTTokenizer import KoBertTokenizer
        tokenizer = KoBertTokenizer.from_pretrained(config.tokenizer_name)

        
    logger.info("loading train dataset")
    train_dataset = semti_dataload(tokenizer=tokenizer, is_train=True)
    train_sampler = DistributedSampler(train_dataset, rank=gpu, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset, batch_size=config.train_batch_size, collate_fn=dynamic_padding_collate_fn, sampler=train_sampler
        )
    
    test_dataset = semti_dataload(tokenizer=tokenizer, is_train=False)
    test_sampler = DistributedSampler(test_dataset, rank=gpu, shuffle=True)
    test_dataloader = DataLoader(
        test_dataset, batch_size=config.test_batch_size, collate_fn=dynamic_padding_collate_fn, sampler=test_sampler
        )
    
    # model = BertForSequenceClassification.from_pretrained("bert-base-multilingual-cased")
    model = BertForSequenceClassification.from_pretrained(config.model_name)
    
    device = gpu
    model.cuda(gpu)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    
    optimizer = Adam(model.parameters(), lr=config.lr)
    total_steps = len(train_dataloader) * config.epochs
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    
    loss_list_between_log_interval = []
    for epoch_id in range(config.epochs):
        train_sampler.set_epoch(config.epochs)

        for step_index, batch_data in tqdm(
            enumerate(train_dataloader), f"[TRAIN] EP:{epoch_id}", total=len(train_dataloader)
        ):
            global_step = len(train_dataloader) * epoch_id + step_index + 1
            optimizer.zero_grad()

            input_ids,  labels, attention_mask = tuple(value.to(device) for value in batch_data)
            # print(input_ids)
            # print(labels)
            # print(input_ids.shape)
            # print(labels.shape)
            
            model_outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

            model_outputs.loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            scheduler.step()

            # for logging
            loss_list_between_log_interval.append(model_outputs.loss.item())

            if gpu==0:
                if global_step % config.train_log_interval == 0:
                    mean_loss = np.mean(loss_list_between_log_interval)
                    logger.info(
                        f"EP:{epoch_id} global_step:{global_step} "
                        f"loss:{mean_loss:.4f} perplexity:{math.exp(mean_loss):.4f}"
                    )
                    loss_list_between_log_interval.clear()
            
                if global_step % config.validation_interval == 0:
                    _validate(model, test_dataloader, device, logger, global_step)
                
                if global_step % config.save_interval == 0:
                    # state_dict = model.state_dict()
                    state_dict = model.module.state_dict()
                    model_path = os.path.join(config.output_dir, f"gpt2_step_{global_step}.pth")
                    logger.info(f"global_step: {global_step} model saved at {model_path}")
                    torch.save(state_dict, model_path)


def _validate(
    model,
    test_dataloader: DataLoader,
    device: torch.device,
    logger: logging.Logger,
    global_step: int,
):
    model.eval()
    loss_list = []
    eval_accuracy = 0
    nb_eval_steps = 0
    loss_list = []
    for batch_data in tqdm(test_dataloader, desc="[EVAL]"):
        with torch.no_grad():
            input_ids, labels, attention_mask = tuple(value.to(device) for value in batch_data)
            model_outputs = model(input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

            logits = model_outputs.logits
            loss_list.append(model_outputs.loss.item())
            
            tmp_eval_accuracy = accuracy_measure(logits.to('cpu').numpy(), labels.to('cpu').numpy())
            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1
    mean_loss = np.mean(loss_list)
    logger.info(f"[EVAL] global_step:{global_step} loss:{mean_loss:.4f} perplexity:{math.exp(mean_loss):.4f}")
    print("  Accuracy: {0:.2f}".format(eval_accuracy/nb_eval_steps))


def accuracy_measure(y_pred, y):
    pred_flattened = np.argmax(y_pred, axis=1).flatten()
    y_flattened = y.flatten()
    return np.sum(pred_flattened == y_flattened) / len(y_flattened)





def _create_logger(output_dir: str):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(asctime)s] %(message)s")

    file_handler = logging.FileHandler(os.path.join(output_dir, "train.log"))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    handler = TqdmLoggingHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s - %(message)s"))
    logger.addHandler(handler)
    return logger
    
if __name__=="__main__":
    device_size = torch.cuda.device_count()
    print(torch.cuda.is_available())
    print(device_size)
    torch.multiprocessing.spawn(main_, nprocs=device_size, args=(device_size, ))