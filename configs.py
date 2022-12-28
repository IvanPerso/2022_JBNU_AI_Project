from typing import NamedTuple


class nsmcConfig(NamedTuple):
    train_dataset: str = 'ratings_train.txt'
    dev_dataset: str = 'ratings_test.txt'

    gpt_model_hub_name: str = "bert-base-multilingual-cased"
    
    max_sequence_length: int = 512
    # max_sequence_length: int = 1024

    epochs: int = 3
    # epochs: int = 20
    lr: float = 5e-5
    # train_batch_size: int = 16
    train_batch_size: int = 32
    test_batch_size: int = 32

    output_dir: str = "outputs/"

    grad_clip: float = 1.0
    warmup_ratio: float = 0.1

    # train_log_interval: int = 50
    # validation_interval: int = 400
    # save_interval: int = 400
    train_log_interval: int = 10
    validation_interval: int = 200
    save_interval: int = 200
    random_seed: int = 0
    tokenizer_name: str = ''
    model_name: str = ''