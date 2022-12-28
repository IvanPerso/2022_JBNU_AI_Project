CUDA_VISIBLE_DEVICES=0 python nsmc_eval.py --model-path 'google_bert/bert_2022.12.27_22.52.03/gpt2_step_1200.pth' --output-dir 'google_bert' --tokenizer_name 'bert-base-multilingual-cased' --model_name 'bert-base-multilingual-cased'
CUDA_VISIBLE_DEVICES=0 python nsmc_eval.py --model-path 'skt_bert/bert_2022.12.28_21.46.43/gpt2_step_1200.pth' --output-dir 'skt_bert' --tokenizer_name 'monologg/kobert' --model_name 'monologg/kobert'
CUDA_VISIBLE_DEVICES=0 python nsmc_eval.py --model-path 'kc_bert/bert_2022.12.28_22.19.17/gpt2_step_1200.pth' --output-dir 'kc_bert' --tokenizer_name 'beomi/kcbert-base' --model_name 'beomi/kcbert-base'

