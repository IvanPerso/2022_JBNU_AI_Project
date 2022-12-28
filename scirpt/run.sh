# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --output-dir 'google_bert' --tokenizer_name 'bert-base-multilingual-cased' --model_name 'bert-base-multilingual-cased'
# CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --output-dir 'skt_bert' --tokenizer_name 'monologg/kobert' --model_name 'monologg/kobert'
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py --output-dir 'kc_bert' --tokenizer_name 'beomi/kcbert-base' --model_name 'beomi/kcbert-base'

