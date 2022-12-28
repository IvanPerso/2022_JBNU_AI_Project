# 필요한 라이브러리 불러오기
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm
import torch

class semti_dataload(Dataset):
    def __init__(self, tokenizer, is_train=True):
        self.tokenizer = tokenizer
        self.is_train = is_train
        self.prefix_data()
        
    def __getitem__(self, index):
        return self.input_ids[index], self.labels[index], self.attention_mask[index]

    def __len__(self):
        # return self.dataset_len
        return len(self.pandas_datas)
    
    def nsmc_pandas_dataload(self):
        if self.is_train:
            self.pandas_datas = pd.read_table('ratings_train.txt')
        else:
            self.pandas_datas = pd.read_table('ratings_test.txt')

    def prefix_data(self):
        self.nsmc_pandas_dataload()
        
        self.input_ids = []
        # self.labels = [[1, 0] if labels == 0 else [0, 1] for labels in self.pandas_datas['label'] ]
        self.labels = [labels for labels in self.pandas_datas['label']]
        self.attention_mask = []
        
        # self.labels = []
        # for labels in self.pandas.datas['labels']:
        #     self.labels.append(labels)
            
        for document in tqdm(self.pandas_datas['document']):
            input_ids = self.tokenizer.encode(str(document))
            self.input_ids.append(input_ids)
            self.attention_mask.append([1] * len(input_ids))


def dynamic_padding_collate_fn(features):
    max_seq_len = max([len(feature[0]) for feature in features])
    input_ids, attention_mask, labels = [], [], []

    for feature in features:
        padded_input_ids = feature[0] + [0] * (max_seq_len - len(feature[0]))
        padded_attention_mask = feature[2] + [0.0] * (max_seq_len - len(feature[2]))


        input_ids.append(padded_input_ids)
        attention_mask.append(padded_attention_mask)
        labels.append(feature[1])

    return torch.tensor(input_ids), torch.tensor(labels), torch.tensor(attention_mask)

    # def data_tokenizer(self):
    #     self.text_tag = []
    #     self.test_tag = []
        
    #     for document in self.train_data['document']:
    #         doc = "[CLS]" + str(document) + "[SEP]"
    #         self.text_tag.append(doc)
            
    #     for document in self.test_data['document']:
    #         doc = "[CLS]" + str(document) + "[SEP]"
    #         self.test_tag.append(doc)
            
    # def pretrained_name_cased(self):
    #     # name 파라미터 넣어서 이름 변경 가능?
        
    #     self.tokenized_data = []
    #     self.tokenized_test_data = []
        
    #     for document in self.text_tag:
    #             tokens = self.tokenizer.tokenize(document)
    #             self.tokenized_data.append(tokens)
                
    #     for document in self.test_tag:
    #             tokens = self.tokenizer.tokenize(document)
    #             self.tokenized_test_data.append(tokens)
        
    # #padding
    # def padding(self, max_len):
    #     self.train_input_ids = []
    #     self.test_input_ids = []
        
    #     for token in self.tokenized_data:
    #         ids = self.tokenizer.convert_tokens_to_ids(token)
    #         ids_len = len(ids)
    #         if ids_len > max_len:
    #             avlid_tokens = ids_len - max_len
    #             ids = ids[:avlid_tokens]
    #         else:    
    #             padd_len = max_len - len(ids)
    #             ids = ids + [0] * padd_len
                
    #         self.train_input_ids.append(ids)
                
    #     for token in self.tokenized_test_data:
    #         ids = self.tokenizer.convert_tokens_to_ids(token)
    #         ids_len = len(ids)
    #         if ids_len > max_len:
    #             avlid_tokens = ids_len - max_len
    #             ids = ids[:avlid_tokens]
    #         else:    
    #             padd_len = max_len - len(ids)
    #             ids = ids + [0] * padd_len
            
    #         self.test_input_ids.append(ids)
    # def masking(self):
    #     self.attention_masks_train = []
    #     self.attention_masks_test = []
        
    #     for ids in self.train_input_ids:
    #         ids_mask = []
    #         for idx in ids:
    #             masked = float(idx>0)
    #             ids_mask.append(masked)
    #         self.attention_masks_train.append(ids_mask)
            
    #     for ids in self.test_input_ids:
    #         ids_mask = []
    #         for idx in ids:
    #             masked = float(idx>0)
    #             ids_mask.append(masked)
    #         self.attention_masks_test.append(ids_mask)
 