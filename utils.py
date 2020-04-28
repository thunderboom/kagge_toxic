# coding: UTF-8
import torch
from tqdm import tqdm
import time
from datetime import timedelta
import pandas as pd
import numpy as np
PAD, CLS = '[PAD]', '[CLS]'  # padding符号, bert中综合信息符号


def build_dataset(config):
    def load_dataset(path, data_type='train'):
        load_data = pd.read_csv(path)
        if data_type == 'test':
            load_data['toxic'] = [0] * len(load_data)
            load_data.rename(columns={'content': 'comment_text'}, inplace=True)
            return load_data[['comment_text', 'toxic', 'lang']]
        elif data_type == 'train':
            return load_data[['comment_text', 'toxic']]
        elif data_type == 'dev':
            return load_data[['comment_text', 'toxic', 'lang']]
        else:
            raise ValueError("data type must choose from ['train', 'dev', 'test']")

    def text_processing(text):
        '''clean the text or translate processing'''
        return text

    def convert_text_features(data_df, pad_size=512):
        contents = []
        for _, line in tqdm(data_df.iterrows()):
            content, label = line['comment_text'], line['toxic']
            content = text_processing(content)
            token = config.tokenizer.tokenize(content)
            token = [CLS] + token
            seq_len = len(token)
            mask = []
            token_ids = config.tokenizer.convert_tokens_to_ids(token)

            if pad_size:
                if len(token) < pad_size:
                    mask = [1] * len(token_ids) + [0] * (pad_size - len(token))
                    token_ids += ([0] * (pad_size - len(token)))
                else:
                    mask = [1] * pad_size
                    token_ids = token_ids[:pad_size]
                    # #use the head+tail
                    # head = round(pad_size/ 2)
                    # tail = -1 * (pad_size - head)
                    # token_ids = token_ids[:head] + token_ids[tail:]
            contents.append((token_ids, int(label), seq_len, mask))
        return contents

    train_part1 = load_dataset(config.train_path_1, 'train')
    train_part2 = load_dataset(config.train_path_2, 'train')
    train_part2.toxic = train_part2.toxic.round().astype(int)  #由于train_path的label为小数
    train = pd.concat([
                    train_part1[['comment_text', 'toxic']],
                    train_part2[['comment_text', 'toxic']].query('toxic==1'),
                    train_part2[['comment_text', 'toxic']].query('toxic==0').sample(n=100000, random_state=128)  #balance the data
                    ])
    train = train.sample(frac=1).reset_index(drop=True)[:10000]  #shuffle train data
    dev = load_dataset(config.dev_path, 'dev')
    test = load_dataset(config.test_path, 'test')[:10000]
    train = convert_text_features(train)
    dev = convert_text_features(dev)
    test = convert_text_features(test)

    return train, dev, test


class DatasetIterater(object):
    def __init__(self, batches, batch_size, device):
        self.batch_size = batch_size
        self.batches = batches
        self.n_batches = len(batches) // batch_size
        self.residue = False  # 记录batch数量是否为整数
        if len(batches) % self.n_batches != 0:
            self.residue = True
        self.index = 0
        self.device = device

    def _to_tensor(self, datas):
        x = torch.LongTensor([_[0] for _ in datas]).to(self.device)
        y = torch.LongTensor([_[1] for _ in datas]).to(self.device)

        # pad前的长度(超过pad_size的设为pad_size)
        seq_len = torch.LongTensor([_[2] for _ in datas]).to(self.device)
        mask = torch.LongTensor([_[3] for _ in datas]).to(self.device)
        return (x, seq_len, mask), y

    def __next__(self):
        if self.residue and self.index == self.n_batches:
            batches = self.batches[self.index * self.batch_size: len(self.batches)]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

        elif self.index >= self.n_batches:
            self.index = 0
            raise StopIteration
        else:
            batches = self.batches[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            batches = self._to_tensor(batches)
            return batches

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.n_batches + 1
        else:
            return self.n_batches


def build_iterator(dataset, config):
    iter = DatasetIterater(dataset, config.batch_size, config.device)
    return iter


def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))
