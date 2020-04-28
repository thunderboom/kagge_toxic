# coding: UTF-8
import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaTokenizer, XLMRobertaConfig
from transformers import AutoTokenizer, AutoModelWithLMHead

class Config(object):

    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'xlm-roberta-base'
        self.train_path_1 = dataset + '/data/jigsaw-toxic-comment-train.csv'                                # 训练集1
        self.train_path_2 = dataset + '/data/jigsaw-unintended-bias-train.csv'                               # 训练集2
        self.dev_path = dataset + '/data/validation.csv'                                    # 验证集
        self.test_path = dataset + '/data/test.csv'                                  # 测试集
        # self.class_list = [x.strip() for x in open(
        #     dataset + '/data/class.txt').readlines()]                                # 类别名单
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.bin'        # 模型训练结果
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = 2                                            # 类别数
        self.num_epochs = 3                                             # epoch数
        self.warmup_proportion = 0.1                                    # warmup占比
        self.batch_size = 32                                             # mini-batch大小
        self.pad_size = 512                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 2e-5                                       # 学习率
        self.pretrained_path = './pretrained_model/xlm-roberta-base/'
        self.tokenizer = XLMRobertaTokenizer.from_pretrained(self.pretrained_path)
        self.hidden_size = 768


class Model(nn.Module):

    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = XLMRobertaModel.from_pretrained(config.pretrained_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask)
        out = self.fc(pooled)
        out = self.sigmoid(out)
        return out
