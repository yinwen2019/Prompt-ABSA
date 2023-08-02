'''
Description:  
Author: XXX
Date: 2023-02-01 14:17:37
'''
import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer

class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SoftPromptModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.model = SoftModel(bert, opt)
        self.dropout = nn.Dropout(opt.bert_dropout)
        self.classifier = nn.Linear(30523, opt.polarities_dim)

    def forward(self, inputs):
        outputs = self.model(inputs)
        outputs = self.classifier(outputs)
        logits = F.softmax(outputs, dim=-1)
        return logits


class SoftModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.bert = bert
        self.opt = opt

        self.attdim = 100
        self.bert_dim = opt.bert_dim
        self.vocab_size = 30523
        self.bert_drop = nn.Dropout(opt.bert_dropout)

        self.embeddings = bert.bert.get_input_embeddings()
        self.layernorm = LayerNorm(self.vocab_size)
        self.prompt_encoder = PromptEncoder(opt).to(opt.device)

        self.dense = nn.Linear(self.vocab_size, self.attdim)

    def get_embeds_input(self, input_ids, prompt_mask):
        bz = input_ids.size(0)
        embeds_inputs = input_ids.clone()
        # prompt_mask # [16, 100]
        raw_embeddings = self.embeddings(embeds_inputs)  # [16, 100, 768]
        prompt_embeddings = self.prompt_encoder()  # [3, 768]
        for bidx in range(bz):
            pidx = 0
            for sidx in range(prompt_mask.size(-1)):
                if prompt_mask[bidx, sidx]:
                    raw_embeddings[bidx, sidx, :] = prompt_embeddings[pidx, :]
                    pidx += 1
        return raw_embeddings

    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask, prompt_mask, predict_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'], inputs['prompt_mask'], inputs['predict_mask']
        batch = input_ids.size(0)
        inputs_embeddings = self.get_embeds_input(input_ids, prompt_mask)
        outputs = self.bert(inputs_embeds=inputs_embeddings, attention_mask=attention_mask, token_type_ids=token_type_ids)
        rawinputs = outputs.logits
        inputs = self.bert_drop(rawinputs)

        # get mask token representation
        selection_mask = predict_mask.unsqueeze(-1)
        outputs = torch.masked_select(inputs, selection_mask)
        outputs = outputs.view(batch, -1)
        outputs = F.relu(outputs)
        return outputs


class PromptEncoder(nn.Module):
    def __init__(self,  opt):
        super().__init__()
        self.hidden_size = opt.bert_dim
        # ent embedding
        self.prompt_indices = torch.LongTensor(list(range(opt.softprompt))).to(opt.device)
        # embedding
        self.embedding = torch.nn.Embedding(opt.softprompt, self.hidden_size).to(opt.device)
        # LSTM
        self.lstm_head = torch.nn.LSTM(input_size=self.hidden_size,
                                       hidden_size=self.hidden_size // 2,
                                       num_layers=2,
                                       dropout=0.4,
                                       bidirectional=True,
                                       batch_first=True)
        self.mlp_head = nn.Sequential(nn.Linear(self.hidden_size, self.hidden_size),
                                      nn.ReLU(),
                                      nn.Linear(self.hidden_size, self.hidden_size))
        self.drop = nn.Dropout(0.3)

    def forward(self):
        input_embeds = self.embedding(self.prompt_indices).unsqueeze(0)
        output_embeds = self.mlp_head(self.lstm_head(input_embeds)[0]).squeeze()
        output_embeds = self.drop(output_embeds)
        return output_embeds