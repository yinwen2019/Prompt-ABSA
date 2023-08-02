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

class BatchNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(BatchNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(1, keepdim=True)
        std = x.std(1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

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


class BertSelfAttentionModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.bertmodel = SPCModel(bert, opt)  # MLP
        #self.bertmodel = AttModel(bert, opt) # Selfattention
        self.classifier = nn.Linear(opt.bert_dim, opt.polarities_dim)

    def forward(self, inputs):
        outputs = self.bertmodel(inputs)
        outputs = self.classifier(outputs)
        logits = F.softmax(outputs, dim=-1)
        return logits


class SPCModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.bert = bert
        self.opt = opt
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)

    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask']
        bert_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        raw_output = bert_output.last_hidden_state
        outputs = self.bert_drop(raw_output)
        outputs = outputs[:, 0]
        return outputs


class AttModel(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.bert = bert
        self.opt = opt
        self.mem_dim = opt.bert_dim // 2
        self.attention_heads = opt.attention_heads
        self.bert_dim = opt.bert_dim
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.drop = nn.Dropout(0.1)
        self.batchnorm = BatchNorm(opt.bert_dim)
        self.layernorm = LayerNorm(opt.bert_dim)

        self.attdim = 100
        self.W = nn.Linear(self.attdim, self.attdim)
        self.V = nn.Linear(self.attdim, self.attdim)
        self.bert2attn = nn.Linear(self.bert_dim, self.attdim)

        self.attn = MultiHeadAttention(opt.attention_heads, self.attdim)

    def forward(self, inputs):
        input_ids, token_type_ids, attention_mask, aspect_mask = inputs['input_ids'], inputs['token_type_ids'], inputs['attention_mask'], inputs['aspect_mask']
        batch = input_ids.size(0)
        contextlen = input_ids.size(1)
        bert_output = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        raw_output = bert_output.last_hidden_state

        inputs = self.layernorm(raw_output)
        inputs = self.bert_drop(inputs)
        inputs = self.bert2attn(inputs)
        # 注意力
        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask = aspect_mask.unsqueeze(-1).repeat(1, 1, 100)
        aspect = (inputs*aspect_mask).sum(dim=1) / asp_wn
        attn_score = self.attn(inputs, inputs, aspect)

        # value_outputs = self.V(inputs)
        node_inputs = inputs.unsqueeze(1).expand(batch, self.attention_heads, contextlen, self.attdim)

        Ax = torch.matmul(attn_score, node_inputs)
        Ax = Ax.mean(dim=1)
        Ax = self.W(Ax)
        attention_outputs = Ax[:, 0]
        outputs = F.relu(attention_outputs)
        return outputs


def attention(query, key,  aspect, bias, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    aspect_scores = torch.tanh(torch.add(torch.matmul(aspect, key.transpose(-2, -1)), bias))

    scores = torch.add(scores, aspect_scores)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn

 
def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()  
        self.d_k = d_model // h  
        self.h = h    
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)
        self.weight_q = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k))
        self.weight_k = nn.Parameter(torch.Tensor(self.h, self.d_k, self.d_k))
        self.bias = nn.Parameter(torch.Tensor(1))
        self.dense = nn.Linear(d_model, self.d_k)

    def forward(self, query, key, aspect):
        nbatches = query.size(0)  
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key))]
        
        batch, aspect_dim = aspect.size()[0], aspect.size()[1]
        aspect = aspect.unsqueeze(1).expand(batch, self.h, aspect_dim)    
        aspect = self.dense(aspect) 
        aspect = aspect.unsqueeze(2).expand(batch, self.h, query.size()[2], self.d_k)
        attn = attention(query, key, aspect, self.bias, dropout=self.dropout)
        return attn