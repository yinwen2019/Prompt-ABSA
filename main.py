import os
import sys

import data_utils


import copy
import random
import logging
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import BertModel, AdamW, RobertaModel,  BertForMaskedLM, RobertaForMaskedLM

from models.BertSelfAttention import BertSelfAttentionModel
from models.HardPrompt import ManuralHardPromptModel
from models.SoftPrompt import SoftPromptModel
from models.Roberta import ManuralHardPromptModel4RoBerta
from data_utils import Tokenizer4Bert, Tokenizer4RoBerta
from Dataset import ManualTriggersData, ManualHardRoBertaData, BertSPCData, ManualHardData, SoftPromptData

logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler(sys.stdout))


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


class Instructor:
    ''' Model training and evaluation '''

    def __init__(self, opt):
        self.opt = opt
        trainset = None
        testset = None
        if opt.model_name == 'bertselfattention':
            tokenizer = Tokenizer4Bert(opt.max_length, opt.pretrained_bert_name)
            bert = BertModel.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bert, opt).to(opt.device)
            trainset = BertSPCData(opt.dataset_file['train'], tokenizer, opt=opt, train_pre=opt.train_pre)
            testset = BertSPCData(opt.dataset_file['test'], tokenizer, opt=opt)
        elif opt.model_name == 'manuralhardprompt':
            tokenizer = Tokenizer4Bert(opt.max_length, opt.pretrained_bert_name)
            bertformask = BertForMaskedLM.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class(bertformask, opt).to(opt.device)
            if opt.triggers:
                trainset = ManualTriggersData(opt.dataset_file['train'], tokenizer, opt=opt, train_pre=opt.train_pre)
                testset = ManualTriggersData(opt.dataset_file['test'], tokenizer, opt=opt)
            else:
                trainset = ManualHardData(opt.dataset_file['train'], tokenizer, opt=opt, train_pre=opt.train_pre)
                testset = ManualHardData(opt.dataset_file['test'], tokenizer, opt=opt)
        elif opt.model_name == 'softprompt':
            tokenizer = Tokenizer4Bert(opt.max_length, opt.pretrained_bert_name)
            bert = BertForMaskedLM.from_pretrained(opt.pretrained_bert_name)
            bert.resize_token_embeddings(tokenizer.vocabsize())
            self.model = opt.model_class(bert, opt).to(opt.device)
            trainset = SoftPromptData(opt.dataset_file['train'], tokenizer, opt=opt, train_pre=opt.train_pre)
            testset = SoftPromptData(opt.dataset_file['test'], tokenizer, opt=opt)
        elif opt.model_name == 'roberta':
            tokenizer = Tokenizer4RoBerta(opt.max_length, opt.pretrained_bert_name)
            roberta = RobertaModel.from_pretrained(opt.pretrained_bert_name)
            robertamask = RobertaForMaskedLM.from_pretrained(opt.pretrained_bert_name)
            self.model = opt.model_class((roberta, robertamask), opt).to(opt.device)
            trainset = ManualHardRoBertaData(opt.dataset_file['train'], tokenizer, opt=opt)
            testset = ManualHardRoBertaData(opt.dataset_file['test'], tokenizer, opt=opt)

        self.train_dataloader = DataLoader(dataset=trainset, batch_size=opt.batch_size, shuffle=True)
        self.test_dataloader = DataLoader(dataset=testset, batch_size=opt.batch_size)

        if opt.device.type == 'cuda':
            logger.info('cuda memory allocated: {}'.format(torch.cuda.memory_allocated(self.opt.device.index)))
        self._print_args()

    def _print_args(self):
        n_trainable_params, n_nontrainable_params = 0, 0
        for p in self.model.parameters():
            n_params = torch.prod(torch.tensor(p.shape))
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params

        logger.info('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        logger.info('training arguments:')

        for arg in vars(self.opt):
            logger.info('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))
        # for name, para in self.model.named_parameters():
        #     print(name)
        #     print(para.requires_grad)

    def _cloze_params(self):
        if self.opt.model_name == 'PTNet_bert':
            for name, para in self.model.named_parameters():
                if 'pro_model.bert' in name:
                    para.requires_grad = False

    def get_bert_optimizer(self, model):
        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        logger.info("bert learning rate on")
        _params = filter(lambda n, p: p.requires_grad, model.named_parameters())
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.opt.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.opt.bert_lr, eps=self.opt.adam_epsilon)

        return optimizer

    def _train(self, criterion, optimizer, max_test_acc_overall=0):
        max_test_acc = 0
        max_f1 = 0
        global_step = 0
        model_path = ''
        for epoch in range(self.opt.num_epoch):
            logger.info('>' * 60)
            logger.info('epoch: {}'.format(epoch))
            n_correct, n_total = 0, 0
            for i_batch, sample_batched in enumerate(self.train_dataloader):
                global_step += 1
                # switch model to training mode, clear gradient accumulators
                self.model.train()
                optimizer.zero_grad()
                inputs = {k: v.to(self.opt.device) for k, v in sample_batched.items()}
                outputs = self.model(inputs)
                targets = sample_batched['polarity'].to(self.opt.device)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
                if global_step % self.opt.log_step == 0:
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    n_total += len(outputs)
                    train_acc = n_correct / n_total
                    test_acc, f1 = self._evaluate()
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                        if test_acc > max_test_acc_overall:
                            if not os.path.exists('./state_dict'):
                                os.mkdir('./state_dict')
                            model_path = './state_dict/{}_{}_acc_{:.4f}_f1_{:.4f}'.format(self.opt.model_name, self.opt.dataset,
                                                                                          test_acc, f1)
                            self.best_model = copy.deepcopy(self.model)
                            logger.info('>> saved: {}'.format(model_path))
                    if f1 > max_f1:
                        max_f1 = f1
                    logger.info(
                        'loss: {:.4f}, train_acc: {:.4f}, test_acc: {:.4f}, f1: {:.4f}'.format(loss.item(), train_acc, test_acc,
                                                                                               f1))
        return max_test_acc, max_f1, model_path

    def _evaluate(self, show_results=False):
        # switch model to evaluation mode
        self.model.eval()
        n_test_correct, n_test_total = 0, 0
        targets_all, outputs_all = None, None
        with torch.no_grad():
            for batch, sample_batched in enumerate(self.test_dataloader):
                inputs = {k: v.to(self.opt.device) for k, v in sample_batched.items()}
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs = self.model(inputs)
                n_test_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                n_test_total += len(outputs)
                targets_all = torch.cat((targets_all, targets), dim=0) if targets_all is not None else targets
                outputs_all = torch.cat((outputs_all, outputs), dim=0) if outputs_all is not None else outputs
        test_acc = n_test_correct / n_test_total
        f1 = metrics.f1_score(targets_all.cpu(), torch.argmax(outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')

        labels = targets_all.data.cpu()
        predict = torch.argmax(outputs_all, -1).cpu()
        if show_results:
            report = metrics.classification_report(labels, predict, digits=4)
            confusion = metrics.confusion_matrix(labels, predict)
            return report, confusion, test_acc, f1

        return test_acc, f1

    def _test(self):
        self.model = self.best_model
        self.model.eval()
        test_report, test_confusion, acc, f1 = self._evaluate(show_results=True)
        logger.info("Precision, Recall and F1-Score...")
        logger.info(test_report)
        logger.info("Confusion Matrix...")
        logger.info(test_confusion)

    def run(self):
        criterion = nn.CrossEntropyLoss()
        # if self.opt.model_name == 'manuralhardprompt':
        #     self._cloze_params()

        # init optimizer
        optimizer = self.get_bert_optimizer(self.model)
        my_lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=0.98)
        max_test_acc_overall = 0
        max_f1_overall = 0
        max_test_acc, max_f1, model_path = self._train(criterion, optimizer, max_test_acc_overall)
        logger.info('max_test_acc: {0}, max_f1: {1}'.format(max_test_acc, max_f1))
        max_test_acc_overall = max(max_test_acc, max_test_acc_overall)
        max_f1_overall = max(max_f1, max_f1_overall)
        # torch.save(self.best_model.state_dict(), model_path)
        logger.info('>> saved: {}'.format(model_path))
        logger.info('#' * 60)
        logger.info('max_test_acc_overall:{}'.format(max_test_acc_overall))
        logger.info('max_f1_overall:{}'.format(max_f1_overall))
        self._test()


def main():
    model_classes = {
        'manuralhardprompt': ManuralHardPromptModel,
        'bertselfattention': BertSelfAttentionModel,
        'softprompt': SoftPromptModel,
        'roberta': ManuralHardPromptModel4RoBerta
    }

    dataset_files = {
        'restaurant': {
            'train': './dataset/Restaurants/train.json',
            'test': './dataset/Restaurants/test.json',
        },
        'laptop': {
            'train': './dataset/Laptops/train.json',
            'test': './dataset/Laptops/test.json'
        },
        'twitter': {
            'train': './dataset/Tweets/train.json',
            'test': './dataset/Tweets/test.json',
        }
    }


    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }

    optimizers = {
        'adadelta': torch.optim.Adadelta,
        'adagrad': torch.optim.Adagrad,
        'adam': torch.optim.Adam,
        'adamax': torch.optim.Adamax,
        'asgd': torch.optim.ASGD,
        'rmsprop': torch.optim.RMSprop,
        'sgd': torch.optim.SGD,
    }

    pretrain = {
        'bert-base-uncased': '/newdisk/PLM/bert-base-uncased',
        # 'bert-base-uncased': '/home/hxm/PLM/bert-base-uncased',
        # 'bert-base-uncased': '/home/Like/flc/wxl/PreModel/bert-base-uncased',
        'bert-large-uncased': '/home/Like/flc/wxl/PreModel/bert-large-uncased',
        'roberta-base': '/home/Like/flc/wxl/PreModel/roberta-base',
        'deberta-v3-base': 'E:/PreModel/deberta-v3-base',
        'deberta-base': 'E:/PreModel/deberta-base'
    }

    # Hyperparameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='hardprompt', type=str, help='hardprompt, softprompt'.join(model_classes.keys()))
    parser.add_argument('--dataset', default='twitter', type=str, help=', '.join(dataset_files.keys()))
    parser.add_argument('--optimizer', default='adam', type=str, help=', '.join(optimizers.keys()))
    parser.add_argument('--initializer', default='xavier_uniform_', type=str, help=', '.join(initializers.keys()))
    parser.add_argument('--num_epoch', default=20, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--log_step', default=10, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int, help='3')
    parser.add_argument('--max_length', default=100, type=int, help='input token max length')
    parser.add_argument('--train_pre', default=None, type=float, help='proportion of the training data size')

    # * prompt
    parser.add_argument('--attention_heads', default=5, type=int, help='number of multi-attention heads')
    parser.add_argument('--device', default=None, type=str, help='cpu, cuda, cuda: (num)')
    parser.add_argument('--seed', default=114, type=int)
    parser.add_argument("--weight_decay", default=0.0005, type=float, help="Weight decay if we apply some.")
    parser.add_argument('--triggers', default=False, type=str, help="Whether to manually set triggers")
    parser.add_argument('--softprompt', default=3, type=int, help="Number of the soft prompts")

    # * bert
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--bert_dim', type=int, default=768)
    parser.add_argument('--bert_dropout', type=float, default=0.3, help='BERT dropout rate.')
    parser.add_argument('--bert_lr', default=2e-5, type=float)
    opt = parser.parse_args()

    opt.model_class = model_classes[opt.model_name]
    opt.dataset_file = dataset_files[opt.dataset]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.pretrained_bert_name = pretrain[opt.pretrained_bert_name]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if opt.device is None else torch.device(
        opt.device)

    # set random seed
    setup_seed(opt.seed)

    # if not os.path.exists('./log'):
    #     os.makedirs('./log', mode=0o777)
    # log_file = '{}-{}-{}.log'.format(opt.model_name, opt.dataset, strftime("%Y-%m-%d_%H-%M-%S", localtime()))
    # logger.addHandler(logging.FileHandler(os.path.join('log', log_file)))

    ins = Instructor(opt)
    ins.run()


if __name__ == '__main__':
    main()
