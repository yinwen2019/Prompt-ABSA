import sys
import time
import argparse
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import transformers
from tqdm import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM


import utils
sys.path.append("..")
from models.HardPrompt import AutoHardPromptModel


logger = logging.getLogger(__name__)


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """

    def __init__(self, module):
        self.grad_in = None
        self.grad_out = None
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]
        self.grad_out = grad_out
        self.grad_in = grad_in

    def get(self):
        return self._stored_gradient


class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """

    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs, trigger_ids):
        # Copy dict so pop operations don't have unwanted side-effects
        input_list = model_inputs.copy()
        trigger_mask = input_list.pop('trigger_mask')
        input_ids = replace_trigger_tokens(input_list, trigger_ids, trigger_mask)
        output = self._model(input_ids)
        return output


class AccuracyFn:
    """
    Computing the accuracy when a label is mapped to multiple tokens is difficult in the current
    framework, since the data generator only gives us the token ids. To get around this we
    compare the target logp to the logp of all labels. If target logp is greater than all (but)
    one of the label logps we know we are accurate.
    """

    def __init__(self, tokenizer, label_map, device, tokenize_labels=False):
        self._all_label_ids = []
        self._pred_to_label = []
        logger.info(label_map)
        for label, label_tokens in label_map.items():
            self._all_label_ids.append(utils.encode_label(tokenizer, label_tokens, tokenize_labels).to(device))
            self._pred_to_label.append(label)
        logger.info(self._all_label_ids)

    def __call__(self, predict_logits, gold_label_ids):
        # Get total log-probability for the true label
        gold_logp = get_loss(predict_logits, gold_label_ids)

        # Get total log-probability for all labels
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)

        # Add up the number of entries where loss is less than or equal to gold_logp.
        ge_count = all_label_logp.le(gold_logp.unsqueeze(-1)).sum(-1)
        correct = ge_count.le(1)  # less than in case of num. prec. issues

        return correct.float()

    # TODO: @rloganiv - This is hacky. Replace with something sensible.
    def predict(self, predict_logits):
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]
        return predictions


class modeloption:
    def __init__(self):
        self.polarities_dim = 3
        self.bert_dropout = 0.3
        self.attention_heads = 5


def load_pretrained(model_name):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    config = AutoConfig.from_pretrained(model_name)
    bertformaskedmodel = AutoModelForMaskedLM.from_pretrained(model_name)
    opt = modeloption()
    model = AutoHardPromptModel(bertformaskedmodel, opt)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
    utils.add_task_specific_tokens(tokenizer)
    bertformaskedmodel.resize_token_embeddings(len(tokenizer))
    embeddings = get_embeddings(bertformaskedmodel, config)
    return config, model, tokenizer, embeddings


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_embeddings(model, config):
    """Returns the wordpiece embedding module."""
    base_model = getattr(model, config.model_type)
    embeddings = base_model.embeddings.word_embeddings
    return embeddings


def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    """Replaces the trigger tokens in input_ids."""
    out = model_inputs.copy()
    input_ids = model_inputs['input_ids']
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)

    try:
        filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
    except RuntimeError:
        filled = input_ids
    out['input_ids'] = filled
    return out


def get_loss(predict_logits, label_ids):
    # predict_logp = F.log_softmax(predict_logits, dim=-1)
    # target_logp = predict_logp.gather(-1, label_ids)
    # target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
    # target_logp = torch.logsumexp(target_logp, dim=-1)

    return F.cross_entropy(predict_logits, label_ids)

def get_correct_num(predict_logits,label_ids):
    predict_incide = torch.argmax(predict_logits, -1)
    num = predict_incide.eq(label_ids).sum()
    return num

def isupper(idx, tokenizer):
    """
    Determines whether a token (e.g., word piece) begins with a capital letter.
    """
    _isupper = False
    # We only want to check tokens that begin words. Since byte-pair encoding
    # captures a prefix space, we need to check that the decoded token begins
    # with a space, and has a capitalized second character.
    if isinstance(tokenizer, transformers.GPT2Tokenizer):
        decoded = tokenizer.decode([idx])
        if decoded[0] == ' ' and decoded[1].isupper():
            _isupper = True
    # For all other tokenization schemes, we can just check the first character
    # is capitalized.
    elif tokenizer.decode([idx])[0].isupper():
        _isupper = True
    return _isupper


def run_model(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading model, tokenizer, etc.')
    config, model, tokenizer, embeddings = load_pretrained(args.model_name)
    model.to(device)
    # register embeddings backward hook function
    embedding_gradient = GradientStorage(embeddings)
    predictor = PredictWrapper(model)
    no_decay = ['bias', 'LayerNorm.weight']

    logger.info("bert learning rate on")
    _params = filter(lambda n, p: p.requires_grad, model.named_parameters())
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.0},
        {'params': [p for n, p in model.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=2e-5, eps=1e-8)
    templatizer = utils.TriggerTemplatizer(
        args.template,
        config,
        tokenizer,
        label_field=args.label_field,
        add_special_tokens=False,
    )

    # Obtain the initial trigger tokens and label mapping
    if args.initial_trigger:
        trigger_ids = tokenizer.convert_tokens_to_ids(args.initial_trigger)
        logger.debug(f'Initial trigger: {args.initial_trigger}')
        logger.info(f'Trigger ids: {trigger_ids}')
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
        logger.info(f'Trigger ids: {trigger_ids}')
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
    best_trigger_ids = trigger_ids.clone()

    logger.info('Loading datasets')
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)

    train_dataset = utils.AutoPromptData(args.train, templatizer, train_pre=0.02)
    dev_dataset = utils.AutoPromptData(args.dev, templatizer)
    train_dataloder = DataLoader(dataset=train_dataset, batch_size=args.bsz, shuffle=True)
    dev_dataloder = DataLoader(dataset=dev_dataset, batch_size=args.bsz)


    # first time evaluate dev acc
    logger.info('Evaluating')
    numerator = 0
    denominator = 0
    for sample_batched in tqdm(dev_dataloder):
        inputs = {k: v.to(device) for k, v in sample_batched.items()}
        labels = sample_batched['polarity'].to(device)
        with torch.no_grad():
            predict_logits = predictor(inputs, trigger_ids)
        numerator += get_correct_num(predict_logits, labels)
        denominator += labels.size(0)
    dev_metric = numerator / (denominator + 1e-13)
    logger.info(f'Dev metric: {dev_metric}')

    best_dev_metric = -float('inf')
    # Measure elapsed time of trigger search
    start = time.time()

    # start iterate
    for i in range(args.iters):

        logger.info(f'Iteration: {i}')
        logger.info('Accumulating Gradient')

        averaged_grad = None
        Trainsum = 0
        # Accumulate
        for sample_batched in tqdm(train_dataloder):
            model.train()
            optimizer.zero_grad()
            model_inputs = {k: v.to(device) for k, v in sample_batched.items()}
            labels = sample_batched['polarity'].to(device)
            Trainsum += labels.size(0)
            predict_logits = predictor(model_inputs, trigger_ids)
            loss = get_loss(predict_logits, labels)
            loss.backward()
            optimizer.step()
            # get emdedding grad_out
            grad = embedding_gradient.get()
            bsz, _, emb_dim = grad.size()
            selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
            grad = torch.masked_select(grad, selection_mask)
            grad = grad.view(bsz, templatizer.num_trigger_tokens, emb_dim)

            if averaged_grad is None:
                averaged_grad = grad.sum(dim=0) / len(train_dataloder)
            else:
                averaged_grad += grad.sum(dim=0) / len(train_dataloder)

        # evaluate candidates
        logger.info('Evaluating Candidates')

        # get random trigger top-k grad indices from vocab
        token_to_flip = random.randrange(templatizer.num_trigger_tokens)
        candidates = hotflip_attack(averaged_grad[token_to_flip],
                                    embeddings.weight,
                                    increase_loss=False,
                                    num_candidates=args.num_cand)

        current_score = 0
        candidate_scores = torch.zeros(args.num_cand, device=device)
        denom = 0
        for sample_batched in tqdm(train_dataloder):

            model_inputs = {k: v.to(device) for k, v in sample_batched.items()}
            labels = sample_batched['polarity'].to(device)
            with torch.no_grad():
                predict_logits = predictor(model_inputs, trigger_ids)
                correct_num = get_correct_num(predict_logits, labels)

            # Update current score
            current_score += correct_num
            denom += labels.size(0)

            # NOTE: Instead of iterating over tokens to flip we randomly change just one each
            # time so the gradients don't get stale.
            for c, candidate in enumerate(candidates):
                temp_trigger = trigger_ids.clone()
                temp_trigger[:, token_to_flip] = candidate
                with torch.no_grad():
                    predict_logits = predictor(model_inputs, temp_trigger)
                    correct_num4cand = get_correct_num(predict_logits, labels)

                candidate_scores[c] += correct_num4cand
        logger.info(f'Train metric: {current_score / (denom + 1e-13): 0.4f}')
        for candindex in candidate_scores:
            logger.info(f'Train metric: {candindex / (denom + 1e-13): 0.4f}')

        if (candidate_scores > current_score).any():
            logger.info('Better trigger detected.')
            best_candidate_score = candidate_scores.max()
            best_candidate_idx = candidate_scores.argmax()
            trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
            logger.info(f'Train metric: {best_candidate_score / (denom + 1e-13): 0.4f}')
        else:
            logger.info('No improvement detected. Skipping evaluation.')
            continue

        # evaluate dev acc
        logger.info('Evaluating')
        numerator = 0
        denominator = 0
        for sample_batched in tqdm(dev_dataloder):
            model_inputs = {k: v.to(device) for k, v in sample_batched.items()}
            labels = sample_batched['polarity'].to(device)
            with torch.no_grad():
                predict_logits = predictor(model_inputs, trigger_ids)
            numerator += get_correct_num(predict_logits, labels)
            denominator += labels.size(0)
        dev_metric = numerator / (denominator + 1e-13)

        logger.info(f'Trigger tokens: {tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))}')
        logger.info(f'Dev metric: {dev_metric}')

        # TODO: Something cleaner. LAMA templates can't have mask tokens, so if
        # there are still mask tokens in the trigger then set the current score
        # to -inf.
        if best_trigger_ids.eq(tokenizer.mask_token_id).any():
            best_dev_metric = float('-inf')

        if dev_metric > best_dev_metric:
            logger.info('Best performance so far')
            best_trigger_ids = trigger_ids.clone()
            best_dev_metric = dev_metric

    best_trigger_tokens = tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))
    logger.info(f'Best tokens: {best_trigger_tokens}')
    logger.info(f'Best dev metric: {best_dev_metric}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, help='Train data path', default='../dataset/Laptops/train.json')
    parser.add_argument('--dev', type=Path, help='Dev data path', default='../dataset/Laptops/test.json')
    parser.add_argument('--template', type=str, default='[CLS] {sentence} [SEP] {aspect} [T] [T] [T] [P] . [SEP]',
                        help='Template string')

    parser.add_argument('--initial-trigger', nargs='+', type=str, default=None, help='Manual prompt')
    parser.add_argument('--label-field', type=str, default='label',
                        help='Name of the label field')

    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--eval-size', type=int, default=256, help='Eval size')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--model-name', type=str, default="/home/hxm/PLM/bert-base-uncased",
                        help='Model name passed to HuggingFace AutoX classes.')
    parser.add_argument('--seed', type=int, default=2)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--use-ctx', action='store_true',
                        help='Use context sentences for relation extraction only')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num-cand', type=int, default=10)
    parser.add_argument('--sentence-size', type=int, default=100)

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # args.template = '[CLS] {sentence} [T] [T] [T] [P] . [SEP]'
    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    run_model(args)
