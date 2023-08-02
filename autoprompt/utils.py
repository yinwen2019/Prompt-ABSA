import csv
import copy
import json
import logging
from multiprocessing.sharedctypes import Value
import random
from collections import defaultdict

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm

MAX_CONTEXT_LEN = 100

logger = logging.getLogger(__name__)


def pad_squeeze_sequence(sequence, *args, **kwargs):
    """Squeezes fake batch dimension added by tokenizer before padding sequence."""
    return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)


class OutputStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """

    def __init__(self, module):
        self._stored_output = None
        module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self._stored_output = output

    def get(self):
        return self._stored_output


class ExponentialMovingAverage:
    def __init__(self, weight=0.3):
        self._weight = weight
        self.reset()

    def update(self, x):
        self._x += x
        self._i += 1

    def reset(self):
        self._x = 0
        self._i = 0

    def get_metric(self):
        return self._x / (self._i + 1e-13)


class Collator:
    """
    Collates transformer outputs.
    """

    def __init__(self, pad_token_id=0):
        self._pad_token_id = pad_token_id

    def __call__(self, features):
        # Separate the list of inputs and labels
        model_inputs, labels = list(zip(*features))
        model_inputs = list(zip(model_inputs))
        print(model_inputs)
        return model_inputs, labels


def encode_label(tokenizer, label, tokenize=False):
    """
    Helper function for encoding labels. Deals with the subtleties of handling multiple tokens.
    """
    if isinstance(label, str):
        if tokenize:
            # Ensure label is properly tokenized, and only retain first token
            # if it gets split into multiple tokens. TODO: Make sure this is
            # desired behavior.
            tokens = tokenizer.tokenize(label)
            if len(tokens) > 1:
                raise ValueError(f'Label "{label}" gets mapped to multiple tokens.')
            if tokens[0] == tokenizer.unk_token:
                raise ValueError(f'Label "{label}" gets mapped to unk.')
            label = tokens[0]
        return torch.tensor(tokenizer.convert_tokens_to_ids([label])).unsqueeze(0)
    elif isinstance(label, list):
        return torch.tensor(tokenizer.convert_tokens_to_ids(label)).unsqueeze(0)
    elif isinstance(label, int):
        return torch.tensor([[label]])


class TriggerTemplatizer:
    """
    An object to facilitate creating transformers-friendly triggers inputs from a template.

    Parameters
    ==========
    template : str
        The template string, comprised of the following tokens:
            [T] to mark a trigger placeholder.
            [P] to mark a prediction placeholder.
            {fields} arbitrary fields instantiated from the dataset instances.
        For example a NLI template might look like:
            "[T] [T] [T] {premise} [P] {hypothesis}"
    tokenizer : PretrainedTokenizer
        A HuggingFace tokenizer. Must have special trigger and predict tokens.
    add_special_tokens : bool
        Whether or not to add special tokens when encoding. Default: False.
    """

    def __init__(self,
                 template,
                 config,
                 tokenizer,
                 label_field='label',
                 add_special_tokens=False, ):
        if not hasattr(tokenizer, 'predict_token') or \
                not hasattr(tokenizer, 'trigger_token'):
            raise ValueError(
                'Tokenizer missing special trigger and predict tokens in vocab.'
                'Use `utils.add_special_tokens` to add them.'
            )
        self._template = template
        self._config = config
        self.max_length = 100
        self._tokenizer = tokenizer
        self._label_field = label_field
        self._add_special_tokens = add_special_tokens

    @property
    def num_trigger_tokens(self):
        return sum(token == '[T]' for token in self._template.split())

    def __call__(self, format_kwargs):
        obj = format_kwargs.copy()
        polarity_dict = {'positive': 0, 'negative': 1, 'neutral': 2}
        polarity = polarity_dict[obj['label']]
        term_start = obj['aspect_post'][0]
        term_end = obj['aspect_post'][1]
        text_list = obj['text_list']

        left, term, right = text_list[: term_start], text_list[term_start: term_end], text_list[term_end:]
        trigger_tokens = ['[T]'] * self.num_trigger_tokens
        tokenizer = self._tokenizer
        left_tokens, term_tokens, right_tokens  = [], [], []
        for ori_i, w in enumerate(left):
            for t in tokenizer.tokenize(w):
                left_tokens.append(t)
        for ori_i, w in enumerate(term):
            for t in tokenizer.tokenize(w):
                term_tokens.append(t)
        for ori_i, w in enumerate(right):
            for t in tokenizer.tokenize(w):
                right_tokens.append(t)

        while len(left_tokens) + len(right_tokens) + len(trigger_tokens) > self.max_length - 2 * len(term_tokens) - 4:
            if len(left_tokens) > len(right_tokens):
                left_tokens.pop(0)
            else:
                right_tokens.pop()
        rawtext_tokens = left_tokens + term_tokens + right_tokens
        # Format the template string
        context_ids = [tokenizer.cls_token_id] + tokenizer.convert_tokens_to_ids(rawtext_tokens) + \
                      [tokenizer.sep_token_id] + tokenizer.convert_tokens_to_ids(term_tokens) + \
                      tokenizer.convert_tokens_to_ids(trigger_tokens) + [tokenizer.convert_tokens_to_ids('[P]')] + [tokenizer.sep_token_id]

        context_len = len(context_ids)

        paddings = [0] * (self.max_length - context_len)
        rawtext_len = len(rawtext_tokens)
        context_second_ids = [0] * (1 + rawtext_len + 1) + [1] * (len(term_tokens) + len(trigger_tokens) + 2) + paddings
        context_attention_mask = [1] * context_len + paddings  # allcontext_mask
        context_ids += paddings
        context_ids = torch.tensor(np.asarray(context_ids, dtype='int64'))
        context_second_ids = torch.tensor(np.asarray(context_second_ids, dtype='int64'))
        context_attention_mask = torch.tensor(np.asarray(context_attention_mask, dtype='int64'))
        trigger_mask = context_ids.eq(self._tokenizer.trigger_token_id)
        predict_mask = context_ids.eq(self._tokenizer.predict_token_id)
        obj['input_ids'] = context_ids
        obj['trigger_mask'] = torch.tensor(trigger_mask)
        obj['predict_mask'] = torch.tensor(predict_mask)
        obj['token_type_ids'] = context_second_ids
        obj['attention_mask'] = context_attention_mask
        obj['polarity'] = polarity

        return obj


def add_task_specific_tokens(tokenizer):
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[T]', '[P]']
    })
    tokenizer.trigger_token = '[T]'
    tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')
    tokenizer.predict_token = '[P]'
    tokenizer.predict_token_id = tokenizer.convert_tokens_to_ids('[P]')


def load_tsv(fname):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            yield row


def load_jsonl(fname):
    with open(fname, 'r') as f:
        for line in f:
            yield json.loads(line)


def load_json(fname):
    with open(fname, 'r') as f:
        data = json.load(f)
        for line in data:
            yield line


LOADERS = {
    '.tsv': load_tsv,
    '.jsonl': load_jsonl,
    '.json': load_json
}


class AutoPromptData(Dataset):
    def __init__(self, fname, templatizer, train_pre=None):
        rawdata = []
        self.data = []
        loader = LOADERS[fname.suffix]

        for d in loader(fname):
            for aspect in d['aspects']:
                text_list = list(d['token'])
                tok = list(d['token'])  # word token
                length = len(tok)  # real length
                # if args.lower == True:
                tok = [t.lower() for t in tok]
                tok = ' '.join(tok)
                asp = list(aspect['term'])  # aspect
                asp = [a.lower() for a in asp]
                asp = ' '.join(asp)
                label = aspect['polarity']  # label
                pos = list(d['pos'])  # pos_tag
                head = list(d['head'])  # head
                deprel = list(d['deprel'])  # deprel

                # position
                aspect_post = [aspect['from'], aspect['to']]
                post = [i - aspect['from'] for i in range(aspect['from'])] \
                       + [0 for _ in range(aspect['from'], aspect['to'])] \
                       + [i - aspect['to'] + 1 for i in range(aspect['to'], length)]
                # aspect mask
                if len(asp) == 0:
                    mask = [1 for _ in range(length)]
                else:
                    mask = [0 for _ in range(aspect['from'])] \
                           + [1 for _ in range(aspect['from'], aspect['to'])] \
                           + [0 for _ in range(aspect['to'], length)]

                sample = {'text': tok, 'aspect': asp, 'pos': pos, 'post': post, 'head': head,
                          'deprel': deprel, 'length': length, 'label': label, 'mask': mask,
                          'aspect_post': aspect_post, 'text_list': text_list}
                rawdata.append(sample)

        for obj in tqdm(rawdata, total=len(rawdata)):
            modelinput = templatizer(obj)
            data = {
                'input_ids': modelinput["input_ids"],
                'token_type_ids': modelinput["token_type_ids"],
                'attention_mask': modelinput["attention_mask"],
                'polarity': modelinput["polarity"],
                'trigger_mask': modelinput['trigger_mask'],
                'predict_mask': modelinput['predict_mask']
            }
            self.data.append(data)
        if train_pre is not None:
            train_size = int(train_pre * len(self.data))
            self.data = self.data[:train_size]
        else:
            self.data = self.data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def load_trigger_dataset(fname, templatizer, limit=None):
    loader = LOADERS[fname.suffix]
    instances = []

    for x in loader(fname):
        for aspect in x['aspects']:
            text = list(x['token'])  # text
            text = [t.lower() for t in text]
            text = ' '.join(text)
            asp = list(aspect['term'])  # aspect token
            asp = [a.lower() for a in asp]
            asp = ' '.join(asp)
            label = aspect['polarity']  # label

            try:
                data = {'sentence': text, 'aspect': asp, 'label': label}
                model_inputs, label_id = templatizer(data)
            except ValueError as e:
                logger.warning('Encountered error "%s" when processing "%s".  Skipping.', e, x)
                continue
            else:
                instances.append((model_inputs, label_id))
    if limit:
        return random.sample(instances, limit)
    else:
        return instances
