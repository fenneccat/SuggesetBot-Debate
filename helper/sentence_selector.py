from flask import Flask
from flask import request
from flask import jsonify
from collections import OrderedDict

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM
import csv
import os
import random
from tqdm import tqdm, trange, tqdm_notebook
import logging

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.modeling import BertForSequenceClassification
from pytorch_pretrained_bert.optimization import BertAdam
from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE

import re
from unidecode import unidecode
from pathlib import Path

logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s', level=logging.WARN)

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""
    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_examples_to_features(examples, label_list, max_seq_length, tokenizer):
        """Loads a data file into a list of `InputBatch`s."""

        label_map = {label : i for i, label in enumerate(label_list)}

        features = []
        for (ex_index, example) in enumerate(examples):
            tokens_a = tokenizer.tokenize(example.text_a)

            tokens_b = None
            if example.text_b:
                tokens_b = tokenizer.tokenize(example.text_b)
                # Modifies `tokens_a` and `tokens_b` in place so that the total
                # length is less than the specified length.
                # Account for [CLS], [SEP], [SEP] with "- 3"
                _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
            else:
                # Account for [CLS] and [SEP] with "- 2"
                if len(tokens_a) > max_seq_length - 2:
                    tokens_a = tokens_a[:(max_seq_length - 2)]

            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids: 0   0   0   0  0     0 0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambigiously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
            segment_ids = [0] * len(tokens)

            if tokens_b:
                tokens += tokens_b + ["[SEP]"]
                segment_ids += [1] * (len(tokens_b) + 1)

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1] * len(input_ids)

            # Zero-pad up to the sequence length.
            padding = [0] * (max_seq_length - len(input_ids))
            input_ids += padding
            input_mask += padding
            segment_ids += padding

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            #print(label_map)
            #print(example.label)
            label_id = label_map[example.label]

            features.append(
                    InputFeatures(input_ids=input_ids,
                                  input_mask=input_mask,
                                  segment_ids=segment_ids,
                                  label_id=label_id))
        return features


class SentenceSelector:
    # For evidence classification
    
    def __init__(self, model_name='weights.02-0.3496.bin'):
        # init logger
        self.logger = logging.getLogger(__name__)
        # load model
        self.model, self.tokenizer, self.device = self._init_evidence_classifier(model_name)
        self.n_gpu = torch.cuda.device_count()

    def _init_evidence_classifier(self, weight_path):
        self.logger.warn('Loading SentenceSelector models...')
        
        # Load a trained model that you have fine-tuned
        model_dir = Path('./model')
        #model_dir = Path('./model')
        
        output_model_file = str(model_dir / weight_path) ## put pretrained weight from training
        num_labels = 2

        model_state_dict = torch.load(output_model_file)
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', state_dict=model_state_dict, num_labels=num_labels)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model.to(device)
        
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

        return model, tokenizer, device
    
    
    def get_evidences(self, claim, candidates):
        
        self.logger.warn('Finding Sentences...')
        processor = SuggestBotProcessor()
        label_list = processor.get_labels()
        max_seq_length = 128

        eval_examples = processor.get_test_examples(claim, candidates, 'dev')
        eval_features = convert_examples_to_features(
            eval_examples, label_list, max_seq_length, self.tokenizer)

        all_input_ids = torch.tensor([f.input_ids for f in eval_features], dtype=torch.long)
        all_input_mask = torch.tensor([f.input_mask for f in eval_features], dtype=torch.long)
        all_segment_ids = torch.tensor([f.segment_ids for f in eval_features], dtype=torch.long)
        all_label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)
        eval_data = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)

        eval_batch_size = 50

        # Run prediction for full data
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

        score = []

        for input_ids, input_mask, segment_ids, label_ids in tqdm(eval_dataloader):
            input_ids = input_ids.to(self.device)
            input_mask = input_mask.to(self.device)
            segment_ids = segment_ids.to(self.device)
            label_ids = label_ids.to(self.device)

            with torch.no_grad():
                tmp_eval_loss = self.model(input_ids, segment_ids, input_mask, label_ids)
                logits = self.model(input_ids, segment_ids, input_mask)

            logits = logits.detach().cpu().numpy()
            label_ids = label_ids.to('cpu').numpy()

            if(score == []): score = logits
            else: score = np.concatenate((score, logits), axis=0)


        theta = 1
        # looping through rows of score
        ps = np.empty(score.shape)
        for i in range(score.shape[0]):
            ps[i,:]  = np.exp(score[i,:] * theta)
            ps[i,:] /= np.sum(ps[i,:])

        positive_confidence = ps[:,1]

        ranked_evidence = [(x,y) for y,x in sorted(zip(positive_confidence, candidates), key = lambda x: x[0], reverse = True)]

        return (claim, ranked_evidence)


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label



class SuggestBotProcessor():
    """Processor for the MRPC data set (GLUE version)."""

    def get_labels(self):
        """See base class."""
        return ["0", "1"]
    
    def clean_sent(self, sentence):
        regex = re.compile('[^a-zA-Z0-9]')
        sentence = unidecode(sentence)
        sentence = sentence.replace('-LRB-', ' ')
        sentence = sentence.replace('-RRB-', ' ')
        sentence = sentence.replace('[REF]', ' ')
        sentence = sentence.replace('[REF', ' ')
        sentence = regex.sub(' ', sentence)

        sentence = re.sub('^\s+|\s+$|\s+(?=\s)', "", sentence)
        
        return sentence.lower()

    def get_test_examples(self, claim, candidates, set_type): ## set_type: "train" or "dev"
        """Creates examples for the training and dev sets."""
        examples = []
        cln_claim = self.clean_sent(claim)
        for (i, sentence) in enumerate(candidates):
            guid = "%s-%s" % (set_type, i)
            text_a = cln_claim ## claim
            text_b = self.clean_sent(sentence) ## sentence
            label = '0' ## label
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

