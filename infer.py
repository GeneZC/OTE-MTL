# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F
import argparse
from bucket_iterator import BucketIterator
from data_utils import ABSADataReader, build_tokenizer, build_embedding_matrix
from models import CMLA, HAST, OTE


class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
       
        absa_data_reader = ABSADataReader(data_dir=opt.data_dir)
        self.tokenizer = build_tokenizer(data_dir=opt.data_dir)
        embedding_matrix = build_embedding_matrix(opt.data_dir, self.tokenizer.word2idx, opt.embed_dim, opt.dataset)
        self.idx2tag, self.idx2polarity = absa_data_reader.reverse_tag_map, absa_data_reader.reverse_polarity_map
        self.model = opt.model_class(embedding_matrix, opt, self.idx2tag, self.idx2polarity).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        self.model.load_state_dict(torch.load(opt.state_dict_path, map_location=lambda storage, loc: storage))
        # switch model to evaluation mode
        self.model.eval()
        torch.autograd.set_grad_enabled(False)

    def evaluate(self, text):
        text_indices = self.tokenizer.text_to_sequence(text)
        text_mask = [1] * len(text_indices)
        t_sample_batched = {
            'text_indices': torch.tensor([text_indices]),
            'text_mask': torch.tensor([text_mask], dtype=torch.uint8),
        }
        with torch.no_grad():
            t_inputs = [t_sample_batched[col].to(self.opt.device) for col in self.opt.input_cols]
            t_ap_spans_pred, t_op_spans_pred, t_triplets_pred = self.model.inference(t_inputs)
        
        return [t_ap_spans_pred, t_op_spans_pred, t_triplets_pred]


if __name__ == '__main__':
    dataset = 'laptop14'
    # set your trained models here
    model_state_dict_paths = {
        'ote': 'state_dict/ote_'+dataset+'.pkl',
    }
    model_classes = {
        'ote': OTE,
    }
    input_colses = {
        'ote': ['text_indices', 'text_mask'],
    }
    target_colses = {
        'ote': ['ap_indices', 'op_indices', 'triplet_indices', 'text_mask'],
    }
    data_dirs = {
        'laptop14': 'datasets/14lap',
        'rest14': 'datasets/14rest',
        'rest15': 'datasets/15rest',
        'rest16': 'datasets/16rest',
    }
    class Option(object): pass
    opt = Option()
    opt.dataset = dataset
    opt.model_name = 'ote'
    opt.eval_cols = ['ap_spans', 'op_spans','triplets']
    opt.model_class = model_classes[opt.model_name]
    opt.input_cols = input_colses[opt.model_name]
    opt.target_cols = target_colses[opt.model_name]
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    opt.embed_dim = 300
    opt.hidden_dim = 300
    opt.polarities_dim = 4
    opt.batch_size = 32
    opt.data_dir = data_dirs[opt.dataset]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    inf = Inferer(opt)

    #rest
    #text = 'Great food but the service was dreadful !'
    #text = 'the atmosphere is attractive , but a little uncomfortable .'
    #laptop
    text = 'I am pleased with the fast log on , speedy WiFi connection and the long battery life ( > 6 hrs ) .'
    triplets = inf.evaluate(text)[2][0]
    words = text.split()
    polarity_map = {0:'N', 1:'NEU', 2:'NEG', 3:'POS'}
    for triplet in triplets:
        ap_beg, ap_end, op_beg, op_end, p = triplet
        ap = ' '.join(words[ap_beg:ap_end+1])
        op = ' '.join(words[op_beg:op_end+1])
        polarity = polarity_map[p]
        print(f'{ap}, {op}, {polarity}')

