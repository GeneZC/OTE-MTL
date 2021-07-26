# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.dynamic_rnn import DynamicRNN
from tag_utils import bio2bieos, bieos2span, find_span_with_end

class Biaffine(nn.Module):
    def __init__(self, opt, in1_features, in2_features, out_features, bias=(True, True)):
        super(Biaffine, self).__init__()
        self.opt = opt
        self.in1_features = in1_features
        self.in2_features = in2_features
        self.out_features = out_features
        self.bias = bias
        self.linear_input_size = in1_features + int(bias[0])
        self.linear_output_size = out_features * (in2_features + int(bias[1]))
        self.linear = nn.Linear(in_features=self.linear_input_size,
                                out_features=self.linear_output_size,
                                bias=False)

    def forward(self, input1, input2):
        batch_size, len1, dim1 = input1.size()
        batch_size, len2, dim2 = input2.size()
        if self.bias[0]:
            ones = torch.ones(batch_size, len1, 1).to(self.opt.device)
            input1 = torch.cat((input1, ones), dim=2)
            dim1 += 1
        if self.bias[1]:
            ones = torch.ones(batch_size, len2, 1).to(self.opt.device)
            input2 = torch.cat((input2, ones), dim=2)
            dim2 += 1
        affine = self.linear(input1)
        affine = affine.view(batch_size, len1*self.out_features, dim2)
        input2 = torch.transpose(input2, 1, 2)
        biaffine = torch.bmm(affine, input2)
        biaffine = torch.transpose(biaffine, 1, 2)
        biaffine = biaffine.contiguous().view(batch_size, len2, len1, self.out_features)

        return biaffine

class MTL(nn.Module):
    def __init__(self, embedding_matrix, opt, idx2tag, idx2polarity):
        super(MTL, self).__init__()
        self.opt = opt
        self.idx2tag = idx2tag
        self.tag_dim = len(self.idx2tag)
        self.idx2polarity = idx2polarity
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.embed_dropout = nn.Dropout(0.5)
        self.lstm = DynamicRNN(opt.embed_dim, opt.hidden_dim, batch_first=True, bidirectional=True)
        self.ap_fc = nn.Linear(2 * opt.hidden_dim, 200)
        self.op_fc = nn.Linear(2 * opt.hidden_dim, 200)
        self.triplet_biaffine = Biaffine(opt, 100, 100, opt.polarities_dim, bias=(True, False))
        self.ap_tag_fc = nn.Linear(100, self.tag_dim)
        self.op_tag_fc = nn.Linear(100, self.tag_dim)

    def calc_loss(self, outputs, targets):
        ap_out, op_out, triplet_out = outputs
        ap_tag, op_tag, triplet, mask = targets
        # tag loss
        ap_tag_loss = F.cross_entropy(ap_out.flatten(0, 1), ap_tag.flatten(0, 1), reduction='none')
        ap_tag_loss = ap_tag_loss.masked_select(mask.flatten(0, 1)).sum() / mask.sum()
        op_tag_loss = F.cross_entropy(op_out.flatten(0, 1), op_tag.flatten(0, 1), reduction='none')
        op_tag_loss = op_tag_loss.masked_select(mask.flatten(0, 1)).sum() / mask.sum()
        tag_loss = ap_tag_loss + op_tag_loss
        # sentiment loss
        mat_mask = mask.unsqueeze(2)*mask.unsqueeze(1)
        sentiment_loss = F.cross_entropy(triplet_out.view(-1, self.opt.polarities_dim), triplet.view(-1), reduction='none')
        sentiment_loss = sentiment_loss.masked_select(mat_mask.view(-1)).sum() / mat_mask.sum()
        return tag_loss + sentiment_loss

    def forward(self, inputs):
        text_indices, text_mask = inputs
        text_len = torch.sum(text_mask, dim=-1)

        embed = self.embed(text_indices)
        embed = self.embed_dropout(embed)

        out, (_, _) = self.lstm(embed, text_len)
        
        ap_rep = F.relu(self.ap_fc(out))
        op_rep = F.relu(self.op_fc(out))
        ap_node, ap_rep = torch.chunk(ap_rep, 2, dim=2)
        op_node, op_rep = torch.chunk(op_rep, 2, dim=2)

        ap_out = self.ap_tag_fc(ap_rep)
        op_out = self.op_tag_fc(op_rep)

        triplet_out = self.triplet_biaffine(ap_node, op_node)

        return [ap_out, op_out, triplet_out]

    def inference(self, inputs):
        text_indices, text_mask = inputs
        text_len = torch.sum(text_mask, dim=-1)
        embed = self.embed(text_indices)
        out, (_, _) = self.lstm(embed, text_len)
        ap_rep = F.relu(self.ap_fc(out))
        op_rep = F.relu(self.op_fc(out))
        ap_node, ap_rep = torch.chunk(ap_rep, 2, dim=2)
        op_node, op_rep = torch.chunk(op_rep, 2, dim=2)

        ap_out = self.ap_tag_fc(ap_rep)
        op_out = self.op_tag_fc(op_rep)

        triplet_out = self.triplet_biaffine(ap_node, op_node)

        batch_size = text_len.size(0)
        ap_tags = [[] for _ in range(batch_size)]
        op_tags = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            for i in range(text_len[b]):
                ap_tags[b].append(ap_out[b, i, :].argmax(0).item())
        for b in range(batch_size):
            for i in range(text_len[b]):
                op_tags[b].append(op_out[b, i, :].argmax(0).item())

        text_indices = text_indices.cpu().numpy().tolist()
        ap_spans = self.aspect_decode(text_indices, ap_tags, self.idx2tag)
        op_spans = self.opinion_decode(text_indices, op_tags, self.idx2tag)
        mat_mask = (text_mask.unsqueeze(2)*text_mask.unsqueeze(1)).unsqueeze(3).expand(
                              -1, -1, -1, self.opt.polarities_dim)  # batch x seq x seq x polarity
        triplet_indices = torch.zeros_like(triplet_out).to(self.opt.device)
        triplet_indices = triplet_indices.scatter_(3, triplet_out.argmax(dim=3, keepdim=True), 1) * mat_mask.float()
        triplet_indices = torch.nonzero(triplet_indices).cpu().numpy().tolist()
        triplets = self.sentiment_decode(text_indices, ap_tags, op_tags, triplet_indices, self.idx2tag, self.idx2polarity)
        
        return [ap_spans, op_spans, triplets]

    @staticmethod
    def aspect_decode(text_indices, tags, idx2tag):
        #text_indices = text_indices.cpu().numpy().tolist()
        batch_size = len(tags)
        result = [[] for _ in range(batch_size)]
        for i, tag_seq in enumerate(tags):
            _tag_seq = list(map(lambda x: idx2tag[x], tag_seq))
            result[i] = bieos2span(bio2bieos(_tag_seq), tp='')
        return result

    @staticmethod
    def opinion_decode(text_indices, tags, idx2tag):
        #text_indices = text_indices.cpu().numpy().tolist()
        batch_size = len(tags)
        result = [[] for _ in range(batch_size)]
        for i, tag_seq in enumerate(tags):
            _tag_seq = list(map(lambda x: idx2tag[x], tag_seq))
            result[i] = bieos2span(bio2bieos(_tag_seq), tp='')
        return result
                
    @staticmethod
    def sentiment_decode(text_indices, ap_tags, op_tags, triplet_indices, idx2tag, idx2polarity):
        #text_indices = text_indices.cpu().numpy().tolist()
        batch_size = len(ap_tags)
        result = [[] for _ in range(batch_size)]
        for i in range(len(triplet_indices)):
            b, ap_i, op_i, po = triplet_indices[i]
            if po == 0:
                continue
            _ap_tags = list(map(lambda x: idx2tag[x], ap_tags[b]))
            _op_tags = list(map(lambda x: idx2tag[x], op_tags[b]))
            ap_beg, ap_end = find_span_with_end(ap_i, text_indices[b], _ap_tags, tp='')
            op_beg, op_end = find_span_with_end(op_i, text_indices[b], _op_tags, tp='')
            triplet = (ap_beg, ap_end, op_beg, op_end, po)
            result[b].append(triplet)
        return result

