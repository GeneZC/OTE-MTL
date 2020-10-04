# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.dynamic_rnn import DynamicRNN
from tag_utils import bio2bieos, bieos2span, find_span_with_end

class CoupledAttn(nn.Module):
    def __init__(self, K=20, hidden_dim=100):
        super(CoupledAttn, self).__init__()
        self.K = K
        self.hidden_dim = hidden_dim
        self.G_a = nn.Linear(hidden_dim, K * hidden_dim, bias=False)
        self.G_p = nn.Linear(hidden_dim, K * hidden_dim, bias=False)
        self.D_a = nn.Linear(hidden_dim, K * hidden_dim, bias=False)
        self.D_p = nn.Linear(hidden_dim, K * hidden_dim, bias=False)
        self.gru_ra = DynamicRNN(input_size=2 * K, hidden_size=2 * K, batch_first=True, rnn_type='GRU')
        self.gru_rp = DynamicRNN(input_size=2 * K, hidden_size=2 * K, batch_first=True, rnn_type='GRU')
        self.va = nn.Linear(2 * K, 1, bias=False)
        self.vp = nn.Linear(2 * K, 1, bias=False)

    def forward(self, inputs):
        (h, h_len), (u_a, u_p) = inputs
        # h: [batch_size, seq_len, hidden_dim]
        # u_a: [hidden_dim]
        # u_p: [hidden_dim]

        batch_size, seq_len, _ = h.size()
        if u_a.dim() == 1:
            u_a = u_a.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1).unsqueeze(3)
            u_p = u_p.unsqueeze(0).unsqueeze(1).expand(batch_size, seq_len, -1).unsqueeze(3)
        else:
            u_a = u_a.unsqueeze(1).expand(-1, seq_len, -1).unsqueeze(3)
            u_p = u_p.unsqueeze(1).expand(-1, seq_len, -1).unsqueeze(3)

        # [batch_size, seq_len, K] <= [batch_size, seq_len, K, hidden_dim] * [hidden_dim, 1]
        h_G_ua = torch.matmul(self.G_a(h).view(batch_size, seq_len, self.K, self.hidden_dim), u_a).squeeze(3)
        h_D_up = torch.matmul(self.D_a(h).view(batch_size, seq_len, self.K, self.hidden_dim), u_p).squeeze(3)
        h_G_up = torch.matmul(self.G_p(h).view(batch_size, seq_len, self.K, self.hidden_dim), u_p).squeeze(3)
        h_D_ua = torch.matmul(self.D_p(h).view(batch_size, seq_len, self.K, self.hidden_dim), u_a).squeeze(3)
        # [batch_size, seq_len, 2K]
        fa = torch.tanh(torch.cat([h_G_ua, h_D_up], -1))
        fp = torch.tanh(torch.cat([h_G_up, h_D_ua], -1))

        # GRU_fa
        ra, (_, _) = self.gru_ra(fa, h_len)

        # GRU_fp
        rp, (_, _) = self.gru_rp(fp, h_len)

        # scalar score
        ea = self.va(ra) # [batch_size, seq_len, 1]
        ep = self.vp(rp) # [batch_size, seq_len, 1]

        return (ra, rp), (ea, ep)

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

class CMLA(nn.Module):
    def __init__(self, embedding_matrix, opt, idx2tag, idx2polarity):
        super(CMLA, self).__init__()
        self.opt = opt
        self.idx2tag = idx2tag
        self.tag_dim = len(self.idx2tag)
        self.idx2polarity = idx2polarity
        self.embed = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float))
        self.K = 20
        self.gru = DynamicRNN(opt.embed_dim, 100, batch_first=True, rnn_type='GRU')
        self.ua = nn.Parameter(torch.FloatTensor(100))
        self.up = nn.Parameter(torch.FloatTensor(100))
        self.CA1 = CoupledAttn()
        self.CA2 = CoupledAttn()
        self.Va = nn.Linear(100, 100, bias=False)
        self.Vp = nn.Linear(100, 100, bias=False)
        self.a_fc = nn.Linear(2 * self.K, self.tag_dim, bias=False)
        self.o_fc = nn.Linear(2 * self.K, self.tag_dim, bias=False)
        self.triplet_biaffine = Biaffine(opt, 2 * self.K, 2 * self.K, opt.polarities_dim, bias=(True, False))
        
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

        # GRU: [batch_size, seq_len, hidden_dim]
        H, (_, _) = self.gru(embed, text_len)

        ua_0, up_0 = self.ua, self.up

        # first layer
        (ra_1, rp_1), (ea_1, ep_1) = self.CA1([(H, text_len), (ua_0, up_0)])

        # aspect
        oa_1 = torch.matmul(H.transpose(1, 2), F.softmax(ea_1, dim=1)).squeeze(2)
        ua_1 = torch.tanh(self.Va(ua_0)) + oa_1

        # opinion
        op_1 = torch.matmul(H.transpose(1, 2), F.softmax(ep_1, dim=1)).squeeze(2)
        up_1 = torch.tanh(self.Vp(up_0)) + op_1

        # second layer
        (ra_2, rp_2), _ = self.CA2([(H, text_len), (ua_1, up_1)])

        ap_feat = ra_2 + ra_1
        op_feat = rp_2 + rp_1
        # ap_node, ap_rep = torch.chunk(ap_feat, 2, dim=2)
        # op_node, op_rep = torch.chunk(op_feat, 2, dim=2)

        ap_out = self.a_fc(ap_feat)
        op_out = self.o_fc(op_feat)

        triplet_out = self.triplet_biaffine(ap_feat, op_feat)

        return [ap_out, op_out, triplet_out]

    def inference(self, inputs):
        text_indices, text_mask = inputs
        text_len = torch.sum(text_mask, dim=-1)

        embed = self.embed(text_indices)

        # GRU: [batch_size, seq_len, hidden_dim]
        H, (_, _) = self.gru(embed, text_len)

        ua_0, up_0 = self.ua, self.up

        # first layer
        (ra_1, rp_1), (ea_1, ep_1) = self.CA1([(H, text_len), (ua_0, up_0)])

        # aspect
        oa_1 = torch.matmul(H.transpose(1, 2), F.softmax(ea_1, dim=1)).squeeze(2)
        ua_1 = torch.tanh(self.Va(ua_0)) + oa_1

        # opinion
        op_1 = torch.matmul(H.transpose(1, 2), F.softmax(ep_1, dim=1)).squeeze(2)
        up_1 = torch.tanh(self.Vp(up_0)) + op_1

        # second layer
        (ra_2, rp_2), _ = self.CA2([(H, text_len), (ua_1, up_1)])

        ap_feat = ra_2 + ra_1
        op_feat = rp_2 + rp_1
        # ap_node, ap_rep = torch.chunk(ap_feat, 2, dim=2)
        # op_node, op_rep = torch.chunk(op_feat, 2, dim=2)

        ap_out = self.a_fc(ap_feat)
        op_out = self.o_fc(op_feat)

        triplet_out = self.triplet_biaffine(ap_feat, op_feat)

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

    