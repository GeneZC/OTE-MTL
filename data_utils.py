# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np
from tag_utils import to2bio

def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec

def build_embedding_matrix(data_dir, word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    if os.path.exists(os.path.join(data_dir, embedding_matrix_file_name)):
        print('>>> loading embedding matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(os.path.join(data_dir, embedding_matrix_file_name), 'rb'))
    else:
        print('>>> loading word vectors ...')
        # words not found in embedding index will be randomly initialized.
        embedding_matrix = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (len(word2idx), embed_dim))
        # <pad>
        embedding_matrix[0, :] = np.zeros((1, embed_dim)) 
        fname = './glove/glove.840B.300d.txt'
        word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        print('>>> building embedding matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(os.path.join(data_dir, embedding_matrix_file_name), 'wb'))
    return embedding_matrix


class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence

def build_tokenizer(data_dir):
    if os.path.exists(os.path.join(data_dir, 'word2idx.pkl')):
        print('>>> loading {0} tokenizer...'.format(data_dir))
        with open(os.path.join(data_dir, 'word2idx.pkl'), 'rb') as f:
                word2idx = pickle.load(f)
                tokenizer = Tokenizer(word2idx=word2idx)
    else:
        filenames = [os.path.join(data_dir, '%s.txt' % set_type) for set_type in ['train', 'dev', 'test']]
        all_text = ''
        for filename in filenames:
            with open(filename, 'r', encoding='utf-8') as fp:
                for line in fp:
                    text = line.strip().split('####')[0]
                    all_text += (text + ' ')
        tokenizer = Tokenizer()
        tokenizer.fit_on_text(all_text)
        print('>>> saving {0} tokenizer...'.format(data_dir))
        with open(os.path.join(data_dir, 'word2idx.pkl'), 'wb') as f:
            pickle.dump(tokenizer.word2idx, f)

    return tokenizer

class ABSADataReader(object):
    def __init__(self, data_dir):
        self.tag_map, self.reverse_tag_map = self._get_tag_map()
        self.polarity_map = {'N':0, 'NEU':1, 'NEG':2, 'POS':3} # NO_RELATION is 0
        self.reverse_polarity_map = {v:k for k,v in self.polarity_map.items()}
        self.data_dir = data_dir

    def get_train(self, tokenizer):
        return self._create_dataset('train', tokenizer)

    def get_dev(self, tokenizer):
        return self._create_dataset('dev', tokenizer)

    def get_test(self, tokenizer):
        return self._create_dataset('test', tokenizer)

    @staticmethod
    def _get_tag_map():
        tag_list = ['O', 'B', 'I']
        tag_map = {tag:i for i, tag in enumerate(tag_list)}
        reverse_tag_map = {i:tag for i, tag in enumerate(tag_list)}
        return tag_map, reverse_tag_map

    def _create_dataset(self, set_type, tokenizer):
        all_data = []

        filename = os.path.join(self.data_dir, '%s.pair' % set_type)
        fp = open(filename, 'r', encoding='utf-8')
        lines = fp.readlines()
        fp.close()

        for i in range(0, len(lines), 2):
            text = lines[i].strip()
            pairs = lines[i+1].strip().split(';')

            text_indices = tokenizer.text_to_sequence(text)
            seq_len = len(text_indices)
            ap_tags = ['O'] * seq_len
            op_tags = ['O'] * seq_len
            ap_op_tags = ['O'] * seq_len

            triplet_indices = np.zeros((seq_len, seq_len), dtype=np.int64)
            ap_spans = []
            op_spans = []
            triplets = []
            for pair in pairs:
                pair = eval(pair)
                ap_beg, ap_end = pair[0]
                op_beg, op_end = pair[1]
                polarity_str = pair[2]
                ap_tags[ap_beg:ap_end+1] = ['T'] * (ap_end-ap_beg+1)
                op_tags[op_beg:op_end+1] = ['T'] * (op_end-op_beg+1)
                ap_op_tags[ap_beg:ap_end+1] = ['T-AP'] * (ap_end-ap_beg+1)
                ap_op_tags[op_beg:op_end+1] = ['T-OP'] * (op_end-op_beg+1)
                polarity = self.polarity_map[polarity_str]
                triplet_indices[ap_end, op_end] = polarity
                if (ap_beg, ap_end) not in ap_spans:
                    ap_spans.append((ap_beg, ap_end))
                if (op_beg, op_end) not in op_spans:
                    op_spans.append((op_beg, op_end))
                triplets.append((ap_beg, ap_end, op_beg, op_end, polarity))

            # convert from ot to bio
            ap_tags = to2bio(ap_tags)
            op_tags = to2bio(op_tags)
            ap_op_tags = to2bio(ap_op_tags)

            ap_indices = [self.tag_map[tag] for tag in ap_tags]
            op_indices = [self.tag_map[tag] for tag in op_tags]

            data = {
                'text_indices': text_indices,
                'ap_indices': ap_indices,
                'op_indices': op_indices,
                'triplet_indices': triplet_indices,
                'ap_spans': ap_spans,
                'op_spans': op_spans,
                'triplets': triplets,
            }
            all_data.append(data)
        
        return all_data

class ABSADataReaderV2(ABSADataReader):
    def __init__(self, data_dir):
        super(ABSADataReaderV2, self).__init__(data_dir)

    def _create_dataset(self, set_type, tokenizer):
        all_data = []

        filename = os.path.join(self.data_dir, '%s_triplets.txt' % set_type)
        fp = open(filename, 'r', encoding='utf-8')
        lines = fp.readlines()
        fp.close()

        for i in range(len(lines)):
            text, pairs = lines[i].strip().split('####')

            text_indices = tokenizer.text_to_sequence(text)
            seq_len = len(text_indices)
            ap_tags = ['O'] * seq_len
            op_tags = ['O'] * seq_len
            ap_op_tags = ['O'] * seq_len

            triplet_indices = np.zeros((seq_len, seq_len), dtype=np.int64)
            ap_spans = []
            op_spans = []
            triplets = []
            for pair in eval(pairs):
                ap_beg, ap_end = pair[0][0], pair[0][-1]
                op_beg, op_end = pair[1][0], pair[1][-1]
                polarity_str = pair[2]
                ap_tags[ap_beg:ap_end+1] = ['T'] * (ap_end-ap_beg+1)
                op_tags[op_beg:op_end+1] = ['T'] * (op_end-op_beg+1)
                ap_op_tags[ap_beg:ap_end+1] = ['T-AP'] * (ap_end-ap_beg+1)
                ap_op_tags[op_beg:op_end+1] = ['T-OP'] * (op_end-op_beg+1)
                polarity = self.polarity_map[polarity_str]
                triplet_indices[ap_end, op_end] = polarity
                if (ap_beg, ap_end) not in ap_spans:
                    ap_spans.append((ap_beg, ap_end))
                if (op_beg, op_end) not in op_spans:
                    op_spans.append((op_beg, op_end))
                triplets.append((ap_beg, ap_end, op_beg, op_end, polarity))

            # convert from ot to bio
            ap_tags = to2bio(ap_tags)
            op_tags = to2bio(op_tags)
            ap_op_tags = to2bio(ap_op_tags)

            ap_indices = [self.tag_map[tag] for tag in ap_tags]
            op_indices = [self.tag_map[tag] for tag in op_tags]

            data = {
                'text_indices': text_indices,
                'ap_indices': ap_indices,
                'op_indices': op_indices,
                'triplet_indices': triplet_indices,
                'ap_spans': ap_spans,
                'op_spans': op_spans,
                'triplets': triplets,
            }
            all_data.append(data)
        
        return all_data