from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

from torch.utils.data import Dataset
from collections import defaultdict


class TrainDataset(Dataset):
    def __init__(self, triples, nentity, negative_sample_size, mode, filter_negative=False, weights=None, type_offset=None):
        assert mode in ['head-batch', 'tail-batch']
        if weights is not None:
            assert len(triples) == len(weights)
        self.triples = triples
        self.nentity = nentity
        self.negative_sample_size = negative_sample_size
        self.mode = mode
        self.weights = torch.ones(len(self.triples), dtype=torch.float) \
            if weights is None else torch.tensor(weights, dtype=torch.float)
        if filter_negative:
            true_triples = defaultdict(list)
            for h,r,t in triples[:,:3]:
                true_triples[(h,r)].append(t)
                true_triples[(t,-r-1)].append(h)
            self.true_triples = {k:torch.tensor(v) for k,v in true_triples.items()}
        else:
            self.true_triples = None
        if type_offset is not None:
            assert triples.shape[-1] == 5
        self.type_offset = type_offset
            

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triplet = self.triples[idx]
        head, relation, tail = triplet[:3]
        if self.mode == 'head-batch': # sample random entities as negative heads
            e1, e2, r = tail, head, -relation-1
        elif self.mode == 'tail-batch':
            e1, e2, r = head, tail, relation

        if self.type_offset is None:
            lower_bound, upper_bound = 0, self.nentity
        else:
            lower_bound, upper_bound = self.type_offset[triplet[3 if self.mode == 'head-batch' else 4]]
        negative_sample = [torch.tensor([e2], dtype=torch.long)]
        if self.true_triples is None:
            negative_sample.append(torch.randint(lower_bound, upper_bound, (self.negative_sample_size,)))
        else:
            tmp_size = 0
            while tmp_size < self.negative_sample_size:
                tmp_sample = torch.randint(lower_bound, upper_bound, (self.negative_sample_size*2,))
                mask = torch.isin(tmp_sample, self.true_triples[(e1,r)], assume_unique=True, invert=True)
                tmp_sample = tmp_sample[mask]
                tmp_size += tmp_sample.shape[0]
                negative_sample.append(tmp_sample)
        negative_sample = torch.cat(negative_sample)[:self.negative_sample_size+1]
        
        if self.mode == 'head-batch':
            head = negative_sample
            tail = torch.tensor([tail], dtype=torch.long)
        elif self.mode == 'tail-batch':
            tail = negative_sample
            head = torch.tensor([head], dtype=torch.long)
        relation = torch.tensor([relation], dtype=torch.long)
        weight = self.weights[idx]

        return head, relation, tail, weight


class TestDataset(Dataset):
    def __init__(self, triples, nentity, mode, negative_samples=None, negative_sample_size=0):
        assert mode in ['head-batch', 'tail-batch']
        if negative_samples is not None:
            assert negative_sample_size == 0
            self.neg_mode = 'given'
            self.negative_samples = negative_samples
        elif negative_sample_size != 0:
            self.neg_mode = 'rand'
            self.negative_sample_size = negative_sample_size
        else:
            self.neg_mode = 'all'
        self.triples = triples
        self.nentity = nentity
        self.mode = mode

    def __len__(self):
        return len(self.triples)

    def __getitem__(self, idx):
        triplet = self.triples[idx]
        head, relation, tail = triplet[:3]
        if self.neg_mode == 'given':
            negative_sample = torch.cat([
                torch.tensor([0], dtype=torch.long), 
                torch.from_numpy(self.negative_samples[idx])])
        elif self.neg_mode == 'rand':
            negative_sample = torch.randint(0, self.nentity, (self.negative_sample_size+1,))
        elif self.neg_mode == 'all':
            negative_sample = torch.arange(-1, self.nentity, dtype=torch.long)

        if self.mode == 'head-batch':
            negative_sample[0] = head
            head = negative_sample
            tail = torch.tensor([tail], dtype=torch.long)
        elif self.mode == 'tail-batch':
            negative_sample[0] = tail
            tail = negative_sample
            head = torch.tensor([head], dtype=torch.long)
        relation = torch.tensor([relation], dtype=torch.long)
        return head, relation, tail


class BidirectionalOneShotIterator(object):
    def __init__(self, dataloader_head, dataloader_tail):
        self.iterator_head = self.one_shot_iterator(dataloader_head)
        self.iterator_tail = self.one_shot_iterator(dataloader_tail)
        self.step = 0

    def __next__(self):
        self.step += 1
        if self.step % 2 == 0:
            data = next(self.iterator_head)
        else:
            data = next(self.iterator_tail)
        return data

    @staticmethod
    def one_shot_iterator(dataloader):
        '''
        Transform a PyTorch Dataloader into python iterator
        '''
        while True:
            for data in dataloader:
                yield data
