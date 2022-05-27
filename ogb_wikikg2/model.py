from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import numpy as np
from pandas import merge_asof

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from torch.utils.data import DataLoader
from ogb_wikikg2.dataloader import TestDataset
from ogb_wikikg2.transfile import TransformerBlock
from collections import defaultdict
from typing import Optional
from tqdm import tqdm
import math
from torch.nn import TransformerEncoderLayer, TransformerEncoder, GRU
import pickle

from ogb.linkproppred import Evaluator


class KGEModel(nn.Module):
    def __init__(self, model_name, nentity, nrelation, hidden_dim, gamma, evaluator,
                 drop, triples):
        super(KGEModel, self).__init__()
        self.model_name = model_name
        self.nentity = nentity
        self.nrelation = nrelation
        self.hidden_dim = hidden_dim
        self.epsilon = 2.0
        self.u = 0.1

        self.drop = drop
        self.triples = triples

        if model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'AutoSF', 'PairRE', 'TripleRE']:
            raise ValueError('model %s not supported' % model_name)

        if model_name == 'RotatE':
            self.entity_dim = hidden_dim * 2
            self.relation_dim = hidden_dim
        elif model_name == 'ComplEx':
            self.entity_dim = hidden_dim * 2
            self.relation_dim = hidden_dim * 2
        elif model_name == 'PairRE':
            self.entity_dim = hidden_dim
            self.relation_dim = hidden_dim * 2
        elif model_name == 'TripleRE':
            self.entity_dim = hidden_dim
            self.relation_dim = hidden_dim * 3

        self.gamma = nn.Parameter(
            torch.Tensor([gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / hidden_dim]),
            requires_grad=False
        )

        anchor_file = 'data/ogbl-wikikg2_20000_anchors_20000_paths_d0.4_b0.0_p0.4_r0.2_pykeen_50sp_bfs.pkl'
        neighbor_file = 'data/neighbors+relations_with_max_degrees.pkl'

        anchor_dim = self.entity_dim
        self.sample_anchors = 20

        node_dim = 32
        self.node_itself = True

        self.sample_neighbors = 5

        self.attn_layers_num = 1
        self.attn_dim = self.entity_dim
        self.attn_heads = 8

        merge_strategy = 'mean_pooling'
        self.mean_pooling = False
        self.linear_proj = False
        if merge_strategy == 'mean_pooling':
            self.mean_pooling = True
        elif merge_strategy == 'linear_proj':
            self.linear_proj = True
        else:
            raise TypeError
        
        mlp_ratio = 4
        
        # 3 types: node itself, neighbor nodes, anchors
        self.type_embedding = nn.Embedding(num_embeddings=3, embedding_dim=self.attn_dim)
        nn.init.uniform_(
            tensor=self.type_embedding.weight,  # .weight for Embedding
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )
        # back to normal relation embs, +1 for the padding relation
        self.relation_embedding = nn.Embedding(num_embeddings=nrelation, embedding_dim=self.relation_dim)
        nn.init.uniform_(
            tensor=self.relation_embedding.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        if self.linear_proj:
            self.set_enc = nn.Sequential(
                nn.Linear(self.attn_dim * (self.sample_anchors + self.sample_neighbors + self.node_itself), self.entity_dim * mlp_ratio), 
                nn.Dropout(self.drop), nn.ReLU(),
                nn.Linear(self.entity_dim * mlp_ratio, self.entity_dim)
            )

        type_ids = []

        self.node_before_attn = None
        if self.node_itself or self.sample_neighbors > 0:
            print("Creating nodes infomation")
            if self.node_itself:
                type_ids.append(0)
                nodes = [[i] for i in range(nentity)]
            else:
                nodes = [[] for i in range(nentity)]
            if self.sample_neighbors > 0:
                type_ids.extend([1] * self.sample_neighbors)
                nb_vocab = pickle.load(open(neighbor_file, 'rb'))
                # convert dict to list
                for i in range(nentity):
                    nb_info = nb_vocab.get(i, {'nbs':[], 'rels':[]})
                    nodes[i].extend([n for n in nb_info['nbs'][:self.sample_neighbors]])
                    if len(nodes[i]) < self.sample_neighbors + self.node_itself:
                        nodes[i].extend([nentity for _ in range(self.sample_neighbors + self.node_itself - len(nodes[i]))])
            self.register_buffer('nodes', torch.tensor(nodes, dtype=torch.long))
            del nodes

            self.node_embeddings = nn.Embedding(num_embeddings=nentity+1, embedding_dim=self.attn_dim if node_dim == 0 else node_dim)
            nn.init.uniform_(
                tensor=self.node_embeddings.weight,  # .weight for Embedding
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            self.node_before_attn = None if node_dim == 0 else nn.Linear(node_dim, self.attn_dim)

        if self.sample_anchors > 0:
            type_ids.extend([2] * self.sample_anchors)
            print("Creating hashes")
            anchors, _, vocab = pickle.load(open(anchor_file, "rb"))
            anchors.append(-99)
            nanchor = len(anchors)
            token2id = {t:i for i,t in enumerate(anchors)}
            hashes = [
                [token2id[i] for i in vocab[e]['ancs'][:self.sample_anchors]] + [nanchor]*(self.sample_anchors - len(vocab[e]['ancs']))
                for e in range(nentity)
            ]
            self.register_buffer('hashes', torch.tensor(hashes, dtype=torch.long))
            del hashes

            self.anchor_embeddings = nn.Embedding(num_embeddings=nanchor+1, embedding_dim=anchor_dim)
            nn.init.uniform_(
                tensor=self.anchor_embeddings.weight,  # .weight for Embedding
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )

        self.register_buffer('type_ids', torch.tensor(type_ids, dtype=torch.long))

        self.attn_layers = nn.ModuleList([
            TransformerBlock(in_feat=self.attn_dim, mlp_ratio=mlp_ratio, num_heads=self.attn_heads, dropout_p=self.drop)
            for _ in range(self.attn_layers_num)
        ])

        print('node embedding: {}\tsample anchors: {}\tsample neighbors: {}'.format(self.node_itself, self.sample_anchors, self.sample_neighbors))
        print('anchor dim: {}\nentity dim: {}\nrelation dim: {}\nnode dim: {}\nmerge strategy:{}'.format(
            anchor_dim, self.entity_dim, self.relation_dim, node_dim, merge_strategy))
        print('attention dim: {}, attention heads: {}\n'.format(self.attn_dim, self.attn_heads))
        print('node layers:\n{}\nattention layers:\n{}'.format(
            self.node_before_attn, 
            self.attn_layers))
    
        self.evaluator = evaluator



    def encode_by_index(self, entities: torch.LongTensor) -> torch.FloatTensor:
        assert len(entities.shape) == 1
        embs_seq = None # [entities.shape, sequence_length, feature_dimension]
        if self.sample_anchors > 0:
            hashes = self.hashes[entities]
            anc_embs = self.anchor_embeddings(hashes)
            embs_seq = anc_embs #if embs_seq is None else torch.cat([embs_seq, anc_embs], dim=-2)
        
        if self.node_itself or self.sample_neighbors > 0:
            nodes = self.nodes[entities]
            node_embs = self.node_embeddings(nodes)
            if self.node_before_attn is not None:
                node_embs = self.node_before_attn(node_embs)
            embs_seq = node_embs if embs_seq is None else torch.cat([node_embs, embs_seq], dim=-2)

        if self.attn_layers_num > 0:
            embs_seq = embs_seq + self.type_embedding(self.type_ids)
            for enc_layer in self.attn_layers:
                embs_seq = enc_layer(embs_seq)

        if self.mean_pooling:
            embs_seq = embs_seq.mean(dim=-2)
        elif self.linear_proj:
            embs_seq = self.set_enc(embs_seq.view(*entities.shape, -1))
        else:
            raise NotImplementedError

        return embs_seq


    def forward(self, sample, mode='single'):
        '''
        Forward function that calculate the score of a batch of triples.
        In the 'single' mode, sample is a batch of triple.
        In the 'head-batch' or 'tail-batch' mode, sample consists two part.
        The first part is usually the positive sample.
        And the second part is the entities in the negative samples.
        Because negative samples and positive samples usually share two elements
        in their triple ((head, relation) or (relation, tail)).
        '''

        if mode == 'single':
            batch_size, negative_sample_size = sample.size(0), 1
            head = self.encode_by_index(sample[:, 0]).unsqueeze(1)
            relation = self.relation_embedding(sample[:, 1]).unsqueeze(1)
            tail = self.encode_by_index(sample[:, 2]).unsqueeze(1)

        elif mode == 'head-batch':
            tail_part, head_part = sample
            batch_size, negative_sample_size = head_part.size(0), head_part.size(1)
            head = self.encode_by_index(head_part.view(-1)).view(batch_size, negative_sample_size, -1)
            relation = self.relation_embedding(tail_part[:, 1]).unsqueeze(1)
            tail = self.encode_by_index(tail_part[:, 2]).unsqueeze(1)

        elif mode == 'tail-batch':
            head_part, tail_part = sample
            batch_size, negative_sample_size = tail_part.size(0), tail_part.size(1)
            head = self.encode_by_index(head_part[:, 0]).unsqueeze(1)
            relation = self.relation_embedding(head_part[:, 1]).unsqueeze(1)
            tail = self.encode_by_index(tail_part.view(-1)).view(batch_size, negative_sample_size, -1)

        else:
            raise ValueError('mode %s not supported' % mode)

        model_func = {
            'TransE': self.TransE,
            'DistMult': self.DistMult,
            'ComplEx': self.ComplEx,
            'RotatE': self.RotatE,
            'AutoSF': self.AutoSF,
            'PairRE': self.PairRE,
            'TripleRE': self.TripleRE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail, mode)
        else:
            raise ValueError('model %s not supported' % self.model_name)

        return score

    def AutoSF(self, head, relation, tail, mode):

        if mode == 'head-batch':
            rs = torch.chunk(relation, 4, dim=-1)
            ts = torch.chunk(tail, 4, dim=-1)
            rt0 = rs[0] * ts[0]
            rt1 = rs[1] * ts[1] + rs[2] * ts[3]
            rt2 = rs[0] * ts[2] + rs[2] * ts[3]
            rt3 = -rs[1] * ts[1] + rs[3] * ts[2]
            rts = torch.cat([rt0, rt1, rt2, rt3], dim=-1)
            score = torch.sum(head * rts, dim=-1)

        else:
            hs = torch.chunk(head, 4, dim=-1)
            rs = torch.chunk(relation, 4, dim=-1)
            hr0 = hs[0] * rs[0]
            hr1 = hs[1] * rs[1] - hs[3] * rs[1]
            hr2 = hs[2] * rs[0] + hs[3] * rs[3]
            hr3 = hs[1] * rs[2] + hs[2] * rs[2]
            hrs = torch.cat([hr0, hr1, hr2, hr3], dim=-1)
            score = torch.sum(hrs * tail, dim=-1)

        return score

    def PairRE(self, head, relation, tail, mode):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail + head - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def TripleRE(self, head, relation, tail, mode):
        re_head, re_mid, re_tail = torch.chunk(relation, 3, dim=2)

        e_h = torch.ones_like(head)
        e_t = torch.ones_like(tail)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * (self.u * re_head + e_h) - tail * (self.u * re_tail + e_t) + re_mid
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def TransE(self, head, relation, tail, mode):
        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = (head + relation) - tail

        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def DistMult(self, head, relation, tail, mode):
        if mode == 'head-batch':
            score = head * (relation * tail)
        else:
            score = (head * relation) * tail

        score = score.sum(dim=2)
        return score

    def ComplEx(self, head, relation, tail, mode):
        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_relation, im_relation = torch.chunk(relation, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            score = re_head * re_score + im_head * im_score
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            score = re_score * re_tail + im_score * im_tail

        score = score.sum(dim=2)
        return score

    def RotatE(self, head, relation, tail, mode):
        pi = 3.14159265358979323846

        re_head, im_head = torch.chunk(head, 2, dim=2)
        re_tail, im_tail = torch.chunk(tail, 2, dim=2)

        # Make phases of relations uniformly distributed in [-pi, pi]

        phase_relation = relation / (self.embedding_range.item() / pi)

        re_relation = torch.cos(phase_relation)
        im_relation = torch.sin(phase_relation)

        if mode == 'head-batch':
            re_score = re_relation * re_tail + im_relation * im_tail
            im_score = re_relation * im_tail - im_relation * re_tail
            re_score = re_score - re_head
            im_score = im_score - im_head
        else:
            re_score = re_head * re_relation - im_head * im_relation
            im_score = re_head * im_relation + im_head * re_relation
            re_score = re_score - re_tail
            im_score = im_score - im_tail

        score = torch.stack([re_score, im_score], dim=0)
        score = score.norm(dim=0)

        score = self.gamma.item() - score.sum(dim=2)
        return score

    @staticmethod
    def train_step(model, optimizer, train_iterator, args):
        '''
        A single train step. Apply back-propation and return the loss
        '''

        model.train()
        optimizer.zero_grad()
        positive_sample, negative_sample, subsampling_weight, mode = next(train_iterator)

        if args.cuda:
            positive_sample = positive_sample.cuda()
            negative_sample = negative_sample.cuda()
            subsampling_weight = subsampling_weight.cuda()

        negative_score = model((positive_sample, negative_sample), mode=mode)
        if args.negative_adversarial_sampling:
            # In self-adversarial sampling, we do not apply back-propagation on the sampling weight
            negative_score = (F.softmax(negative_score * args.adversarial_temperature, dim=1).detach()
                              * F.logsigmoid(-negative_score)).sum(dim=1)
        else:
            negative_score = F.logsigmoid(-negative_score).mean(dim=1)

        positive_score = model(positive_sample)
        positive_score = F.logsigmoid(positive_score).squeeze(dim=1)
        if args.uni_weight:
            positive_sample_loss = - positive_score.mean()
            negative_sample_loss = - negative_score.mean()
        else:
            positive_sample_loss = - (subsampling_weight * positive_score).sum() / subsampling_weight.sum()
            negative_sample_loss = - (subsampling_weight * negative_score).sum() / subsampling_weight.sum()

        loss = (positive_sample_loss + negative_sample_loss) / 2

        if args.regularization != 0.0:
            # Use L3 regularization for ComplEx and DistMult
            regularization = args.regularization * (
                    model.entity_embedding.norm(p=3) ** 3 +
                    model.relation_embedding.norm(p=3).norm(p=3) ** 3
            )
            loss = loss + regularization
            regularization_log = {'regularization': regularization.item()}
        else:
            regularization_log = {}

        loss.backward()

        optimizer.step()

        log = {
            **regularization_log,
            'positive_sample_loss': positive_sample_loss.item(),
            'negative_sample_loss': negative_sample_loss.item(),
            'loss': loss.item()
        }

        return log

    @staticmethod
    def test_step(model, test_triples, args, random_sampling=False):
        '''
        Evaluate the model on test or valid datasets
        '''

        model.eval()

        # Prepare dataloader for evaluation
        test_dataloader_head = DataLoader(
            TestDataset(
                test_triples,
                args,
                'head-batch',
                random_sampling
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataloader_tail = DataLoader(
            TestDataset(
                test_triples,
                args,
                'tail-batch',
                random_sampling
            ),
            batch_size=args.test_batch_size,
            num_workers=max(1, args.cpu_num // 2),
            collate_fn=TestDataset.collate_fn
        )

        test_dataset_list = [test_dataloader_head, test_dataloader_tail]

        test_logs = defaultdict(list)

        step = 0
        total_steps = sum([len(dataset) for dataset in test_dataset_list])

        with torch.no_grad():
            for test_dataset in test_dataset_list:
                for positive_sample, negative_sample, mode in test_dataset:
                    if args.cuda:
                        positive_sample = positive_sample.cuda()
                        negative_sample = negative_sample.cuda()

                    score = model((positive_sample, negative_sample), mode)

                    batch_results = model.evaluator.eval({'y_pred_pos': score[:, 0],
                                                          'y_pred_neg': score[:, 1:]})
                    for metric in batch_results:
                        test_logs[metric].append(batch_results[metric])

                    if step % args.test_log_steps == 0:
                        logging.info('Evaluating the model... (%d/%d)' % (step, total_steps))

                    step += 1

            metrics = {}
            for metric in test_logs:
                metrics[metric] = torch.cat(test_logs[metric]).mean().item()

        return metrics
