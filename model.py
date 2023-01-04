from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F

from transfile import TransformerBlock
from collections import defaultdict
import pickle
import os
import re
import time

def no_error_listdir(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return os.listdir(path)

class KGEModel(nn.Module):
    def __init__(self, processor, args):
        super(KGEModel, self).__init__()
        self.model_name = args.model
        self.processor = processor
        self.nentity = processor.nentity
        self.nrelation = processor.nrelation
        self.epsilon = 2.0
        self.u = args.triplere_u

        if self.model_name not in ['TransE', 'DistMult', 'ComplEx', 'RotatE', 'AutoSF', 'PairRE', 'TripleRE']:
            raise ValueError('model %s not supported' % self.model_name)

        if self.model_name == 'RotatE':
            self.entity_dim = args.hidden_dim * 2
            self.relation_dim = args.hidden_dim
        elif self.model_name == 'ComplEx':
            self.entity_dim = args.hidden_dim * 2
            self.relation_dim = args.hidden_dim * 2
        elif self.model_name == 'PairRE':
            self.entity_dim = args.hidden_dim
            self.relation_dim = args.hidden_dim * 2
        elif self.model_name == 'TripleRE':
            self.entity_dim = args.hidden_dim
            self.relation_dim = args.hidden_dim * 3

        self.gamma = nn.Parameter(
            torch.Tensor([args.gamma]),
            requires_grad=False
        )

        self.embedding_range = nn.Parameter(
            torch.Tensor([(self.gamma.item() + self.epsilon) / args.hidden_dim]),
            requires_grad=False
        )

        # self._init_Default()
        # self._encode_func = self._encode_Default

        self._init_StarGraph(args)
        self._encode_func = self._encode_StarGraph

    def _init_StarGraph(self, args):
        anchor_dim = self.entity_dim
        self.sample_anchors = args.sample_anchors
        self.use_anchor_path = args.use_anchor_path and args.sample_anchors > 0

        self.node_itself = args.sample_center
        self.sample_neighbors = args.sample_neighbors

        self.concate = False
        if self.concate:
            self.relation_dim += args.node_dim * (self.node_itself + self.sample_neighbors) \
                * (self.relation_dim // self.entity_dim)

        self.relation_embedding = nn.Embedding(num_embeddings=self.nrelation, embedding_dim=self.relation_dim)
        nn.init.uniform_(
            tensor=self.relation_embedding.weight,
            a=-self.embedding_range.item(),
            b=self.embedding_range.item()
        )

        type_ids = [] # token type embedding, center or neighbor or anchor
        path_ids = [[] for i in range(self.nentity)] # the path (relation type) from each anchor to the target node

        self.node_before_attn = None
        if self.node_itself or self.sample_neighbors > 0:
            print("Creating nodes infomation")
            if self.node_itself:
                if not self.concate:
                    type_ids.append(0)
                nodes = [[i] for i in range(self.nentity)]
            else:
                nodes = [[] for i in range(self.nentity)]
            if self.sample_neighbors > 0:
                if not self.concate:
                    type_ids.extend([1] * self.sample_neighbors)
                nb_vocab = self.tokenize_neighbors(self.processor, self.sample_neighbors)
                # convert dict to list
                for i in range(self.nentity):
                    nb_info = nb_vocab.get(i, [])
                    nodes[i].extend(nb_info[:self.sample_neighbors] + [self.nentity] * (self.sample_neighbors - len(nb_info)))
            self.register_buffer('nodes', torch.tensor(nodes, dtype=torch.long))
            del nodes
            # the extra embedding for padding entity
            self.node_embedding = nn.Embedding(num_embeddings=self.nentity+1, embedding_dim=anchor_dim if args.node_dim == 0 else args.node_dim)
            nn.init.uniform_(
                tensor=self.node_embedding.weight,  # .weight for Embedding
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            if (not self.concate) and args.node_dim != 0 and args.node_dim != anchor_dim:
                self.node_before_attn = nn.Linear(args.node_dim, anchor_dim)

        if self.sample_anchors > 0:
            type_ids.extend([2] * self.sample_anchors)
            print("Creating hashes")
            anchors, vocab = self.tokenize_anchors(self.processor, args.anchor_size, self.sample_anchors, args.anchor_skip_ratio)
            if args.anchor_share_embedding:
                assert args.sample_center or args.sample_neighbors > 0
                assert args.node_dim == 0 or args.node_dim == anchor_dim
                token2id = {i:i for i in range(self.nentity)}
                token2id[-1] = self.nentity
            else:
                anchor_size = len(anchors)
                token2id = {t:i for i,t in enumerate(anchors)}
                token2id[-1] = anchor_size
            hashes = []
            for i in range(self.nentity):
                anclist = [path for path in vocab[i] if len(path) <= 3][:self.sample_anchors]
                anclist.extend([[-1]] * (self.sample_anchors - len(anclist)))
                hashes.append([token2id[path[0]] for path in anclist])
                path_ids[i].extend([[-1]*(3-len(path)) + path[1:] for path in anclist])
            self.register_buffer('hashes', torch.tensor(hashes, dtype=torch.long))
            del hashes
            if args.anchor_share_embedding:
                self.anchor_embedding = self.node_embedding
            else:
                self.anchor_embedding = nn.Embedding(num_embeddings=anchor_size+1, embedding_dim=anchor_dim)
                nn.init.uniform_(
                    tensor=self.anchor_embedding.weight,  # .weight for Embedding
                    a=-self.embedding_range.item(),
                    b=self.embedding_range.item()
                )

        self.attn_layers = None
        if args.attn_layers_num > 0:
            self.register_buffer('type_ids', torch.tensor(type_ids, dtype=torch.long))
            if args.add_type_embedding:
                # 3 types: node itself, neighbor nodes, anchors
                self.type_embedding = nn.Embedding(num_embeddings=3, embedding_dim=anchor_dim)
                nn.init.uniform_(
                    tensor=self.type_embedding.weight,  # .weight for Embedding
                    a=-self.embedding_range.item(),
                    b=self.embedding_range.item()
                )
            else:
                self.type_embedding = None
            self.attn_layers = nn.ModuleList([
                TransformerBlock(in_feat=anchor_dim, mlp_ratio=args.mlp_ratio, num_heads=8, head_dim=args.head_dim, dropout_p=args.drop)
                for _ in range(args.attn_layers_num)
            ])

        if self.use_anchor_path:
            self.register_buffer('path_ids', torch.tensor(path_ids, dtype=torch.long))
            npath = self.path_ids.max() + 1
            self.path_ids[self.path_ids == -1] = npath
            path_dim = anchor_dim * 2 # for node_emb * path_weight + path_bias
            self.path_embeddings = nn.Embedding(num_embeddings=npath+1, embedding_dim=path_dim)
            nn.init.uniform_(
                tensor=self.path_embeddings.weight,
                a=-self.embedding_range.item(),
                b=self.embedding_range.item()
            )
            with torch.no_grad():
                self.path_embeddings.weight[npath] = 0

        del type_ids, path_ids

        self.mean_pooling = False
        self.linear_proj = False
        if args.merge_strategy == 'mean_pooling':
            self.mean_pooling = True
        elif args.merge_strategy == 'linear_proj':
            self.linear_proj = True
            self.set_enc = nn.Sequential(
                nn.Linear(anchor_dim * (self.sample_anchors + (self.sample_neighbors + self.node_itself)*(1-self.concate)), self.entity_dim * args.mlp_ratio), 
                nn.Dropout(args.drop), nn.ReLU(),
                nn.Linear(self.entity_dim * args.mlp_ratio, self.entity_dim)
            )
        else:
            raise TypeError('The merge_strategy available: [ mean_pooling, linear_proj ], now is {}'.format(args.merge_strategy))

        print('node embedding: {}\tsample anchors: {}\tsample neighbors: {}'.format(self.node_itself, self.sample_anchors, self.sample_neighbors))
        print('anchor dim: {}\nentity dim: {}\nrelation dim: {}\nnode dim: {}\nmerge strategy:{}'.format(
            anchor_dim, self.entity_dim, self.relation_dim, args.node_dim, args.merge_strategy))
        print('node layers:\n{}\nattention layers:\n{}'.format(
            self.node_before_attn, 
            self.attn_layers))


    def _encode_StarGraph(self, entities: torch.LongTensor) -> torch.FloatTensor:
        prev_shape = entities.shape
        if len(prev_shape) > 1:
            entities = entities.view(-1)
        embs_seq = None # [entities.shape, sequence_length, feature_dimension]

        if self.sample_anchors > 0:
            hashes = self.hashes[entities]
            anc_embs = self.anchor_embedding(hashes)
            if self.use_anchor_path:
                dim = anc_embs.shape[-1]
                paths = self.path_ids[entities]
                path_embs = self.path_embeddings(paths)
                anc_embs = anc_embs * (path_embs[...,0,:dim] + 1) + path_embs[...,0,dim:]
                anc_embs = anc_embs * (path_embs[...,1,:dim] + 1) + path_embs[...,1,dim:]
            embs_seq = anc_embs #if embs_seq is None else torch.cat([embs_seq, anc_embs], dim=-2)
        
        if (self.node_itself or self.sample_neighbors > 0) and not self.concate:
            nodes = self.nodes[entities]
            node_embs = self.node_embedding(nodes)
            if self.node_before_attn is not None:
                node_embs = self.node_before_attn(node_embs)
            embs_seq = node_embs if embs_seq is None else torch.cat([node_embs, embs_seq], dim=-2)
        
        if self.attn_layers is not None:
            if self.type_embedding is not None:
                embs_seq = embs_seq + self.type_embedding(self.type_ids)
            for enc_layer in self.attn_layers:
                embs_seq = enc_layer(embs_seq)

        if self.mean_pooling:
            embs_seq = embs_seq.mean(dim=-2)
        elif self.linear_proj:
            embs_seq = self.set_enc(embs_seq.view(embs_seq.shape[0], -1))
        else:
            raise NotImplementedError

        if (self.node_itself or self.sample_neighbors > 0) and self.concate:
            nodes = self.nodes[entities]
            node_embs = self.node_embedding(nodes).view(nodes.shape[0], -1)
            embs_seq = torch.cat([node_embs, embs_seq], dim=-1)

        return embs_seq.view(*prev_shape, -1)


    def _init_Default(self):
        self.entity_embedding = nn.Embedding(self.nentity, embedding_dim=self.entity_dim)
        nn.init.uniform_(
            tensor=self.entity_embedding.weight, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )
        
        self.relation_embedding = nn.Embedding(self.nrelation, embedding_dim=self.relation_dim)
        nn.init.uniform_(
            tensor=self.relation_embedding.weight, 
            a=-self.embedding_range.item(), 
            b=self.embedding_range.item()
        )

    def _encode_Default(self, entities):
        return self.entity_embedding(entities)

    def _encode_cache(self, entities):
        return self.entity_embedding[entities]

    def cache_entity_embedding(self, batch_size=1024):
        if 'entity_embedding' in self._modules:
            return
        # calculate the entity embeddings
        entity_embedding = []
        for i in range(0, self.nentity, batch_size):
            entities = torch.arange(
                i, min(i+batch_size, self.nentity), dtype=torch.long, 
                device=self.relation_embedding.weight.device)
            entity_embedding.append(self._encode_func(entities).detach())
        self.register_buffer('entity_embedding', torch.cat(entity_embedding, dim=0))
        self._prev_encode_func = self._encode_func
        self._encode_func = self._encode_cache
        print('cache_entity_embedding')

    def detach_entity_embedding(self):
        if 'entity_embedding' in self._modules:
            return
        assert 'entity_embedding' in self._buffers and hasattr(self, '_prev_encode_func'), \
            'The entity embeddings should be cached before detached.'
        delattr(self, 'entity_embedding')
        self._encode_func = self._prev_encode_func
        print('detach_entity_embedding')

    
    @staticmethod
    def tokenize_anchors(processor, anchor_size=0.1, sample_anchors=20, skip_ratio=0.5):

        def read_previous_file(anchor_size, sample_anchors):
            fpattern = '{}_{}-{}_(\\d+)_anchors.pkl'.format(processor.name, anchor_size, skip_ratio)
            for fname in no_error_listdir('data/'):
                r = re.match(fpattern, fname)
                if r:
                    pre_sample = int(r.group(1))
                    if pre_sample >= sample_anchors:
                        print('Reading exsiting anchor file {}...'.format('data/'+fname), end='')
                        pretime = time.time()
                        anchors, vocab = pickle.load(open('data/'+fname, "rb"))
                        print('{:.2f} seconds taken.'.format(time.time()-pretime))
                        return (anchors, vocab)
            return None
        
        # create anchor file
        nentity = processor.nentity
        anchor_size = int(anchor_size if anchor_size > 1 else nentity*anchor_size)
        assert anchor_size <= nentity
        anchors_vocab = read_previous_file(anchor_size, sample_anchors)
        if anchors_vocab is not None:
            return anchors_vocab

        print('Creating anchor file...', end='')
        pretime = time.time()
        # get graph info
        nbors = defaultdict(set)
        edges = defaultdict(set)
        degrees = defaultdict(int)
        # out_num = defaultdict(int)
        triples = processor.get_original_train_triples()[:,:3]
        nrelation = triples[:,1].max() + 1
        for h,r,t in triples:
            nbors[h].add(t)
            nbors[t].add(h)
            degrees[h] += 1
            degrees[t] += 1
            edges[(h,t)].add(r)
            edges[(t,h)].add(r+nrelation) # inversed relation
            # out_num[(h,r)] += 1
            # out_num[(t,r+nrelation)] += 1
        # keep the edge with the minimal index between head and tail
        uni_edges = {}
        for k,v in edges.items():
            if len(v) == 1:
                uni_edges[k] = v.pop()
                continue
            # uni_edges[k] = sorted(v, key=lambda x: (out_num[(k[0],x)],x))[0]
            uni_edges[k] = sorted(v)[0]
        # choose anchors from all the nodes
        ordered_nodes = sorted(range(nentity), key=lambda x: degrees[x], reverse=True)
        anchors = set()
        for i in ordered_nodes:
            tmp_nbors = nbors[i]
            if len(tmp_nbors & anchors) / len(tmp_nbors) > skip_ratio:
                continue
            anchors.add(i)
            if len(anchors) == anchor_size:
                break
        if len(anchors) != anchor_size:
            print('\n\tset to use {} anchors, but only {} anchors are generated'.format(anchor_size, len(anchors)))
            anchor_size = len(anchors)
            anchors_vocab = read_previous_file(anchor_size, sample_anchors)
            if anchors_vocab is not None:
                return anchors_vocab
        # one-hop neighbor nodes that are anchors
        one_hop_ancs = {}
        for i in range(nentity):
            one_hop_ancs[i] = nbors[i] & anchors

        anc_rank = {i:degrees[i] for i in anchors} # anchors sorted by incresing degrees
        node_rank = {i:len(one_hop_ancs[i]) for i in range(nentity)}

        def get_anclist_uniform(i):
            tmp_ancs = set()
            ancs_list = []
            sorted_nbors = sorted(nbors[i] - one_hop_ancs[i], key=lambda x:node_rank[x])
            while len(tmp_ancs) < sample_anchors:
                ancs_num = len(tmp_ancs)
                self_ancs = one_hop_ancs[i] - tmp_ancs
                if len(self_ancs) > 0:
                    self_ancs = sorted(self_ancs, key=lambda x:anc_rank[x])
                    tmp_ancs.add(self_ancs[0])
                    ancs_list.append([self_ancs[0], uni_edges[(self_ancs[0],i)]])
                for j in sorted_nbors:
                    nbor_ancs = one_hop_ancs[j] - tmp_ancs
                    if len(nbor_ancs) == 0:
                        continue
                    nbor_ancs = sorted(nbor_ancs, key=lambda x:anc_rank[x])
                    tmp_ancs.add(nbor_ancs[0])
                    ancs_list.append([nbor_ancs[0], uni_edges[(nbor_ancs[0],j)], uni_edges[(j,i)]])
                if len(tmp_ancs) == ancs_num:
                    break
            return ancs_list[:sample_anchors]

        vocab = {}
        lens = []
        for i in range(nentity):
            tmp_anc_list = get_anclist_uniform(i)
            lens.append(len(tmp_anc_list))
            if len(tmp_anc_list) == 0:
                vocab[i] = []
            else:
                vocab[i] = tmp_anc_list
        # from collections import Counter
        # print(Counter(lens))
        print('{:.2f} seconds taken.'.format(time.time()-pretime))

        anchor_file = 'data/{}_{}-{}_{}_anchors.pkl'.format(processor.name, anchor_size, skip_ratio, sample_anchors)
        print('Saving to {}...'.format(anchor_file), end='')
        pretime = time.time()
        pickle.dump((anchors, vocab), open(anchor_file, "wb"))
        print('{:.2f} seconds taken.'.format(time.time()-pretime))
        return anchors, vocab
        

    @staticmethod
    def tokenize_neighbors(processor, sample_neighbors):

        # read exsiting neighbor file
        fpattern = '{}_(\\d+)_neighbors.pkl'.format(processor.name)
        for fname in no_error_listdir('data/'):
            r = re.match(fpattern, fname)
            if r:
                pre_sample = int(r.group(1))
                if pre_sample >= sample_neighbors:
                    print('Reading exsiting neighbor file {}...'.format('data/'+fname), end='')
                    pretime = time.time()
                    vocab = pickle.load(open('data/'+fname, "rb"))
                    print('{:.2f} seconds taken.'.format(time.time()-pretime))
                    return vocab
        
        print('Creating neighbor file...', end='')
        pretime = time.time()
        # sample neighbors with max degrees     
        nbors = defaultdict(set)
        degrees = defaultdict(int)
        triples = processor.get_original_train_triples()[:,:3]
        for h,r,t in triples:
            nbors[h].add(t)
            nbors[t].add(h)
            degrees[h] += 1
            degrees[t] += 1

        vocab = {}
        for i in range(processor.nentity):
            vocab[i] = sorted(nbors[i], key=lambda x:degrees[x], reverse=True)[:sample_neighbors]
            # nbor_rels = [edges[(n,i)] if (n,i) in edges else edges[(i,n)]+nrelation for n in vocab[i]]
        print('{:.2f} seconds taken.'.format(time.time()-pretime))

        neighbor_file = 'data/{}_{}_neighbors.pkl'.format(processor.name, sample_neighbors)
        print('Saving to {}...'.format(neighbor_file), end='')
        pretime = time.time()
        pickle.dump(vocab, open(neighbor_file, "wb"))
        print('{:.2f} seconds taken.'.format(time.time()-pretime))
        return vocab


    def forward(self, head, relation, tail):
        head = self._encode_func(head)
        relation = self.relation_embedding(relation)
        tail = self._encode_func(tail)

        model_func = { # 'mode' is deprecated in this version of codes
            # 'TransE': self.TransE,
            # 'DistMult': self.DistMult,
            # 'ComplEx': self.ComplEx,
            # 'RotatE': self.RotatE,
            # 'AutoSF': self.AutoSF,
            'PairRE': self.PairRE,
            'TripleRE': self.TripleRE,
        }

        if self.model_name in model_func:
            score = model_func[self.model_name](head, relation, tail)
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

    def PairRE(self, head, relation, tail):
        re_head, re_tail = torch.chunk(relation, 2, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * re_head - tail * re_tail + head - tail
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def TripleRE(self, head, relation, tail):
        re_head, re_mid, re_tail = torch.chunk(relation, 3, dim=2)

        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)

        score = head * (self.u * re_head + 1) - tail * (self.u * re_tail + 1) + re_mid
        score = self.gamma.item() - torch.norm(score, p=1, dim=2)
        return score

    def TransE(self, head, relation, tail, mode):
        head = F.normalize(head, 2, -1)
        tail = F.normalize(tail, 2, -1)
        if mode == 'head-batch':
            score = head + (relation - tail)
        else:
            score = head + relation - tail

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
