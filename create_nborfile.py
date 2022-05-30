from collections import defaultdict
from ogb.linkproppred import LinkPropPredDataset
import pickle


nbors_per_node = 5
neighbor_file = 'data/neighbors+relations_with_max_degrees.pkl'

dataset = LinkPropPredDataset(name='ogbl-wikikg2')
num_nodes = dataset.graph['num_nodes']
num_rels = dataset.graph['edge_reltype'].max() + 1
split_dict = dataset.get_edge_split()
train_triples = split_dict['train']

nbors = defaultdict(set)
edges = defaultdict(set)
degrees = defaultdict(int)
for h,r,t in zip(train_triples['head'], train_triples['relation'], train_triples['tail']):
    nbors[h].add(t)
    nbors[t].add(h)
    degrees[h] += 1
    degrees[t] += 1
    edges[(h,t)] = r

neighbors = {}
for i in range(num_nodes):
    sorted_nbors = sorted(nbors[i], key=lambda x:degrees[x], reverse=True)[:nbors_per_node]
    nbor_rels = [edges[(n,i)] if (n,i) in edges else edges[(i,n)]+num_rels for n in sorted_nbors]
    neighbors[i] = {'nbs': sorted_nbors, 'rels': nbor_rels}
pickle.dump(neighbors, open(neighbor_file, "wb"))
