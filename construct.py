from scipy.sparse import csr_matrix
from params import args
import numpy as np
import pickle
import copy
import os

def construct_graphs(seq, num_items, distance, prefix):
    user = list()
    r, c, d = list(), list(), list()
    for i, seq in enumerate(seqs):
        print(f"Processing {i}/{len(seqs)} (>﹏<)    ", end='\r')
        for dist in range(1, distance + 1):
            if dist >= len(seq): break;
            r += copy.deepcopy(seq[+dist:])
            c += copy.deepcopy(seq[:-dist])
            r += copy.deepcopy(seq[:-dist])
            c += copy.deepcopy(seq[+dist:])
    d = np.ones_like(r)
    iigraph = csr_matrix((d, (r, c)), shape=(num_items, num_items))
    print('Constructed i-i graph, density=%.6f' % (len(d) / (num_items ** 2)))
    with open(prefix + 'trn', 'wb') as fs:
        pickle.dump(iigraph, fs)

if __name__ == '__main__':

    # dataset = input('Choose a dataset: ')
    dataset = args.data
    prefix = './datasets/' + dataset + '/'

    # distance  = int(input('Max distance of edge: '))
    distance = args.ii_dis

    with open(prefix + 'seq', 'rb') as fs:
        seqs = pickle.load(fs)

    if dataset == ('books'):
        num_items = 54756
    elif dataset == ('toys'):
        num_items = 54784
    elif dataset == ('retailrocket'):
        num_items = 43886

    construct_graphs(seqs, num_items, distance, prefix)
