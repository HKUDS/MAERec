import pickle
import copy
import numpy as np
from scipy.sparse import csr_matrix, coo_matrix, dok_matrix
from params import args
import scipy.sparse as sp
from logger import log
import torch as t
import torch.utils.data as data
import torch.utils.data as dataloader

def neg_sample(item_set, item_size):
    item = np.random.randint(1, item_size - 1)
    while item in item_set:
        item = np.random.randint(1, item_size - 1)
    return item

class DataHandler:
    def __init__(self):
        predir = 'datasets/' + args.data + '/'
        self.trn_file = predir + 'trn'
        self.seq_file = predir + 'seq'
        self.tst_file = predir + 'tst'
        self.neg_file = predir + 'neg'

    def load_adj(self, filename):
        with open(filename, 'rb') as fs:
            ret = (pickle.load(fs) != 0).astype(np.float32)
        if type(ret) != coo_matrix:
            ret = sp.coo_matrix(ret)
        return ret

    def load_seq(self, filename):
        with open(filename, 'rb') as fs:
            ret = pickle.load(fs)
        return ret

    def normalize(self, mat):
        degree = np.array(mat.sum(axis=-1))
        dInvSqrt = np.reshape(np.power(degree, -0.5), [-1])
        dInvSqrt[np.isinf(dInvSqrt)] = 0.0
        dInvSqrtMat = sp.diags(dInvSqrt)
        return mat.dot(dInvSqrtMat).transpose().dot(dInvSqrtMat).tocoo()

    def make_torch_adj(self, mat):
        mat = (mat + sp.eye(mat.shape[0]))
        mat = (mat != 0) * 1.0
        mat = self.normalize(mat)
        idxs = t.from_numpy(np.vstack([mat.row, mat.col]).astype(np.int64))
        vals = t.from_numpy(mat.data.astype(np.float32))
        shape = t.Size(mat.shape)
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def make_all_one_adj(self, adj):
        idxs = adj._indices()
        vals = t.ones_like(adj._values())
        shape = adj.shape
        return t.sparse.FloatTensor(idxs, vals, shape).cuda()

    def load_data(self):
        trn = self.load_adj(self.trn_file)
        tst = self.load_seq(self.tst_file)
        seq = self.load_seq(self.seq_file)
        neg = self.load_seq(self.neg_file)
        neg = np.array(neg)
        args.user, args.item = len(seq), trn.shape[0]
        self.ii_dok = trn.todok()
        self.ii_adj = self.make_torch_adj(trn)
        self.ii_adj_all_one = self.make_all_one_adj(self.ii_adj)
        trn_data = SeqRecTrnDataset(seq)
        tst_data = SeqRecTstDataset(tst, neg)
        self.trn_loader = dataloader.DataLoader(trn_data, batch_size=args.trn_batch, shuffle=True, num_workers=0)
        self.tst_loader = dataloader.DataLoader(tst_data, batch_size=args.tst_batch, shuffle=False, num_workers=0)

class SeqRecTrnDataset(data.Dataset):
    def __init__(self, user_seq):
        self.user_seq = user_seq
        self.max_len = args.max_seq_len

    def sample_data(self, items, input_ids, target_pos):
        copied_input_ids = copy.deepcopy(input_ids)
        target_neg = []
        seq_set = set(items)
        for i in range(len(input_ids)):
            target_neg.append(neg_sample(seq_set, args.item))

        pad_len = self.max_len - len(input_ids)
        input_ids  = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids  = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids)  == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        cur_rec_tensors = (
            t.tensor(input_ids, dtype=t.long),
            t.tensor(target_pos, dtype=t.long),
            t.tensor(target_neg, dtype=t.long),
        )

        return cur_rec_tensors

    def __getitem__(self, index):
        items = self.user_seq[index]
        input_ids = items[:-1]
        target_pos = items[1:]
        return self.sample_data(items, input_ids, target_pos)

    def __len__(self):
        return len(self.user_seq)

class SeqRecTstDataset(data.Dataset):
    def __init__(self, user_seq, neg_samples):
        self.user_seq = user_seq
        self.user_neg = neg_samples
        self.max_len = args.max_seq_len

    def sample_data(self, user_id, items, input_ids, answer):
        copied_input_ids = copy.deepcopy(input_ids)

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:]

        assert len(input_ids) == self.max_len

        neg = self.user_neg[user_id]
        cur_rec_tensors = (
            t.tensor(input_ids, dtype=t.long),
            t.tensor(answer, dtype=t.long),
            t.tensor(neg, dtype=t.long)
        )

        return cur_rec_tensors

    def __getitem__(self, index):
        user_id = index
        items = self.user_seq[index]
        input_ids = items[:-1]
        answer = [items[-1]]
        return self.sample_data(user_id, items, input_ids, answer)

    def __len__(self):
        return len(self.user_seq)

class TrnData(data.Dataset):
    def __init__(self, coomat):
        self.rows = coomat.row
        self.cols = coomat.col
        self.dokmat = coomat.todok()
        self.negs = np.zeros(len(self.rows)).astype(np.int32)

    def negSampling(self):
        for i in range(len(self.rows)):
            u = self.rows[i]
            while True:
                iNeg = np.random.randint(1, args.item)
                if (u, iNeg) not in self.dokmat:
                    break
            self.negs[i] = iNeg

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return self.rows[idx], self.cols[idx], self.negs[idx]

class TstData(data.Dataset):
    def __init__(self, coomat, trnMat):
        self.csrmat = (trnMat.tocsr() != 0) * 1.0

        tstLocs = [None] * coomat.shape[0]
        tstUsrs = set()
        for i in range(len(coomat.data)):
            row = coomat.row[i]
            col = coomat.col[i]
            if tstLocs[row] is None:
                tstLocs[row] = list()
            tstLocs[row].append(col)
            tstUsrs.add(row)
        tstUsrs = np.array(list(tstUsrs))
        self.tstUsrs = tstUsrs
        self.tstLocs = tstLocs

    def __len__(self):
        return len(self.tstUsrs)

    def __getitem__(self, idx):
        return self.tstUsrs[idx], np.reshape(self.csrmat[self.tstUsrs[idx]].toarray(), [-1])
