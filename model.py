import torch.nn.functional as F
from statistics import mean
from params import args
from torch import nn
import numpy as np
import torch as t
import math
import time

init = nn.init.xavier_uniform_
uniform_init = nn.init.uniform

def sparse_dropout(x, keep_prob):
    msk = (t.rand(x._values().size()) + keep_prob).floor().type(t.bool)
    idx = x._indices()[:, msk]
    val = x._values()[msk]
    return t.sparse.FloatTensor(idx, val, x.shape).cuda()

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.item_emb = nn.Parameter(init(t.empty(args.item, args.latdim))) # args.item = num_real_item + 1
        self.gcn_layers = nn.Sequential(*[GCNLayer() for i in range(args.num_gcn_layers)])

    def get_ego_embeds(self):
        return self.item_emb

    def forward(self, encoder_adj):
        embeds = [self.item_emb]
        for i, gcn in enumerate(self.gcn_layers):
            embeds.append(gcn(encoder_adj, embeds[-1]))
        return sum(embeds), embeds

class TrivialDecoder(nn.Module):
    def __init__(self):
        super(TrivialDecoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(args.latdim * 3, args.latdim, bias=True),
            nn.ReLU(),
            nn.Linear(args.latdim, 1, bias=True),
            nn.Sigmoid()
        )
        self.apply(self.init_weights)

    def forward(self, embeds, pos, neg):
        # pos: (batch, 2), neg: (batch, num_reco_neg, 2)
        pos_emb, neg_emb = [], []
        pos_emb.append(embeds[-1][pos[:,0]])
        pos_emb.append(embeds[-1][pos[:,1]])
        pos_emb.append(embeds[-1][pos[:,0]] * embeds[-1][pos[:,1]])
        neg_emb.append(embeds[-1][neg[:,:,0]])
        neg_emb.append(embeds[-1][neg[:,:,1]])
        neg_emb.append(embeds[-1][neg[:,:,0]] * embeds[-1][neg[:,:,1]])
        pos_emb = t.cat(pos_emb, -1) # (n, latdim * 3)
        neg_emb = t.cat(neg_emb, -1) # (n, num_reco_neg, latdim * 3)
        pos_scr = t.exp(t.squeeze(self.MLP(pos_emb))) # (n)
        neg_scr = t.exp(t.squeeze(self.MLP(neg_emb))) # (n, num_reco_neg)
        neg_scr = t.sum(neg_scr, -1) + pos_scr
        loss = -t.sum(pos_scr / (neg_scr + 1e-8) + 1e-8)
        return loss

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(args.latdim * args.num_gcn_layers ** 2, args.latdim * args.num_gcn_layers, bias=True),
            nn.ReLU(),
            nn.Linear(args.latdim * args.num_gcn_layers, args.latdim, bias=True),
            nn.ReLU(),
            nn.Linear(args.latdim, 1, bias=True),
            nn.Sigmoid()
        )
        self.apply(self.init_weights)

    def forward(self, embeds, pos, neg):
        # pos: (batch, 2), neg: (batch, num_reco_neg, 2)
        pos_emb, neg_emb = [], []
        for i in range(args.num_gcn_layers):
            for j in range(args.num_gcn_layers):
                pos_emb.append(embeds[i][pos[:,0]] * embeds[j][pos[:,1]])
                neg_emb.append(embeds[i][neg[:,:,0]] * embeds[j][neg[:,:,1]])
        pos_emb = t.cat(pos_emb, -1) # (n, latdim * num_gcn_layers ** 2)
        neg_emb = t.cat(neg_emb, -1) # (n, num_reco_neg, latdim * num_gcn_layers ** 2)
        pos_scr = t.exp(t.squeeze(self.MLP(pos_emb))) # (n)
        neg_scr = t.exp(t.squeeze(self.MLP(neg_emb))) # (n, num_reco_neg)
        neg_scr = t.sum(neg_scr, -1) + pos_scr
        loss = -t.sum(pos_scr / (neg_scr + 1e-8) + 1e-8)
        return loss

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()

    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)

class SASRec(nn.Module):
    def __init__(self):
        super(SASRec, self).__init__()
        self.pos_emb = nn.Parameter(init(t.empty(args.max_seq_len, args.latdim)))
        self.layers = nn.Sequential(*[TransformerLayer() for i in range(args.num_trm_layers)])
        self.LayerNorm = nn.LayerNorm(args.latdim)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.apply(self.init_weights)

    def get_seq_emb(self, sequence, item_emb):
        seq_len = sequence.size(1)
        pos_ids = t.arange(seq_len, dtype=t.long, device=sequence.device)
        pos_ids = pos_ids.unsqueeze(0).expand_as(sequence)
        itm_emb = item_emb[sequence]
        pos_emb = self.pos_emb[pos_ids]
        seq_emb = itm_emb + pos_emb
        seq_emb = self.LayerNorm(seq_emb)
        seq_emb = self.dropout(seq_emb)
        return seq_emb

    def forward(self, input_ids, item_emb):
        attention_mask = (input_ids > 0).long()
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)

        subsequent_mask = t.triu(t.ones(attn_shape), diagonal=1) # torch.uint8
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        subsequent_mask = subsequent_mask.long()
        subsequent_mask = subsequent_mask.cuda()

        extended_attention_mask = extended_attention_mask * subsequent_mask
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        seq_embs = [self.get_seq_emb(input_ids, item_emb)]
        for trm in self.layers:
            seq_embs.append(trm(seq_embs[-1], extended_attention_mask))
        seq_emb = sum(seq_embs)

        return seq_emb

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class TransformerLayer(nn.Module):
    def __init__(self):
        super(TransformerLayer, self).__init__()
        self.attention = SelfAttentionLayer()
        self.intermediate = IntermediateLayer()

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output

class SelfAttentionLayer(nn.Module):
    def __init__(self):
        super(SelfAttentionLayer, self).__init__()
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.latdim / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.latdim, self.all_head_size)
        self.key = nn.Linear(args.latdim, self.all_head_size)
        self.value = nn.Linear(args.latdim, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.latdim, args.latdim)
        self.LayerNorm = nn.LayerNorm(args.latdim)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

        self.apply(self.init_weights)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = t.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = t.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            init(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

class IntermediateLayer(nn.Module):
    def __init__(self):
        super(IntermediateLayer, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(args.latdim, args.latdim * 4, bias=True),
            nn.GELU(),
            nn.Linear(args.latdim * 4, args.latdim, bias=True),
            nn.Dropout(args.hidden_dropout_prob),
            nn.LayerNorm(args.latdim)
        )

    def forward(self, x):
        return self.layers(x)

class LocalGraph(nn.Module):
    def __init__(self):
        super(LocalGraph, self).__init__()

    def make_noise(self, scores):
        noise = t.rand(scores.shape).cuda()
        noise = -t.log(-t.log(noise))
        return scores + noise

    def forward(self, adj, embeds, foo=None):
        order = t.sparse.sum(adj, dim=-1).to_dense().view([-1, 1])
        fstEmbeds = t.spmm(adj, embeds) - embeds
        fstNum = order

        emb = [fstEmbeds]
        num = [fstNum]

        for i in range(args.mask_depth):
            adj = sparse_dropout(adj, args.path_prob ** (i + 1))
            emb.append((t.spmm(adj, emb[-1]) - emb[-1]) - order * emb[-1])
            num.append((t.spmm(adj, num[-1]) - num[-1]) - order)
            order = t.sparse.sum(adj, dim=-1).to_dense().view([-1, 1])

        subgraphEmbeds = sum(emb) / (sum(num) + 1e-8)
        subgraphEmbeds = F.normalize(subgraphEmbeds, p=2)

        embeds = F.normalize(embeds, p=2)
        scores = t.sum(subgraphEmbeds * embeds, dim=-1)
        scores = self.make_noise(scores)

        _, candidates = t.topk(scores, args.num_mask_cand)

        return scores, candidates

class RandomMaskSubgraphs(nn.Module):
    def __init__(self):
        super(RandomMaskSubgraphs, self).__init__()

    def normalize(self, adj):
        degree = t.pow(t.sparse.sum(adj, dim=1).to_dense() + 1e-12, -0.5)
        newRows, newCols = adj._indices()[0, :], adj._indices()[1, :]
        rowNorm, colNorm = degree[newRows], degree[newCols]
        newVals = adj._values() * rowNorm * colNorm
        return t.sparse.FloatTensor(adj._indices(), newVals, adj.shape)

    def forward(self, adj, seeds):
        rows = adj._indices()[0, :]
        cols = adj._indices()[1, :]

        masked_rows = []
        masked_cols = []
        masked_idct = []

        for i in range(args.mask_depth):
            curSeeds = seeds if i == 0 else nxtSeeds
            nxtSeeds = list()
            idct = None
            for seed in curSeeds:
                rowIdct = (rows == seed)
                colIdct = (cols == seed)
                if idct == None:
                    idct = t.logical_or(rowIdct, colIdct)
                else:
                    idct = t.logical_or(idct, t.logical_or(rowIdct, colIdct))
            nxtRows = rows[idct]
            nxtCols = cols[idct]
            masked_rows.extend(nxtRows)
            masked_cols.extend(nxtCols)
            rows = rows[t.logical_not(idct)]
            cols = cols[t.logical_not(idct)]
            nxtSeeds = nxtRows + nxtCols
            if len(nxtSeeds) > 0 and i != args.mask_depth - 1:
                nxtSeeds = t.unique(nxtSeeds)
                cand = t.randperm(nxtSeeds.shape[0])
                nxtSeeds = nxtSeeds[cand[:int(nxtSeeds.shape[0] * args.path_prob ** (i + 1))]] # the dropped edges from P^k

        masked_rows = t.unsqueeze(t.LongTensor(masked_rows), -1)
        masked_cols = t.unsqueeze(t.LongTensor(masked_cols), -1)
        masked_edge = t.hstack([masked_rows, masked_cols])
        encoder_adj = self.normalize(t.sparse.FloatTensor(t.stack([rows, cols], dim=0), t.ones_like(rows).cuda(), adj.shape))

        return encoder_adj, masked_edge
