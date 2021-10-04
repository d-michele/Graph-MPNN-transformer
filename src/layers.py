import itertools
import os

import torch
import torch.nn as nn
import random
import numpy as np
import networkx as nx

import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
from typing import Optional, Any
from pecanpy.node2vec import PreComp
from numba import get_num_threads, jit, prange
from numba.np.ufunc.parallel import _get_thread_id

from torch.nn.init import xavier_uniform_

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")


class EncoderBlock(nn.Module):
    def __init__(self, features, adj_lists, g_adj_lists, sampler, emb_size, n_head, hid_dim, dropout, name, num_samples,
                 affine=False, lap_enc=None):
        """
        Transformer encoder block

        :param features:
        :param adj_lists:
        :param g_adj_lists:
        :param sampler:
        :param emb_size:
        :param n_head:
        :param hid_dim:
        :param dropout:
        :param name:
        :param num_samples:
        :param affine:
        :param lap_enc:
        """
        super(EncoderBlock, self).__init__()
        self.features = features
        self.adj_lists = adj_lists
        self.g_adj_lists = g_adj_lists
        self.sampler = sampler
        self.num_samples = num_samples
        self.name = name
        self.emb_size = emb_size
        self.affine = affine
        self.lap_enc = lap_enc
        self.dropout = nn.Dropout(p=dropout)

        self.encoder = TransformerEncoderLayer(
            emb_size, n_head, hid_dim, dropout)

    # @profile
    def forward(self, nodes):
        neighs_seq = {node: self.adj_lists[int(node)] for node in nodes}
        nodes_emb, neighs_emb, neighs_seq, padding_mask = self.sampler(
            nodes, neighs_seq, None, num_sample=self.num_samples)

        h_queries = nodes_emb
        neighs_emb = neighs_emb.permute(1, 0, 2)
        h_keys_values = neighs_emb.type(torch.float32).to(device)

        h_queries = self.dropout(h_queries)
        h_keys_values = self.dropout(h_keys_values)
        h_queries = h_queries.unsqueeze(0)

        assert h_queries.shape[2] == self.emb_size and h_keys_values.shape[2] == self.emb_size
        h = self.encoder(h_queries, h_keys_values, h_keys_values,
                         src_key_padding_mask=padding_mask)

        return torch.squeeze(h)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim_model, nhead, dim_feedforward=2048, dropout=0.1, activation="relu",
                 layer_norm_eps=1e-5, batch_first=False):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(dim_model, nhead, dropout=dropout)
        # self.self_attn = MyAttention(dim_model, dim_model, nhead, dropout=dropout)

        # Implementation of Feedforward model
        self.linear1 = nn.Linear(dim_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, dim_model)

        self.norm1 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.norm2 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.norm3 = nn.LayerNorm(dim_model, eps=layer_norm_eps)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = F.relu

        self._reset_parameters()

    def _reset_parameters(self):
        r"""Initiate parameters in the transformer model."""
        for p in self.parameters():
            if p.dim() > 1:
                xavier_uniform_(p)
            # else:
            #     p.data.fill_(0)


    # @profile
    def forward(self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor,
                src_mask: Optional[torch.Tensor] = None,
                src_key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        r"""Pass the input through the encoder layer.

        Args:

        Shape:
        """
        queries = self.norm3(queries)
        keys = self.norm3(keys)
        values = self.norm3(values)
        src2 = self.self_attn(queries, keys, values,
                              key_padding_mask=src_key_padding_mask)[0]
        src = queries + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class MyAttention(nn.Module):
    def __init__(self, c_in, c_out, nheads=8, dropout=0.0, attention='dot', concat_heads=True):
        super().__init__()
        self.nheads = nheads
        concat_heads = concat_heads
        # if concat_heads:
        #     assert c_out % nheads == 0, "Number of output features must be a multiple of the cout of heads"
        #     c_out = c_out // nheads

        # Sub-modules and parameters needed in the layer
        self.projectionQ = nn.Linear(c_in, c_out * nheads)
        self.projectionK = nn.Linear(c_in, c_out * nheads)
        self.projectionV = nn.Linear(c_in, c_out * nheads)
        # self.projectionO = nn.Linear(c_out * nheads, c_out * nheads).to(device)
        self.projectionO = nn.Linear(c_out * nheads, c_out)
        # a = nn.Parameter(torch.Tensor(nheads, 2 * c_out)).to(device) # One per head
        if attention == 'neural_net':
            self.a = nn.Parameter(torch.Tensor(nheads, c_out))
            self.b = nn.Parameter(torch.Tensor(nheads))
            self.a_h = nn.Parameter(torch.Tensor(nheads, nheads))
            self.b_h = nn.Parameter(torch.Tensor(nheads))

        self.activation = F.relu
        self.dropout = nn.Dropout(dropout)
        # self.dropout = nn.Dropout(0.0)
        self.leakyrelu = nn.LeakyReLU(0.2)

        # Initialization from the original implementation
        # gain 1.414
        nn.init.xavier_uniform_(self.projectionQ.weight.data)
        nn.init.xavier_uniform_(self.projectionK.weight.data)
        nn.init.xavier_uniform_(self.projectionV.weight.data)
        nn.init.xavier_uniform_(self.projectionO.weight.data)
        if attention == 'neural_net':
            nn.init.xavier_uniform_(self.a.data)
            nn.init.xavier_uniform_(self.a_h.data)
        nn.init.constant_(self.b, .0)
        nn.init.constant_(self.b_h, .0)

    def forward(self, queries, keys, values, key_padding_mask=None):
        batch_size, num_nodes = 1, queries.size(1)
        # print(queries.shape)

        queries = queries.permute(1, 0, 2)
        queries = torch.unsqueeze(queries, 0)
        keys = keys.permute(1, 0, 2)
        keys = torch.unsqueeze(keys, 0)
        values = values.permute(1, 0, 2)
        values = torch.unsqueeze(values, 0)

        keys = keys * \
               ~torch.unsqueeze(torch.unsqueeze(key_padding_mask, -1), 0)
        values = values * \
                 ~torch.unsqueeze(torch.unsqueeze(key_padding_mask, -1), 0)

        queries = self.projectionQ(queries)
        keys = self.projectionK(keys)
        values = self.projectionV(values)
        queries = queries.view(batch_size, queries.size(1), self.nheads, -1)
        keys = keys.view(batch_size, keys.size(
            1), keys.size(2), self.nheads, -1)
        values = values.view(batch_size, values.size(
            1), values.size(2), self.nheads, -1)
        # queries = torch.unsqueeze(queries, 2)

        # #### Neural network Attention
        # # a_input2 = queries**2 + keys**2 - 2*queries*keys
        # a_input2 = queries - keys
        # # a_input2 = (torch.cat([queries.repeat(1, 1, keys.size(2), 1, 1), keys], dim=-1))
        # # a_input2 = torch.cat([queries.repeat(1, 1, keys.size(2), 1, 1), (keys - queries)], dim=-1)
        # attn_logits = torch.einsum('bijhc,hc->bijh', a_input2, self.a) + self.b
        # attn_logits = torch.einsum('bijh, hf -> bijf', self.activation(attn_logits), self.a_h) + self.b_h

        # # a_input2 = torch.cat([queries.repeat(1, 1, keys.size(2), 1, 1), keys], dim=-1)
        # queries_sq = torch.squeeze(queries, 2)
        ####

        # vaswanii attention
        attn_logits = torch.einsum("bqhc, bqkhc -> bqkh", queries, keys)
        attn_logits = attn_logits / \
                      torch.sqrt(torch.tensor(keys.size(-1),
                                              dtype=torch.float32, device=device))
        ## ~ is bitwise negation operator
        attn_logits = attn_logits * ~torch.unsqueeze(torch.unsqueeze(key_padding_mask, -1), 0)
        attn_logits = attn_logits.masked_fill(attn_logits == 0, -9e15)

        attn_probs = self.dropout(F.softmax(attn_logits, dim=2))
        out_feat = torch.einsum("bqkhc, bqkh -> bqhc", values, attn_probs)

        out_feat = out_feat.reshape(batch_size, num_nodes, -1)
        out_feat = self.projectionO(out_feat)

        return out_feat


class AddNorm(nn.Module):
    def __init__(self, normalized_shape, dropout, **kwargs):
        super(AddNorm, self).__init__(**kwargs)
        self.dropout = nn.Dropout(dropout)
        self.ln = nn.LayerNorm(normalized_shape)

    def forward(self, X, Y):
        return self.ln(self.dropout(Y) + X)


class PositionWiseFFN(nn.Module):
    def __init__(self, ffn_num_input, ffn_num_hidden, ffn_num_outputs,
                 **kwargs):
        super(PositionWiseFFN, self).__init__(**kwargs)
        self.dense1 = nn.Linear(ffn_num_input, ffn_num_hidden)
        self.relu = nn.ReLU()
        self.dense2 = nn.Linear(ffn_num_hidden, ffn_num_outputs)

    def forward(self, X):
        return self.dense2(self.relu(self.dense1(X)))


class NeighbourSampler(nn.Module):
    def __init__(self, features, padding_index, input_l=False, lap_embeddings=None,
                 emb_size=None, lap_enc=None, dropout=0.0, **kwargs):
        super(NeighbourSampler, self).__init__(**kwargs)
        self.features = features
        self.padding_index = padding_index
        self.input = input_l
        self.lap_enc = lap_enc
        dropout = 0.2
        self.dropout = nn.Dropout(dropout)
        if input_l:
            self.input_embeddings = nn.Linear(1433, emb_size)
            self.layer_norm = nn.LayerNorm(emb_size)
            if lap_enc is not None:
                self.lap_embedding = nn.Linear(lap_enc.shape[1], emb_size)
                nn.init.xavier_uniform_(self.lap_embedding.weight.data)

    def forward(self, nodes, to_neighs, g_neighs, num_sample=10):
        # Local pointers to functions (speed hack)
        _set = set
        if num_sample is not None:
            _sample = random.sample
            samp_neighs = [_set(_sample(to_neigh, num_sample))
                           if len(to_neigh) >= num_sample else to_neigh for to_neigh in to_neighs.values()]
        else:
            samp_neighs = to_neighs.values()

        unique_nodes_list = list(set.union(*samp_neighs))
        # unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        # mask = torch.autograd.Variable(torch.zeros(len(samp_neighs), len(unique_nodes)))
        # column_indices = [unique_nodes[n] for samp_neigh in samp_neighs for n in samp_neigh]
        # row_indices = [i for i in range(len(samp_neighs)) for j in range(len(samp_neighs[i]))]
        # mask[row_indices, column_indices] = 1
        # mask = mask.to(device)
        # num_neigh = mask.sum(1, keepdim=True)
        # mask = mask.div(num_neigh)
        samp_neighs_l = list([list(neighs) for neighs in samp_neighs])
        seq_length = torch.tensor([len(seq) for seq in samp_neighs_l])
        num_sample = max(seq_length)
        if num_sample is None:
            num_sample = max(seq_length)
            # print(num_sample)
        # if not self.input:
        embed_matrix = self.features(torch.tensor(unique_nodes_list, dtype=torch.long, device=device)).type(
            torch.float32).to(device)
        # nodes_t = torch.tensor(nodes, device=device).to(device)
        nodes_t = nodes
        nodes_emb = self.features(nodes_t).type(torch.float32).to(device)
        if self.input:
            if self.lap_enc is not None:
                embed_matrix = self.input_embeddings(embed_matrix) + self.lap_embedding(self.lap_enc[unique_nodes_list])
                # self.lap_enc[unique_nodes_list]
                nodes_emb = self.input_embeddings(nodes_emb) + self.lap_embedding(self.lap_enc[nodes_t])
                # self.lap_enc[nodes_t]
                embed_matrix = self.dropout(embed_matrix)
            else:
                embed_matrix = self.input_embeddings(embed_matrix)
                nodes_emb = self.input_embeddings(nodes_emb)

        reverse_index = {unique_nodes_list[x]: x for x in range(
            len(unique_nodes_list))}
        nodes_list_unique_index = [
            [reverse_index[x] for x in samp_neighs_l[i]] for i in range(len(samp_neighs_l))]
        nodes_list_unique_index_t = [torch.tensor(
            neighs) for neighs in nodes_list_unique_index]
        # nodes_list_unique_index_t = torch.tensor(nodes_list_unique_index).to(device)
        # samp_neighs_t = torch.nn.utils.rnn.pad_sequence(nodes_list_unique_index_t,
        #                                                 padding_value=self.padding_index).to(device)
        samp_neighs_t_tmp = torch.nn.utils.rnn.pad_sequence(nodes_list_unique_index_t,
                                                            padding_value=nodes_list_unique_index_t[0][0]).to(device)
        # self.padding_index = nodes_list_unique_index_t[0][0]
        # samp_neighs_t = samp_neighs_t.t()
        samp_neighs_t_tmp = samp_neighs_t_tmp.t()
        neighs_emb = torch.index_select(embed_matrix, 0, samp_neighs_t_tmp.reshape(-1)) \
            .reshape(-1, num_sample, embed_matrix.shape[1])

        # neigh_emb = self.features(neighs_seq.to(device))

        # # to_feats = mask.mm(embed_matrix.type(torch.FloatTensor).to(device))
        # else:
        #     # nodes_emb = self.features(torch.tensor(nodes, device=device))
        #     samp_neighs_t = [torch.tensor(neighs) for neighs in samp_neighs_l]
        #     samp_neighs_t = torch.nn.utils.rnn.pad_sequence(samp_neighs_t, padding_value=self.padding_index)
        #     neighs_emb = self.features(samp_neighs_t.to(device))
        #     # neighs_emb = torch.gather(embed_matrix, 0, samp_neighs_t)

        # create paddnig masks
        if num_sample is None:
            num_sample = max(seq_length)
        # padding_mask = [[i >= seq for i in range(num_sample)] for seq in seq_length]
        # padding_mask = torch.tensor(padding_mask, device=device)
        padding_mask = torch.arange(1, num_sample + 1).repeat(len(seq_length), 1).reshape(len(seq_length), -1).to(
            device)
        max_seqs = torch.unsqueeze(seq_length, 1).repeat(
            1, num_sample).to(device)
        padding_mask = padding_mask > max_seqs

        return nodes_emb, neighs_emb, samp_neighs_l, padding_mask


class Node2VecSampler(nn.Module):
    def __init__(self, features, adj_list, p=1, q=1, padding_index=0, input_l=False, emb_size=64,
                 lap_enc=None, graph=None, workers=8, n2v=None):
        super(Node2VecSampler, self).__init__()
        if n2v is None:
            self.n2v = PreComp(q, p, workers, True)
            ids = [str(i) for i in range(graph.order())]
            self.n2v.from_mat(nx.to_numpy_array(graph), ids)
            # precompute and save 2nd order transition probs (for PreComp only)
            self.n2v.preprocess_transition_probs()
        else:
            self.n2v = n2v
        # load graph from edgelist file
        # dir = os.getcwd()
        # self.n2v.read_edg(dir + "/../cora/cora.cites", weighted=False, directed=False)

        # # self.G = nx.DiGraph()
        # self.G = graph
        # self.graph_construction(list(range(len(adj_list))), adj_list)
        self.p = p
        self.q = q
        # self.directed = True
        #
        # self.alias_nodes = {}
        # self.alias_edges = {}
        #
        self.features = features
        self.input = input_l
        self.lap_enc = lap_enc
        self.padding_index = padding_index
        if input_l:
            self.input_embeddings = nn.Linear(1433, emb_size)
            self.layer_norm = nn.LayerNorm(emb_size)
            if lap_enc is not None:
                self.lap_embedding = nn.Linear(lap_enc.shape[1], emb_size)

    def forward(self, nodes, to_neighs, num_sample=10, val=False):
        # generate random walks, which could then be used to train w2v
        if self.walks is not None and self.training is True:
            return self.walks[0], self.walks[1].self.walks[2], self.walks[3]
        nodes_np = nodes.detach().cpu().data.numpy()
        walks = self.n2v.simulate_walks(nodes_np, num_walks=1, walk_length=15)
        # alternatively, generate the embeddings directly using ``embed``
        # emd = g.embed()

        # # if not self.input_layer:
        # nodes = [int(node) for node in nodes]
        # if self.G == None:
        #     self.graph_construction(nodes, to_neighs)
        # self.preprocess_transition_probs()
        # # walks = self.simulate_walks(nodes, 1, num_sample)
        # walks = self.simulate_walks(nodes, 15, 5)
        #
        groups = {}
        for w in walks:
            w = [int(n) for n in w]
            groups.setdefault(w[0], []).append(w)

        sorted_walks = list(groups.values())
        # sorted_walks = [[int(n) for n in ns] for ns in sorted_walks]
        _set = set
        samp_neighs = [_set(itertools.chain.from_iterable(w))
                       for w in sorted_walks]
        print(len(samp_neighs))
        unique_nodes_list = list(set.union(*samp_neighs))
        # unique_nodes_list = [int(n) for n in unique_nodes_list]
        unique_nodes = {n: i for i, n in enumerate(unique_nodes_list)}
        # samp_neighs_l = list([list(neighs) for neighs in samp_neighs])
        samp_neighs_l = list([list(neighs) for neighs in samp_neighs])
        seq_length = torch.tensor([len(seq) for seq in samp_neighs_l])
        num_sample = max(seq_length)
        if num_sample is None:
            num_sample = max(seq_length)
            # print(num_sample)
        # if not self.input:
        embed_matrix = self.features(torch.tensor(unique_nodes_list, dtype=torch.long, device=device)).type(
            torch.float32).to(device)
        # nodes_t = torch.tensor(nodes, device=device).to(device)
        nodes_t = nodes
        nodes_emb = self.features(nodes_t).type(torch.float32).to(device)
        if self.input:
            if self.lap_enc is not None:
                embed_matrix = self.input_embeddings(
                    embed_matrix) + self.lap_embedding(self.lap_enc[unique_nodes_list])
                nodes_emb = self.input_embeddings(
                    nodes_emb) + self.lap_embedding(self.lap_enc[nodes_t])
            else:
                embed_matrix = self.input_embeddings(embed_matrix)
                nodes_emb = self.input_embeddings(nodes_emb)

        reverse_index = {unique_nodes_list[x]: x for x in range(
            len(unique_nodes_list))}
        nodes_list_unique_index = [
            [reverse_index[x] for x in samp_neighs_l[i]] for i in range(len(samp_neighs_l))]
        nodes_list_unique_index_t = [torch.tensor(
            neighs) for neighs in nodes_list_unique_index]
        # nodes_list_unique_index_t = torch.tensor(nodes_list_unique_index).to(device)
        samp_neighs_t = torch.nn.utils.rnn.pad_sequence(nodes_list_unique_index_t,
                                                        padding_value=self.padding_index).to(device)
        # samp_neighs_t_tmp = torch.nn.utils.rnn.pad_sequence(nodes_list_unique_index_t,
        #                                                     padding_value=nodes_list_unique_index_t[0][0]).to(device)
        # samp_neighs_t_tmp = torch.nn.utils.rnn.pad_sequence(nodes_list_unique_index_t,
        #                                                     padding_value=self.padding_index).to(device)
        # self.padding_index = nodes_list_unique_index_t[0][0]
        samp_neighs_t = samp_neighs_t.t()
        # samp_neighs_t_tmp = samp_neighs_t_tmp.t()
        neighs_emb = torch.index_select(embed_matrix, 0, samp_neighs_t.reshape(-1)) \
            .reshape(-1, num_sample, embed_matrix.shape[1])

        # create paddnig masks
        if num_sample is None:
            num_sample = max(seq_length)
        padding_mask = torch.arange(1, num_sample + 1).repeat(len(seq_length), 1).reshape(len(seq_length), -1).to(
            device)
        max_seqs = torch.unsqueeze(seq_length, 1).repeat(
            1, num_sample).to(device)
        padding_mask = padding_mask > max_seqs

        # self.walks = (nodes_emb, neighs_emb, samp_neighs_t, padding_mask)
        return nodes_emb, neighs_emb, samp_neighs_t, padding_mask

    def graph_construction(self, nodes, adj_lists):
        if self.G is not None:
            self.G = self.G.to_directed()
        else:
            self.G = nx.DiGraph()
            # all graph adj list
            for node, neighs in adj_lists.items():
                for neigh in neighs:
                    self.G.add_edge(node, neigh)

        for edge in self.G.edges():
            self.G[edge[0]][edge[1]]['weight'] = 1
            # self.G[edge[1]][edge[0]]['weight'] = 1
