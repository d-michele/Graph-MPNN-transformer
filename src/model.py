import math
import profile
from collections import defaultdict
from functools import partial

import networkx as nx
import torch
import torch.nn as nn

from layers import NeighbourSampler, TransformerEncoderLayer, EncoderBlock, Node2VecSampler
from my_precomp import MyPreComp

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# device = torch.device("cpu")

class TransformerModel(nn.Module):
    def __init__(self, feat_data: torch.Tensor, adj_lists: defaultdict, g_adj_list: defaultdict, lap_enc: torch.Tensor,
                 num_layer: int, emb_size: int, n_head: int, hid_size: int, dropout: float,
                 num_classes: int, sampler: str, num_samples: int = None, g_nx: nx.Graph = None, **kwargs):
        """
        Initialize the graph neural network

        :param feat_data: Nodes features. Size:(V', d) V' graph order, d dimension of each node feature vector
        :param  adj_lists: adjacency list containing both, normal-normal edges, global-global edges and global-normal
            edges
        :param g_adj_list: adjacenty list containing global-normal edges and global-global edges, len = V'
        :param lap_enc: laplacian positional encoder size: (V', d_l) . The dimension d_l is chosen when computing the
            positional encoding
        :param num_layer: number of layers of the graph neural network.
        :param emb_size: Dimension of input embedding
        :param n_head: Number of attention heads
        :param hid_size: Hidden size
        :param dropout: Dropout rate
        :param num_classes: Output size
        :param sampler: String that indicate the neighborhood sampling strategy
        :param num_samples: number of samples drawn used in neighborhood sampler
        :param g_nx: graph needed for node2vec samples
        :param kwargs:
        """
        super(TransformerModel, self).__init__(**kwargs)
        self.adj_lists = adj_lists
        self.num_samples = num_samples
        g_order, num_feat = feat_data.shape
        if num_layer < 1:
            raise ValueError(
                f"Number of layers must be at least 1. Actual:{num_layer}")
        if sampler == 'neighborhood':
            sampler_fun = partial(
                NeighbourSampler, padding_index=g_order, lap_enc=lap_enc)
        elif sampler == 'node2vec':
            p = 0.25
            q = 0.25
            workers = 8
            n2v = MyPreComp(q, p, workers, True)
            ids = [str(i) for i in range(g_nx.order())]
            n2v.from_mat(nx.to_numpy_array(g_nx), ids)
            # precompute and save 2nd order transition probs (for PreComp only)
            n2v.preprocess_transition_probs()
            sampler_fun = partial(Node2VecSampler, p=p,
                                  q=q, graph=g_nx, adj_list=adj_lists, n2v=n2v)
        else:
            raise ValueError("Chosen sampler not implemented")

        self.encBlks = []
        l = 0
        self.features = nn.Embedding(
            g_order + 1, num_feat, padding_idx=g_order)
        # we add the embedding also for padding with the id 2709
        feat_data_pad = torch.cat((feat_data, torch.zeros(
            (1, feat_data.shape[1]), device=device)), dim=0)
        self.features.weight = nn.Parameter(feat_data_pad, requires_grad=False)
        self.sampler1 = sampler_fun(
            self.features, input_l=True, emb_size=emb_size, lap_enc=lap_enc)
        self.last_enc_block = EncoderBlock(self.features, adj_lists, g_adj_list, self.sampler1, emb_size, n_head,
                                           hid_size,
                                           dropout, name='t' + str(l), num_samples=self.num_samples, affine=True,
                                           lap_enc=lap_enc)
        self.encBlks.append(self.last_enc_block)
        for l in range(1, num_layer):
            sampler_block = sampler_fun(self.last_enc_block)
            # self.sampler2 = Node2VecSampler(lambda nodes: self.encBlk1(nodes), p=p, q=q)
            self.last_enc_block = EncoderBlock(self.last_enc_block, adj_lists, g_adj_list, sampler_block, emb_size,
                                               n_head, hid_size,
                                               dropout, 't' + str(l), num_samples=self.num_samples)
            self.encBlks.append(self.last_enc_block)

        self.dropout = nn.Dropout(p=dropout)
        self.dense = nn.Linear(hid_size, num_classes)
        nn.init.xavier_uniform_(self.dense.weight.data)
        self.loss_ = nn.CrossEntropyLoss()

    def forward(self, nodes):
        """

        :param nodes: node indexes
        :return: output scores
        """
        h = self.last_enc_block(nodes)

        h = self.dropout(h)
        scores = self.dense(torch.squeeze(h))

        return scores

    def loss(self, scores, labels):
        """
        Model's loss function

        :param scores: output scores
        :param labels: class labels
        :return: loss
        """
        return self.loss_(scores, torch.squeeze(labels))
