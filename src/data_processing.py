from __future__ import annotations

from typing import Sequence, Dict, Tuple, List

import numpy as np
import torch
import community as cl
import networkx as nx
import scipy.sparse as sp

from collections import defaultdict

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_cora(data_dir: str = './cora/') -> Sequence[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load cora dataset from filesystem

    :param data_dir: filepath from which to load data
    :return: tuple containing node features, nodes' labels and the adjacency list
    """
    num_nodes = 2708
    num_feats = 1433
    feat_data = np.zeros((num_nodes, num_feats))
    labels = np.empty((num_nodes, 1), dtype=np.int64)
    node_map = {}
    label_map = {}
    with open(data_dir + 'cora.content') as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            feat_data[i, :] = list(map(float, info[1:-1]))
            node_map[info[0]] = i
            if not info[-1] in label_map:
                label_map[info[-1]] = len(label_map)
            labels[i] = label_map[info[-1]]

    adj_lists = defaultdict(set)
    with open(data_dir + 'cora.cites') as fp:
        for i, line in enumerate(fp):
            info = line.strip().split()
            paper1 = node_map[info[0]]
            paper2 = node_map[info[1]]
            adj_lists[paper1].add(paper2)
            adj_lists[paper2].add(paper1)

    # feat_data = feat_data / np.expand_dims(np.sum(feat_data, axis=1), axis=1)

    return feat_data, labels, adj_lists


def create_networkx_graph(adj_list: Dict[int, Sequence[int]]) -> nx.Graph:
    """
    Generate a networkx graph from an adjacency list

    :param adj_list: graph adjacency list
    :return: networkx graph of the corresponding adjacency list
    """
    g_nx = nx.Graph()
    edge_list = [[[k, v] for v in vs] for k, vs in adj_list.items()]
    edge_list = [x for sublist in edge_list for x in sublist]
    # print(f'edge list length : {len(edge_list)}')
    g_nx.add_edges_from(edge_list)

    return g_nx


def get_louvain_partition(g_nx: nx.Graph, level_partition: int, verbose: int = 0) -> Dict[int, int]:
    """
    Compute the partition of the graph nodes and returns the level partition specified by the argument

    :param g_nx: networkx graph used for the partition
    :param level_partition: level of partition that we want
    :param verbose: verbosity level
    :return: partition dictionary with the node as key and the partition as value
    """
    dendo = cl.generate_dendrogram(g_nx)
    if verbose > 0:
        for level in range(len(dendo) - 1):
            print("partition at level", level, "is",
                  cl.partition_at_level(dendo, level))
            print('modularity at level', level, 'is', cl.modularity(cl.partition_at_level(dendo, level), g_nx))
            print('number of partitions at level', level, 'is', max(cl.partition_at_level(dendo, level).values()))
    print('biggest partition', np.max(
        np.bincount(list(cl.partition_at_level(dendo, level_partition).values()))))
    return cl.partition_at_level(dendo, level_partition)


def invert_partitions_dict(partitions: Dict[int, int], min_size_threshold: int = 1) -> Dict[int, List[int]]:
    """
    Filter partitions based on their size
    :param partitions: partitions dictionary
    :param min_size_threshold: filter partition that have size < minimum size. No filter by default
    :return: inverted dictionary partitions with partitions as keys and the list of nodes that belong to them as a value
    """
    partition_assignment = list(partitions.values())
    partition_sizes = np.bincount(np.array(partition_assignment))
    # filter the partition in order to retain only the ones with size bigger than n
    allowed_partiton_indexes = np.argwhere(partition_sizes >= min_size_threshold)
    partition_list = [[k, v] for k, v in partitions.items()]
    filtered_partitions = list(filter(lambda t: t[1] in allowed_partiton_indexes, partition_list))
    inverted_partitions_dict = {}
    for v, p in filtered_partitions:
        inverted_partitions_dict.setdefault(p, []).append(v)

    return inverted_partitions_dict


def create_global_nodes_edges(partitions_dict: Dict[int, List[int]], feat_data: np.ndarray,
                              global_features_aggr: str = "mean",
                              connections: str = 'partitions') -> Tuple[np.ndarray, np.ndarray]:
    """
    Add summary nodes that are fully connected between each other. They are also connected either to all the nodes in
        the graph or only on to the nodes of the partition they represent
        partition

    :param partitions_dict: this dictionary indicates for each partition which nodes belong to.
    :param feat_data: nodes' feature vectors
    :param global_features_aggr: aggregation strategy used for creating the feature vectors of global nodes
    :param connections: connection strategy between regular nodes and global nodes
    :return:
    """
    N = feat_data.shape[0]
    num_global_nodes = len(partitions_dict)
    g_adj_list = defaultdict(set)
    g_nodes_feat = []
    g_id = feat_data.shape[0]

    for p in partitions_dict.keys():
        # g_node = torch.index_select(torch.tensor(feat_data), 0, torch.tensor(partitions_dict[p], dtype=torch.int64))
        g_node = np.take(feat_data, np.array(partitions_dict[p], dtype=np.int), 0)
        if global_features_aggr == "mean":
            g_node = np.mean(g_node, 0)
        elif global_features_aggr == "max":
            g_node = np.max(g_node, 0)
        else:
            raise ValueError(f"{global_features_aggr} aggregation does not exist")
        # global node connected with all the other nodes
        g_nodes_feat.append(g_node)
        if connections == 'all':
            for node in partitions_dict[p]:
                g_adj_list[node].add(g_id)
            g_adj_list[g_id] = (set(range(N + num_global_nodes)))
            # remove self edge
            g_adj_list[g_id].remove(g_id)
        # global node connected with partitions only ant to all global nodes
        elif connections == 'partitions':
            # g_adj_list[g_id] = set(node for node in partitions_dict[p]).union(set(range(N, N + num_global_nodes)))
            for node in partitions_dict[p]:
                g_adj_list[g_id].add(node)
                g_adj_list[node].add(g_id)
            # adding edges to all global nodes
            g_adj_list[g_id].update(set(range(N, N + num_global_nodes)))
            # remove self edge
            g_adj_list[g_id].remove(g_id)
        else:
            raise ValueError('connections value', connections, 'does not exist')
        # e_list = [[g_id, node] for node in range(2708)]
        # e_global_list = [[g_id]]
        # edges_lists.extend(e_list)
        g_id += 1

    return g_nodes_feat, g_adj_list


def create_louvain_global_nodes_edges(g_nx: nx.Graph, feat_data: np.ndarray,
                                      level_partition: int, global_features_aggr: str,
                                      connections: str = 'partitions') -> Tuple[np.ndarray, np.ndarray]:

    partition = get_louvain_partition(g_nx, level_partition=level_partition)
    inverted_partition = invert_partitions_dict(partition)
    g_feat_data, g_adj_list = create_global_nodes_edges(inverted_partition, feat_data, global_features_aggr, connections=connections)
    print('number of global nodes is', len(g_feat_data))
    g_feat_data = np.array(g_feat_data)
    # g_feat_data = np.array([t.numpy() for t in g_feat_data])

    # p_adj_list = full_partition_graph(partition)
    # g_adj_list.update((k, g_adj_list[k].union(p_adj_list[k])) for k in range(len(p_adj_list)))

    return g_feat_data, g_adj_list


def add_global_nodes_edges(g_nx : nx.Graph, feat_data: np.ndarray, adj_list: np.ndarray,
                           g_feat_data: np.ndarray, g_adj_list: np.ndarray):
    """

    :param g_nx: 
    :param feat_data: 
    :param adj_list: 
    :param g_feat_data: 
    :param g_adj_list: 
    :return: 
    """
    feat_data = np.concatenate([feat_data, g_feat_data], 0)
    # adj_list.update((k, adj_list[k].union(g_adj_list[k])) for k in range(len(g_adj_list)))
    adj_list.update((k, adj_list[k].union(g_adj_list[k])) for k in range(len(feat_data)))
    g_edge_list = [[[k, v] for v in vs] for k, vs in g_adj_list.items()]
    g_edge_list = [x for sublist in g_edge_list for x in sublist]
    g_nx.add_edges_from(g_edge_list)

    return g_nx, feat_data, adj_list


def laplacian_positional_encoding(g_nx, pos_enc_dim=2):
    """
        Graph positional encoding v/ Laplacian eigenvectors
    """

    # Laplacian
    A = nx.linalg.graphmatrix.adj_matrix(g_nx).astype(float)
    # N = sp.diags(np.array(sorted(g_nx.degree))[:, 1].clip(1) ** -0.5, dtype=float)
    # N = sp.diags(np.array(sorted(g_nx.degree))[:, 1] ** -0.5, dtype=float)
    # L = sp.eye(len(g_nx.nodes)) - N @ A @ N
    L = sp.csgraph.laplacian(A, normed=True)

    # # Eigenvectors with scipy
    # # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim+1, which='SR')
    # EigVal, EigVec = sp.linalg.eigs(L, k=pos_enc_dim + 1, which='SR', tol=1e-2)  # for 40 PEs
    # EigVec = EigVec[:, EigVal.argsort()]  # increasing order
    # pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float().to(device)

    # Eignenvectors numpy
    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim + 1]).float().to(device)

    return pos_enc
