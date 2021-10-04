import os
import sys
from pprint import pprint

import torch
import numpy as np
import random
import time
import yaml

from config import Config
from model import TransformerModel
from sklearn.metrics import f1_score
from torch.utils.tensorboard import SummaryWriter
from data_processing import *

# device = torch.device('cpu')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument()
    try:
        config = Config()
    except yaml.YAMLError as exc:
        print(exc)
        sys.exit(0)
    # tracemalloc.start()
    writer = SummaryWriter()
    pprint(vars(config))

    # Todo dataloader
    feat_data, labels, adj_list = load_cora()
    N = feat_data.shape[0]
    # features =nn.Embedding(2708, 1433)
    # features.weight = nn.Parameter(torch.FloatTensor(feat_data), requires_grad=False)
    num_nodes = feat_data.shape[0]
    rand_indices = np.random.permutation(num_nodes)
    if config.split == 'gcn':
        test = rand_indices[1708:]
        val = rand_indices[140:640]
        train = list(rand_indices[:140])
    elif config.split == 'caley':
        test = rand_indices[2208:]
        val = rand_indices[1708:2208]
        train = list(rand_indices[:1708])
    else:
        raise ValueError("Choose a datasplit between gcn or caley")

    g_nx = create_networkx_graph(adj_list)
    lap_enc = None
    g_adj_list = None
    if config.laplacian_encoding:
        lap_enc = laplacian_positional_encoding(g_nx)
    if config.global_nodes:
        g_feat_data, g_adj_list = create_louvain_global_nodes_edges(g_nx, feat_data, config.level_partition,
                                                            config.global_features_aggr,
                                                            config.global_nodes_connections)
        if config.laplacian_encoding:
            global_lap_enc = torch.zeros(g_feat_data.shape[0], 2, device=device)
            lap_enc_nodes = [lap_enc[list(filter(lambda x: x < N, g_adj_list[i]))] for i in
                             range(len(adj_list), len(g_adj_list))]
            global_lap_enc = torch.stack([torch.mean(enc, dim=0) for enc in lap_enc_nodes])
            lap_enc = torch.cat([lap_enc, global_lap_enc], dim=0)
        g_nx, feat_data, adj_list = add_global_nodes_edges(g_nx, feat_data, adj_list, g_feat_data, g_adj_list)

    # add self loop
    for v, u in adj_list.items():
        adj_list[v].add(v)

    feat_data = torch.tensor(feat_data, device=device)
    model = TransformerModel(feat_data, adj_list, g_adj_list, lap_enc, config.layers, config.hid_dim,
                             config.n_head, config.hid_dim, config.dropout, num_classes=7, sampler=config.sampler,
                             num_samples=config.num_samples, g_nx=g_nx).to(device)
    if config.optimizer == 'adam':
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.lr,
                                     weight_decay=config.weight_decay)
    elif config.optimizer == 'sgd':
        optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=config.weight_decay,
                                    momentum=config.sgd_momentum)
    else:
        raise ValueError(f"{config.optimizer} not yet implemented")


    train = torch.tensor(train, device=device)
    val = torch.tensor(val, device=device)
    test = torch.tensor(test, device=device)
    labels = torch.tensor(labels, dtype=torch.long, device=device)

    times = []
    i = 1
    best_val = 0
    timestr = time.strftime("%Y%m%d-%H%M%S")
    bm_ckpt_filename = timestr + '.pt'
    for epoch in range(config.epochs):
        model.train()
        idx = torch.randperm(train.shape[0])
        batch_nodes = train[idx].view(train.size())
        start_time = time.time()
        optimizer.zero_grad()
        scores = model(batch_nodes)
        loss = model.loss(scores, torch.index_select(labels, 0, batch_nodes))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        end_time = time.time()
        times.append(end_time - start_time)
        print(epoch, float(loss.data))
        writer.add_scalar('Loss/train', float(loss.data), i)
        i += 1

        if i % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_output = model(val)
                f1 = f1_score(torch.index_select(labels, 0, val).detach().cpu().data.numpy(),
                              val_output.detach().cpu().data.numpy().argmax(axis=1), average="micro")
                print("Validation F1:", f1)
                print("Average batch time:", np.mean(times))
                writer.add_scalar('F1/validation', f1, i)
                if f1 > best_val:
                    best_val = f1
                    torch.save({'epoch': epoch, 'model': model.state_dict(), 'optimizer': optimizer.state_dict(),
                                'f1': f1}, bm_ckpt_filename)

    print(f'Best model:')
    best_model = TransformerModel(feat_data, adj_list, g_adj_list, lap_enc, config.layers, config.hid_dim,
                             config.n_head, config.hid_dim, config.dropout, num_classes=7, sampler=config.sampler,
                             num_samples=config.num_samples, g_nx=g_nx)
    best_model_dict = torch.load(bm_ckpt_filename)
    best_model.load_state_dict(best_model_dict['model'])
    optimizer.load_state_dict(best_model_dict['optimizer'])
    print(f"=> loaded checkpoint {bm_ckpt_filename} (epoch {best_model_dict['epoch']}")
    best_model.to(device)
    best_model.eval()
    with torch.no_grad():
        val_output = best_model(val)
        val_f1 = f1_score(torch.index_select(labels, 0, val).detach().cpu().data.numpy(),
                          val_output.detach().cpu().data.numpy().argmax(axis=1), average="micro")
        print("Validation F1:", val_f1)
        print("Average batch time:", np.mean(times))
        writer.add_scalar('Best epoch F1/val', val_f1, i-1)

        test_output = best_model(test)
        test_f1 = f1_score(torch.index_select(labels, 0, test).detach().cpu().data.numpy(),
                           test_output.detach().cpu().data.numpy().argmax(axis=1), average="micro")
        print("Test F1:", test_f1)
        print("Average batch time:", np.mean(times))
        writer.add_scalar('F1/test', test_f1, i-1)
        writer.add_scalar('Early stopping epoch', best_model_dict['epoch'], i-1)
    writer.close()


def set_seed(seed):
    """
    Set numpy and pytorch random seeds

    :param seed:
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def set_deterministic(b):
    """
    Make pytorch deterministic for p

    :param b:
    """
    torch.backends.cudnn.benchmark = not b
    torch.backends.cudnn.deterministic = b
    # torch.use_deterministic_algorithms(b)


if __name__ == '__main__':
    print(os.getcwd())
    set_seed(1)
    # set_deterministic(True)
    main()
