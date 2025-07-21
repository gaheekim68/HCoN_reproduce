import numpy as np
import scipy.sparse as sp
import torch
import sys
import random
import math
import os
import scipy.io as scio
    

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    

def seed_everything(seed=2021):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
import numpy as np
import torch
from scipy.sparse import csr_matrix
from sklearn.preprocessing import OneHotEncoder

def load_custom_data(data, split_idx_lst):
    """
    Converts PyG-style data and split list into the format expected by run.py
    Args:
        data: PyG Data object with edge_index, x, y, num_hyperedges
        split_idx_lst: list of dicts with 'train' and 'valid' keys
    Returns:
        H, X, Y, labels, idx_train_list, idx_val_list, idx_test_list
    """

    # 1. Incidence matrix H: [num_nodes, num_hyperedges]
    row = data.edge_index[0].cpu().numpy()
    col = data.edge_index[1].cpu().numpy()
    N = data.x.shape[0]
    M = data.num_hyperedges.item()
    H = csr_matrix((np.ones(len(row)), (row, col)), shape=(N, M))

    # 2. Node features X
    X = csr_matrix(data.x.cpu().numpy())

    # 3. Hyperedge features Y (identity matrix if featureless)
    Y = np.eye(M)

    # 4. One-hot labels
    y = data.y.cpu().numpy().reshape(-1, 1)
    enc = OneHotEncoder(sparse=False, categories='auto')
    labels = enc.fit_transform(y)  # shape (N, C)

    # 5. idx_train_list, idx_test_list from split_idx_lst
    idx_train_list = [s['train'].cpu().numpy() for s in split_idx_lst]
    idx_val_list = [s['valid'].cpu().numpy() for s in split_idx_lst]
    idx_test_list = [s['test'].cpu().numpy() for s in split_idx_lst]
    
    idx_train_list = np.array(idx_train_list, dtype=object)
    idx_val_list = np.array(idx_val_list, dtype=object)
    idx_test_list = np.array(idx_test_list, dtype=object)

    return H, X, Y, labels, idx_train_list, idx_val_list, idx_test_list


def load_data(dataset_str):
    data_mat = scio.loadmat("data/{}.mat".format(dataset_str))
    h = data_mat['h']
    X = data_mat['X']
    Y = data_mat['Y']
    labels = data_mat['labels']
    idx_train_list = data_mat['idx_train_list']
    idx_val_list = data_mat['idx_val_list']
    
    X = normalize_features(X)
    Y = normalize_features(Y)
    
    return h, X, Y, labels, idx_train_list, idx_val_list


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    if np.where(rowsum == 0)[0].shape[0] != 0:
        indices = np.where(rowsum == 0)[0]
        for i in indices:
            rowsum[i] = float('inf')
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels) 


def normalize_sparse_hypergraph_symmetric(H):
    
    # rowsum = np.array(H.sum(1))
    # r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    # r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    # D = sp.diags(r_inv_sqrt)
    
    # colsum = np.array(H.sum(0))
    # r_inv_sqrt = np.power(colsum, -1).flatten()
    # r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    # B = sp.diags(r_inv_sqrt)

    # row normalization (Dv^{-1/2})
    rowsum = np.array(H.sum(1)).flatten()
    rowsum[rowsum == 0] = 1e-12  # ⚠️ zero 방지
    r_inv_sqrt = np.power(rowsum, -0.5)
    D = sp.diags(r_inv_sqrt)

    # column normalization (De^{-1})
    colsum = np.array(H.sum(0)).flatten()
    colsum[colsum == 0] = 1e-12  # ⚠️ zero 방지
    c_inv = np.power(colsum, -1)
    B = sp.diags(c_inv)
    
    Omega = sp.eye(B.shape[0])

    hx1 = D.dot(H).dot(Omega).dot(B).dot(H.transpose()).dot(D)
    hx2 = D.dot(H).dot(Omega).dot(B)

    return hx1, hx2

        
        