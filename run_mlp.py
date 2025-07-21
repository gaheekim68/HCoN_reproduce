import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils import dotdict, load_custom_data
import datasets
import edgnn_utils

# ───────────────────────────────
# simple MLP model define
# ───────────────────────────────
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# ───────────────────────────────
# training function 
# ───────────────────────────────
def train_simple_mlp(X, labels, idx_train, idx_val, idx_test,
                     hidden_dim=64, lr=0.01, weight_decay=5e-4, epochs=200):
    if not isinstance(X, torch.Tensor):
        X = torch.from_numpy(X.toarray()).float()
    if not isinstance(labels, torch.Tensor):
        labels = torch.from_numpy(labels).long() 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X, labels = X.to(device), labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    model = SimpleMLP(X.shape[1], hidden_dim, labels.max().item() + 1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        logits = model(X)
        loss = loss_fn(logits[idx_train], labels[idx_train])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            model.eval()
            val_acc = (logits[idx_val].argmax(1) == labels[idx_val]).float().mean().item()
            print(f"[Epoch {epoch}] Loss: {loss.item():.4f}, Val Acc: {val_acc:.4f}")

    model.eval()
    test_acc = (model(X)[idx_test].argmax(1) == labels[idx_test]).float().mean().item()
    print(f"\n✅ Test Accuracy: {test_acc:.4f}")
    return test_acc

# ───────────────────────────────
# Main
# ───────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id', type=str, default='0')
    parser.add_argument('--dataname', type=str, default='cora')
    parser.add_argument('--data_dir', type=str, default='../dataset/cocitation/cora')
    parser.add_argument('--raw_data_dir', type=str, default='../edgnn-hypergraph-dataset/cocitation/cora')
    parser.add_argument('--train_prop', type=float, default=0.5)
    parser.add_argument('--valid_prop', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--wd', type=float, default=1e-4)
    parser.add_argument('--hd', type=int, default=64)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id

    # 데이터셋 로딩
    data_obj = datasets.HypergraphDataset(
        root=args.data_dir,
        name=args.dataname,
        path_to_download=args.raw_data_dir).data

    split_idx_lst = [
        edgnn_utils.rand_train_test_idx(data_obj.y, args.train_prop, args.valid_prop)
        for _ in range(10)
    ]

    H, X, Y, labels, idx_train_list, idx_val_list, idx_test_list = load_custom_data(data_obj, split_idx_lst)

    acc_test = []
    for i in range(10):
        idx_train = torch.tensor(np.array(idx_train_list[i]).astype(np.int64), dtype=torch.long)
        idx_val = torch.tensor(np.array(idx_val_list[i]).astype(np.int64), dtype=torch.long)
        idx_test = torch.tensor(np.array(idx_test_list[i]).astype(np.int64), dtype=torch.long)

        int_labels = np.argmax(labels, axis=1)

        test_acc = train_simple_mlp(X, int_labels, idx_train, idx_val, idx_test,
                                    hidden_dim=args.hd,
                                    lr=args.lr,
                                    weight_decay=args.wd,
                                    epochs=500)
        acc_test.append(test_acc)
        print(f"Trial {i + 1}: Test Acc = {test_acc:.4f}")

    acc_test = np.array(acc_test) * 100
    print("Average accuracy: {:.2f}±{:.2f}".format(acc_test.mean(), acc_test.std()))
