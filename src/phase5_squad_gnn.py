# src/phase5_squad_gnn.py

"""
Phase 5: Squad Graph Analytics & GNN Modeling (Full-Graph Training)

This script performs:
1. Loads Phase 1 match-level data and Phase 2 player embeddings.
2. Constructs a player co-occurrence graph based on squad (groupId) membership.
3. Aggregates player-level stats as node features.
4. Builds a PyTorch Geometric Data object with train/test masks.
5. Defines and trains a full-graph GraphSAGE model with early stopping.
6. Evaluates RMSE on the test set and saves GNN predictions.
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
from sklearn.metrics import mean_squared_error

# Paths
data_dir     = 'data/processed'
embed_path   = os.path.join(data_dir, 'pubg_phase2_player_embeddings.csv')
matches_path = os.path.join(data_dir, 'pubg_phase1_matches.csv')
output_preds = os.path.join(data_dir, 'pubg_phase5_gnn_preds.csv')

# 1. Load data
emb_df    = pd.read_csv(embed_path)
match_df  = pd.read_csv(matches_path)[['Id','sessionId','groupId','winPlacePerc']]

# 2. Compute target: average winPlacePerc per player
target_df = match_df.groupby('Id')['winPlacePerc'].mean().reset_index()

# 3. Build node features by merging embeddings with target
node_features = emb_df.merge(target_df, on='Id', how='left').fillna(0)

# 4. Build squad-based edges: connect players sharing groupId in a session
edges  = []
grouped = match_df.groupby(['sessionId','groupId'])['Id']
for _, players in grouped:
    ids = players.values
    m   = len(ids)
    if m > 1:
        src = np.repeat(ids, m-1)
        dst = np.concatenate([np.delete(ids, i) for i in range(m)])
        edges.append(np.vstack([src, dst]))
edge_array = np.hstack(edges) if edges else np.empty((2,0), dtype=int)

# Map Id to node index
i2idx    = {pid: idx for idx, pid in enumerate(node_features['Id'])}
src_idx  = [i2idx[p] for p in edge_array[0] if p in i2idx]
dst_idx  = [i2idx[p] for p in edge_array[1] if p in i2idx]
edge_index = torch.tensor([src_idx, dst_idx], dtype=torch.long)

# 5. Prepare PyG Data object
x     = torch.tensor(node_features.filter(like='emb_').values, dtype=torch.float)
y     = torch.tensor(node_features['winPlacePerc'].values, dtype=torch.float)
nodes = x.size(0)

# Create train/test masks
perm       = np.random.RandomState(42).permutation(nodes)
train_size = int(0.8 * nodes)
train_mask = torch.zeros(nodes, dtype=torch.bool)
train_mask[perm[:train_size]] = True
test_mask = torch.zeros(nodes, dtype=torch.bool)
test_mask[perm[train_size:]] = True

data = Data(x=x, edge_index=edge_index, y=y)
data.train_mask = train_mask
data.test_mask  = test_mask

# 6. Define GraphSAGE model
class GNN(torch.nn.Module):
    def __init__(self, in_dim, hid_dim=32):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hid_dim)
        self.conv2 = SAGEConv(hid_dim, hid_dim)
        self.lin   = torch.nn.Linear(hid_dim, 1)
    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        return self.lin(x).squeeze()

model     = GNN(x.size(1))
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
criterion = torch.nn.MSELoss()

# 7. Full-graph training loop with early stopping
best_val = float('inf')
patience = 5
cnt      = 0
for epoch in range(1, 21):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        out_eval   = model(data.x, data.edge_index)
        train_loss = criterion(out_eval[data.train_mask], data.y[data.train_mask]).item()
        val_loss   = criterion(out_eval[data.test_mask],  data.y[data.test_mask]).item()

    print(f'Epoch {epoch}/20, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
    if val_loss < best_val:
        best_val = val_loss
        cnt      = 0
    else:
        cnt += 1
        if cnt >= patience:
            print(f'Early stopping at epoch {epoch}')
            break

# 8. Final evaluation and save predictions
model.eval()
with torch.no_grad():
    preds = model(data.x, data.edge_index).cpu().numpy()
rmse = mean_squared_error(data.y[data.test_mask].cpu().numpy(), preds[data.test_mask], squared=False)
print(f'GNN Test RMSE (full-graph): {rmse:.4f}')

# 9. Save GNN predictions per player
out_df = pd.DataFrame({'Id': node_features['Id'], 'gnn_pred': preds})
out_df.to_csv(output_preds, index=False)
print(f'✔ GNN predictions saved to {output_preds}')
