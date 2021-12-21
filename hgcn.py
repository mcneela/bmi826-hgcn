from data import dataset
from basic_models import LogReg, MultiMembershipClassifier
from layers import HyperbolicGCN
from geometry import HyperboloidModel

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GCN 
from torch_geometric.loader import DataLoader
from radam import RiemannianAdam

# GCN stuff
in_channels = dataset[0].x.size(1)
hidden_channels = in_channels // 2
num_layers = 4
out_channels = dataset[0].y.size(1)
manifold = HyperboloidModel(1.0)
model = HyperbolicGCN(manifold, in_channels, out_channels, 1.0, 1.0, .2, F.sigmoid)
ppi_loader = DataLoader(dataset[0], batch_size=1)
optimizer = RiemannianAdam(model.parameters(), lr=0.01)
loss_fn = nn.BCELoss()
num_epochs = 18 

model.train()
for k in range(num_epochs):
	print(k)
	total_loss = 0
	pred = F.sigmoid(model((dataset[0].x, dataset[0].edge_index))[0])
	pred = torch.nan_to_num(pred, nan=0.0)
	loss = loss_fn(pred, dataset[0].y)
	print(loss)
	loss.backward()
	optimizer.step()

model.eval()
pred = F.sigmoid(model(dataset[0].x, dataset[0].edge_index))
print(pred)
pred[pred < 0.5] = 0
pred[pred >= 0.5] = 1
size = dataset[0].y.size()
n = size[0] * size[1]
miss = torch.count_nonzero(dataset[0].y - pred)
acc = (n - miss) / n