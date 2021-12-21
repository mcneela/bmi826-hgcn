from data import dataset
from basic_models import LogReg, MultiMembershipClassifier

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GAT
from torch_geometric.loader import DataLoader

# GCN stuff
N = len(dataset)
avg_acc, avg_p, avg_r = 0.0, 0.0, 0.0
for i in range(N):
	print(f"Running graph {i}")
	in_channels = dataset[i].x.size(1)
	hidden_channels = in_channels // 2
	num_layers = 4
	out_channels = dataset[i].y.size(1)
	model = GAT(in_channels, hidden_channels, num_layers, out_channels)

	ppi_loader = DataLoader(dataset[i], batch_size=1)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
	loss_fn = nn.BCELoss()
	num_epochs = 18 

	model.train()
	for k in range(num_epochs):
		total_loss = 0
		pred = F.sigmoid(model(dataset[i].x, dataset[i].edge_index))
		loss = loss_fn(pred, dataset[i].y)
		loss.backward()
		optimizer.step()

	model.eval()
	pred = F.sigmoid(model(dataset[i].x, dataset[i].edge_index))
	pred[pred < 0.5] = 0
	pred[pred >= 0.5] = 1
	size = dataset[i].y.size()
	n = size[0] * size[1]
	miss = torch.count_nonzero(dataset[i].y - pred)
	acc = (n - miss) / n
	avg_acc += acc / N

	tp = torch.count_nonzero(dataset[i].y)
	fp = len(torch.where(torch.logical_and(dataset[i].y == 0, pred == 1))[0])
	fn = len(torch.where(torch.logical_and(dataset[i].y == 1, pred == 0))[0])
	precision = tp / (tp + fp)
	avg_p += precision / N

	recall = tp / (tp + fn)
	avg_r += recall / N
