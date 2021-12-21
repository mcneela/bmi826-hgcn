from data import dataset
from basic_models import LogReg, MultiMembershipClassifier

import torch
import torch.nn as nn
from torch_geometric.nn.models import Node2Vec
from torch_geometric.loader import DataLoader

# Node2Vec stuff
n2v = Node2Vec(dataset[0].edge_index, 256, 5, 3)
loader = n2v.loader(batch_size=64, shuffle=True, num_workers=4)
optimizer = torch.optim.Adam(list(n2v.parameters()), lr=0.01)

# Logistic regression stuff
# input_dim = dataset[0].x.size(1)
num_classes = dataset[0].y.size(1) 
ppi_loader = DataLoader(dataset[0], batch_size=64, shuffle=True)
logreg = LogReg(256, num_classes)
logreg_optim = torch.optim.Adam(logreg.parameters(), lr=0.05)
loss_fn = nn.BCEWithLogitsLoss()

def train_node2vec(num_epochs):
	n2v.train()
	for k in range(num_epochs):
		total_loss = 0
		for pos_rw, neg_rw in loader:
			optimizer.zero_grad()
			loss = n2v.loss(pos_rw, neg_rw)
			loss.backward()
			optimizer.step()
			total_loss += loss.item()
		print(f"Epoch: {k}, Loss: {total_loss}")
	return total_loss / len(loader)

def train_logreg(num_epochs):
	logreg.train()
	for k in range(num_epochs):
		total_loss = 0
		embed = n2v(dataset[0].edge_index)
		pred = logreg(embed[0])
		loss = loss_fn(pred, dataset[0].y)
		loss.backward()
		logreg_optim.step()

	logreg.eval()
	embed = n2v(dataset[0].edge_index)
	pred = logreg(embed[0])
	pred_01 = pred[pred < 0.5] = 0
	pred_01 = pred_01[pred >= 0.5] = 1
	size = dataset.y.size()
	n = size[0] * size[1]
	miss = torch.count_nonzero(dataset.y - pred_01)
	acc = (n - miss) / n
	return acc


if __name__ == '__main__':
	print("Training node2vec")
	train_node2vec(3)
	train_logreg(3)

