import torch
import torch.nn as nn
import torch.nn.functional as F

class NodeClassifier(nn.Module):
	def __init__(self, num_classes, pos_weight=False):
		super(NodeClassifier, self).__init__()
		self.num_classes = num_classes

		if num_classes > 2:
			self.f1_average = 'micro'
		else:
			self.f1_average = 'binary'

		self.weights = torch.tensor([1.] * num_classes)

	def decode(self, h, adj, idx):
		output = self.decoder.decode(h, adj)
		return F.log_softmax(output[idx], dim=1)

	def compute_metrics(self, embeds, data, split):
		idx = data[f'idx_{split}']
		output = self.decode(embeds, data['adj_train_norm'], idx)
		loss = F.nll_loss(output, data['labels'][idx], self.weights)
		return loss
		# acc, f1 = acc_f1(output, data['labels'][idx], average=self.f1_average)
		# metrics = {'loss': loss, 'acc': acc, 'f1': f1}
		# return metrics

	def init_metric_dict(self):
		return {'acc': -1, 'f1': -1}

	def has_improved(self, m1, m2):
		return m1['f1'] < m2['f2']