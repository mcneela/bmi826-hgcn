import torch
import torch.nn as nn
import torch.nn.functional as F

class LogReg(nn.Module):
	def __init__(self, input_dim, num_classes):
		super(LogReg, self).__init__()
		self.input_dim = input_dim
		self.num_classes = num_classes
		self.linear = nn.Linear(input_dim, num_classes)
		self._init_weights()

	def _init_weights(self):
		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.fill_(0.0)

	def forward(self, x):
		logits = self.linear(x)
		return logits

class MultiMembershipClassifier(nn.Module):
	def __init__(self, input_dim, num_classes):
		super(MultiMembershipClassifier, self).__init__()
		self.input_dim = input_dim
		self.num_classes = num_classes
		# self.classifiers = [LogReg(input_dim, 2) for n in range(num_classes)]
		self.linear = nn.Linear(input_dim, num_classes)
		# self.model = nn.Sequential(*self.classifiers)

	def forward(self, x):
		logits = self.model(x)
		return logits