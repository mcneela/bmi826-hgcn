import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.utils import to_dense_adj

from geometry import HyperboloidModel

class DenseAttn(nn.Module):
	def __init__(self, in_dim, dropout):
		super(DenseAttn, self).__init__()
		self.dropout = dropout
		self.in_dim = in_dim
		self.linear = nn.Linear(2 * in_dim, 1, bias=True)

	def forward(self, x, adj):
		n = x.size(0)
		x_l = torch.unsqueeze(x, 1)
		x_l = x_l.expand(-1, n, -1)
		x_r = torch.unsqueeze(x, 0)
		x_r = x_r.expand(n, -1, -1)

		x_cat = torch.cat((x_l, x_r), dim=2)
		attn_adj = self.linear(x_cat).squeeze()
		attn_adj = F.sigmoid(attn_adj)
		adj = to_dense_adj(adj, max_num_nodes=attn_adj.size(0))
		if len(adj.size()) == 3:
			adj = adj.squeeze(0)
		# if adj.size() != attn_adj.size():
		# 	diff = attn_adj.size(0) - adj.size(0)
		# 	for i in range(diff):
		# 		adj = torch.cat((adj, torch.tensor([[0] * adj.size(1)])), 0)
		# 	idx_vec = torch.tensor([adj.size(1) + j for j in range(diff)])
		# 	one_hot = F.one_hot(idx_vec).t()
		# 	adj = torch.cat((adj, one_hot), 1)
		print(adj.size())
		attn_adj = torch.mul(adj, attn_adj)
		return attn_adj

class HyperbolicLinear(nn.Module):
	def __init__(self, manifold, in_dim, out_dim, dropout=0.5):
		super(HyperbolicLinear, self).__init__()
		self.manifold = manifold
		self.in_dim= in_dim
		self.out_dim = out_dim
		self.dropout = dropout
		self.W = nn.Parameter(torch.Tensor(out_dim, in_dim))
		self.b = nn.Parameter(torch.Tensor(out_dim))
		self._init_params()

	def _init_params(self):
		nn.init.xavier_uniform_(self.W, gain=math.sqrt(2))
		nn.init.constant_(self.b, 0)

	def forward(self, x):
		# x = x.unsqueeze(-1)
		dropout_applied = F.dropout(self.W, self.dropout, training=self.training)
		Wx = self.manifold.matvec_multiply(dropout_applied, x)
		print(" Wx ")
		print(Wx)
		h = self.manifold.proj(Wx)
		o = torch.zeros(h.size(1))
		o[0] = torch.sqrt(self.manifold.K)
		pt = self.manifold.parallel_transport(o, h, self.b)
		print(" Parallel Transport ")
		print(pt)
		plus_bias = self.manifold.exp_map(h, pt)
		print(" Bias addition ")
		print(plus_bias)
		return plus_bias

class HyperbolicActivation(nn.Module):
	def __init__(self, manifold, in_curv, out_curv, act):
		super(HyperbolicActivation, self).__init__()
		self.in_curv = in_curv
		self.out_curv = out_curv
		self.act = act
		self.manifold_in = HyperboloidModel(in_curv)
		self.manifold_out = HyperboloidModel(out_curv)

	def forward(self, x):
		o = torch.zeros(x.size(1))
		o[0] = torch.sqrt(self.manifold_in.K)
		xt = self.act(self.manifold_in.log_map(o, x))
		o = torch.zeros(xt.size(1))
		o[0] = torch.sqrt(self.manifold_out.K)
		xt = self.manifold_out.proj_tan(o, xt)
		return self.manifold_out.proj(self.manifold_out.exp_map(o, xt))

class HyperbolicAggregation(nn.Module):
	def __init__(self, manifold, in_dim, dropout, use_attn=True, local_agg=True):
		super(HyperbolicAggregation, self).__init__()
		self.manifold = manifold
		self.in_dim = in_dim
		self.dropout = dropout
		self.local_agg = local_agg
		self.use_attn = use_attn
		if self.use_attn:
			self.attn = DenseAttn(in_dim, dropout)

	def forward(self, x, adj):
		o = torch.zeros(x.size())
		o[0] = torch.sqrt(self.manifold.K)
		x_tan = self.manifold.log_map(o, x)
		if self.use_attn:
			if self.local_agg:
				x_loc_tan = []
				for i in range(x.size(0)):
					x_loc_tan.append(self.manifold.log_map(x[i], x))
				x_loc_tan = torch.stack(x_loc_tan, dim=0)
				adj_attn = self.attn(x_tan, adj)
				support_t = torch.sum(adj_attn.unsqueeze(-1) * x_loc_tan, dim=1)
				# support_t = support_t.squeeze(0)
				# print(support_t)
				# print(support_t.size())
				# print(x)
				# print(x.size())
				output = self.manifold.proj(self.manifold.exp_map(x, support_t))
				return output
			else:
				adj_attn = self.attn(x_tan, adj)
				support_t = torch.matmul(adj_attn, x_tan)
		else:
			support_t = torch.spmm(adj, x_tan)
		o = torch.zeros(support_t.size())
		o[0] = torch.sqrt(self.manifold.K)
		output = self.manifold.proj(self.manifold.exp_map(o, support_t))
		return output

class HNNLayer(nn.Module):
	def __init__(self, manifold, in_dim, out_dim, in_curv, out_curv, dropout, act, use_bias=True):
		super(HNNLayer, self).__init__()
		self.linear = HyperbolicLinear(manifold, in_dim, out_dim, dropout=dropout)
		self.activation = HyperbolicActivation(manifold, in_curv, out_curv, act)

	def forward(self, x):
		h = self.linear(x)
		h = self.activation(x)
		return h

class FermiDiracDecoder(nn.Module):
    def __init__(self, r, t):
        super(FermiDiracDecoder, self).__init__()
        self.r = r
        self.t = t

    def forward(self, dist):
        probs = 1. / (torch.exp((dist - self.r) / self.t) + 1.0)
        return probs

class HyperbolicGCN(nn.Module):
	def __init__(self, manifold, in_dim, out_dim, c_in, c_out, dropout, act,
				use_bias=True, use_attn=True, local_agg=True):
		super().__init__()
		self.linear = HyperbolicLinear(manifold, in_dim, out_dim, dropout=dropout)
		self.agg = HyperbolicAggregation(manifold, out_dim, dropout, use_attn=use_attn, local_agg=local_agg)
		self.activation = HyperbolicActivation(manifold, c_in, c_out, act)

	def forward(self, input):
		x, adj = input
		h = self.linear(x)
		h = self.agg(h, adj)
		h = self.activation(h)
		output = h, adj
		return output

if __name__ == '__main__':
	manifold = HyperboloidModel(1)
	linear = HyperbolicLinear(manifold, 5, 2)
	x = torch.tensor([1., 2., 3., 4., 5.])
	print(linear(x))