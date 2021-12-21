import torch

class StableArccosh(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		x = x.clamp(min=1.0 + 1e-15)
		ctx.save_for_backward(x)
		z = x.double()
		return (z + torch.sqrt(z.pow(2) - 1)).clamp_min_(1e-15).log_().to(x.dtype)

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		return grad_output / (input ** 2 - 1) ** 0.5

class StableArcsinh(torch.autograd.Function):
	@staticmethod
	def forward(ctx, x):
		ctx.save_for_backward(x)
		z = x.double()
		return (z + torch.sqrt_(1 + z.pow(2))).clamp_min_(1e-15).log_().to(x.dtype)
	
	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		return grad_output / (1 + input ** 2) ** 0.5

def arcsinh(x):
	return StableArcsinh.apply(x)

def arccosh(x):
	return StableArccosh.apply(x)

class HyperboloidModel(object):
	def __init__(self, K):
		"""
		- args -
		K: where K is the curvature of the manifold.
		"""
		self.K = torch.tensor(K)
		self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
		self.min_norm = 1e-15
		self.max_norm = 1e6

	def to_hyperbolic(v):
		return torch.tensor([0, *v])

	# def lorentz_product(self, x, y, keepdim=True):
	# 	dot = -x[0] * y[0] + torch.dot(x[1:], y[1:])
	# 	if keepdim:
	# 		dot = dot.view(dot.shape + (1,))
	# 	return dot
	def _lambda_x(self, x):
		c = 1/self.K
		x_sqnorm = torch.sum(x.data.pow(2), dim=-1, keepdim=True)
		return 2 / (1. - c * x_sqnorm).clamp_min(self.min_norm)

	def egrad2rgrad(self, p, dp, c):
		c = 1/self.K
		lambda_p = self._lambda_x(p)
		dp /= lambda_p.pow(2)
		return 

	def lorentz_product(self, x, y, keepdim=True):
		res = torch.sum(x * y, dim=1) - 2 * x[..., 0] * y[..., 0]
		if keepdim:
			res = res.view(res.shape + (1,))
		return res

	def lorentz_norm(self, x):
		dot = self.lorentz_product(x, x)
		return torch.sqrt(torch.clamp(dot, min=self.eps[x.dtype]))

	def dist(self, x, y):
		prod = self.lorentz_product(x, y)
		theta = torch.clamp(-prod / self.K, min=1.0 + self.eps[x.dtype])
		dist = torch.sqrt(self.K) * arccosh(theta)
		return torch.clamp(dist, max=50.0)

	def exp_map(self, x, v):
		norm_v = torch.clamp(self.lorentz_norm(v), min=self.min_norm)
		root_K = torch.sqrt(self.K)
		return (torch.cosh(norm_v / root_K) * x 
			+ root_K * arcsinh(norm_v / root_K) * (v / norm_v))

	def log_map(self, x, y):
		num = (torch.clamp(y + (1/self.K) * 
			self.lorentz_product(x, y) * x, min=1 + self.eps[x.dtype]))
		denom = self.lorentz_norm(num)
		return self.dist(x, y) * (num/denom)

	def parallel_transport(self, v, x, y):
		logxy = self.log_map(x, y)
		logyx = self.log_map(y, x)
		num = self.lorentz_product(logxy, v)
		denom = torch.clamp(self.dist(x, y) ** 2, min=self.min_norm)
		return v - (num / denom) * (logxy + logyx)

	def matvec_multiply(self, W, v):
		o = torch.zeros(v.size(1))
		o[0] = torch.sqrt(self.K)
		proj = self.log_map(o, v)
		Wv = proj @ W.transpose(-1, -2)
		o = torch.zeros(Wv.size(1))
		o[0] = torch.sqrt(self.K)
		return self.exp_map(o, Wv)

	# def proj(self, x):
	# 	first_term = torch.sqrt(1 + torch.norm(x[1:], p=2) ** 2)
	# 	return torch.tensor([first_term, *(x[1:])])
	def proj(self, x):
		d = x.size(-1) - 1
		y = x.narrow(-1, 1, d)
		y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2
		mask = torch.ones_like(x)
		mask[:, 0] = 0
		vals = torch.zeros_like(x)
		vals[:, 0:1] = torch.sqrt(torch.clamp(self.K + y_sqnorm, min=self.eps[x.dtype]))
		return vals + mask * x

	def proj_tan(self, x, v):
		return v + self.lorentz_product(x, v) * x

# 	def proj_tan(self, u, x):
# 		d = x.size(1) - 1
# 		ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
# 		mask = torch.ones_like(u)
# 		print(mask.size())
# 		mask[:, 0] = 0
# 		vals = torch.zeros_like(u)
# 		vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
# 		return vals + mask * u
	
# class HyperboloidModel(object):
# 	"""
# 	Hyperboloid manifold class.

# 	We use the following convention: -x0^2 + x1^2 + ... + xd^2 = -K

# 	c = 1 / K is the hyperbolic curvature. 
# 	"""

# 	def __init__(self, K):
# 		super(HyperboloidModel, self).__init__()
# 		self.name = 'Hyperboloid'
# 		self.eps = {torch.float32: 1e-7, torch.float64: 1e-15}
# 		self.min_norm = 1e-15
# 		self.max_norm = 1e6
# 		self.K = torch.tensor(K)

# 	def lorentz_product(self, x, y, keepdim=True):
# 		res = torch.sum(x * y, dim=-1) - 2 * x[..., 0] * y[..., 0]
# 		if keepdim:
# 			res = res.view(res.shape + (1,))
# 		return res

# 	def lorentz_norm(self, u, keepdim=True):
# 		dot = self.lorentz_product(u, u, keepdim=keepdim)
# 		return torch.sqrt(torch.clamp(dot, min=self.eps[u.dtype]))

# 	def sqdist(self, x, y):
# 		prod = self.lorentz_product(x, y)
# 		theta = torch.clamp(-prod / self.K, min=1.0 + self.eps[x.dtype])
# 		sqdist = self.K * arccosh(theta) ** 2
# 		# clamp distance to avoid nans in Fermi-Dirac decoder
# 		return torch.clamp(sqdist, max=50.0)

# 	def proj(self, x):
# 		d = x.size(-1) - 1
# 		y = x.narrow(-1, 1, d)
# 		y_sqnorm = torch.norm(y, p=2, dim=1, keepdim=True) ** 2 
# 		mask = torch.ones_like(x)
# 		mask[:, 0] = 0
# 		vals = torch.zeros_like(x)
# 		vals[:, 0:1] = torch.sqrt(torch.clamp(self.K + y_sqnorm, min=self.eps[x.dtype]))
# 		return vals + mask * x

# 	def proj_tan(self, u, x):
# 		x = x.unsqueeze(1)
# 		d = x.size(1) - 1
# 		ux = torch.sum(x.narrow(-1, 1, d) * u.narrow(-1, 1, d), dim=1, keepdim=True)
# 		mask = torch.ones_like(u)
# 		mask[:, 0] = 0
# 		vals = torch.zeros_like(u)
# 		vals[:, 0:1] = ux / torch.clamp(x[:, 0:1], min=self.eps[x.dtype])
# 		return vals + mask * u

# 	def proj_tan0(self, u):
# 		narrowed = u.narrow(-1, 0, 1)
# 		vals = torch.zeros_like(u)
# 		vals[:, 0:1] = narrowed
# 		return u - vals

# 	def exp_map(self, u, x):
# 		sqrtK = self.K ** 0.5
# 		normu = self.lorentz_norm(u)
# 		normu = torch.clamp(normu, max=self.max_norm)
# 		theta = normu / sqrtK
# 		theta = torch.clamp(theta, min=self.min_norm)
# 		result = torch.cosh(theta) * x + torch.sinh(theta) * u / theta
# 		return self.proj(result)
		
# 	def log_map(self, x, y):
# 		xy = torch.clamp(self.lorentz_product(x, y) + self.K, max=-self.eps[x.dtype]) - self.K
# 		u = y + xy * x * (1/self.K)
# 		normu = self.lorentz_norm(u)
# 		normu = torch.clamp(normu, min=self.min_norm)
# 		dist = self.sqdist(x, y) ** 0.5
# 		result = dist * u / normu
# 		return self.proj_tan(result, x)

# 	def exp_map0(self, u):
# 		sqrtK = self.K ** 0.5
# 		d = u.size(-1) - 1
# 		x = u.narrow(-1, 1, d).view(-1, d)
# 		x_norm = torch.norm(x, p=2, dim=1, keepdim=True)
# 		x_norm = torch.clamp(x_norm, min=self.min_norm)
# 		theta = x_norm / sqrtK
# 		res = torch.ones_like(u)
# 		res[:, 0:1] = sqrtK * torch.cosh(theta)
# 		res[:, 1:] = sqrtK * torch.sinh(theta) * x / x_norm
# 		return self.proj(res)

# 	def log_map0(self, x):
# 		sqrtK = self.K ** 0.5
# 		d = x.size(-1) - 1
# 		y = x.narrow(-1, 1, d).view(-1, d)
# 		y_norm = torch.norm(y, p=2, dim=1, keepdim=True)
# 		y_norm = torch.clamp(y_norm, min=self.min_norm)
# 		res = torch.zeros_like(x)
# 		theta = torch.clamp(x[:, 0:1] / sqrtK, min=1.0 + self.eps[x.dtype])
# 		res[:, 1:] = sqrtK * arccosh(theta) * y / y_norm
# 		return res

# 	def mobius_add(self, x, y):
# 		u = self.log_map0(y)
# 		v = self.ptransp0(x, u)
# 		return self.exp_map(v, x)

# 	def mobius_matvec(self, m, x):
# 		u = self.log_map0(x)
# 		mu = u @ m.transpose(-1, -2)
# 		return self.exp_map0(mu)

# 	def ptransp(self, x, y, u):
# 		logxy = self.log_map(x, y)
# 		logyx = self.log_map(y, x)
# 		sqdist = torch.clamp(self.sqdist(x, y), min=self.min_norm)
# 		alpha = self.lorentz_product(logxy, u) / sqdist
# 		res = u - alpha * (logxy + logyx)
# 		return self.proj_tan(res, y)

# 	def ptransp0(self, x, u):
# 		sqrtK = self.K ** 0.5
# 		x0 = x.narrow(-1, 0, 1)
# 		d = x.size(-1) - 1
# 		y = x.narrow(-1, 1, d)
# 		y_norm = torch.clamp(torch.norm(y, p=2, dim=1, keepdim=True), min=self.min_norm)
# 		y_normalized = y / y_norm
# 		v = torch.ones_like(x)
# 		v[:, 0:1] = - y_norm 
# 		v[:, 1:] = (sqrtK - x0) * y_normalized
# 		print(u.unsqueeze(0).size())
# 		u = u.unsqueeze(0)
# 		alpha = torch.sum(y_normalized * u[:, 1:], dim=1, keepdim=True) / sqrtK
# 		res = u - alpha * v
# 		return self.proj_tan(res, x)

# 	def to_poincare(self, x):
# 		sqrtK = self.K ** 0.5
# 		d = x.size(-1) - 1
# 		return sqrtK * x.narrow(-1, 1, d) / (x[:, 0:1] + sqrtK)
