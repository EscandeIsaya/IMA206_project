import numpy as np
from optim_w import WOptimizer

class AtasiNet : 
	def __init__(self,A) -> None:
		self.A = A
		self.optimizer = WOptimizer(A)
		return

	def compute_w(self,A = None) : 
		if A == None : 
			optimizer = self.optimizer
		else : 
			optimizer = WOptimizer(A)
		return optimizer.minimize_pseudo_inverse()

	def run_algorithm(self, y, A, mu, beta, K):
		# Initialize variables
		ymax = np.max(np.abs(y))
		epsilon = 0.005 * ymax
		beta_0 = 0.01
		gamma_0 = np.zeros(A.shape[1])
		D_0 = np.zeros_like(y)

		# Pre-compute W
		W = self.compute_W()

		# Initialize variables for the loop
		gamma_k = gamma_0
		D_k = D_0
		k = 0

		for k in range(K):
			# Update Wk
			Wk = beta[k] * W
			# Update zk
			z_k = gamma_k - Wk @ D_k
			# Update Dk
			D_k = A @ gamma_k - y
			# Update θk
			theta_k = mu[k] / (np.abs(z_k) + epsilon)
			# Update γk+1
			gamma_k = self.eta_threshold(z_k, theta_k)

		# Final reflectivity profile
		gamma = gamma_k
		return gamma, theta_k
	
	def __call__(self, y, A, mu, beta, K):
		return self.run_algorithm(y, A, mu, beta, K)