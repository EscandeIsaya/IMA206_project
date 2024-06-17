import numpy as np
from optim_w import WOptimizer

class AtasiNet : 
	def __init__(self,A) -> None:
		self.A = A
		self.optimizer = WOptimizer(A)
		self.W 
		return

	def eta_threshold(self, z, theta):
		return np.sign(z) * np.maximum(np.abs(z) - theta, 0)

	
	def compute_w(self,A = None) : 
		if A == None : 
			optimizer = self.optimizer
		else : 
			optimizer = WOptimizer(A)
		self.W = optimizer.minimize_pseudo_inverse()
		return self.W

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

	def train(self, Y, A, K, epochs, learning_rate=0.01):
		# Initialize parameters to be learned
		mu = np.random.rand(K)
		beta = np.random.rand(K)
		gamma_list = []


		for epoch in range(epochs):
			for y in Y:
				gamma_k, theta_k = self.run_algorithm(y, A, mu, beta, K)
				gamma_list.append(gamma_k)

				# Compute gradients (this is a simplified approach)
				grad_mu = (gamma_k - mu) / (np.abs(gamma_k) + 1e-5)
				grad_beta = (gamma_k - beta) / (np.abs(gamma_k) + 1e-5)

				# Update parameters
				mu -= learning_rate * grad_mu
				beta -= learning_rate * grad_beta

				# Print out the current loss and parameters (optional)
				loss = np.linalg.norm(A @ gamma_k - y)
				print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Mu: {mu}, Beta: {beta}")

		gamma = np.mean(gamma_list, axis=0)
		return mu, beta, gamma
	
	def __call__(self, y, A, mu, beta, K):
		return self.run_algorithm(y, A, mu, beta, K)
