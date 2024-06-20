import numpy as np
from optim_w import WOptimizer

class AtasiNet : 
	def __init__(self,A) -> None:
		self.A = A
		self.optimizer = WOptimizer(A)
		self.W = None
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
		if self.W is None : 
			W = self.compute_w()
		else :
			W = self.W

		# Initialize variables for the loop
		gamma_k = gamma_0
		gamma_l = np.zeros((K,gamma_k.shape[0]))
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
			gamma_l[k]=gamma_k

		# Final reflectivity profile
		gamma = gamma_k
		return gamma, theta_k

	def train(self, Y, gamma_labels, epochs, A=None, learning_rate=0.01):
		if A is None:
			A = self.A
		K = self.K
		mu = self.mu
		beta = self.beta

		for epoch in range(epochs):
			for y, gamma_label in zip(Y, gamma_labels):
				gamma_k, theta_k = self.run_algorithm(y, A, mu, beta, K)
                
				# Compute loss using labels
				loss = np.linalg.norm(gamma_k - gamma_label)

				# Compute gradients using the loss
				grad_mu = (gamma_k - mu) / (np.abs(gamma_k - gamma_label) + 1e-5)
				grad_beta = (gamma_k - beta) / (np.abs(gamma_k - gamma_label) + 1e-5)

				# Update parameters
				mu -= learning_rate * grad_mu
				beta -= learning_rate * grad_beta

				# Print the current loss and parameters for monitoring
				print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}, Mu: {mu}, Beta: {beta}")

		self.mu = mu
		self.beta = beta

	def predict(self, y, A=None):
		if A is None:
			A = self.A
		K = self.K
		mu = self.mu
		beta = self.beta
		gamma_k, _ = self.run_algorithm(y, A, mu, beta, K)
		return gamma_k
	
	def __call__(self, y, A=None):
		return self.predict(y,A)
