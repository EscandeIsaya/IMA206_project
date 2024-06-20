import numpy as np
from optim_w import WOptimizer

class AtasiNet : 
	def __init__(self,A,K=10,W=None) -> None:
		self.A = A
		self.optimizer = WOptimizer(A)
		self.W = W
		self.K = K
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
		epsilon = 0.005 * np.max(np.abs(y))
		gamma_k = np.zeros(A.shape[1])
		D_0 = np.zeros_like(y)

		# Pre-compute W
		if self.W is None : 
			W = self.compute_w()
		else :
			W = self.W

		# Initialize variables for the loop
		gamma_l = np.zeros((K,gamma_k.shape[0]))
		D_k = D_0
		k = 0

		for k in range(K):
			Wk = beta[k] * W
			for i in range(gamma_k.shape[0]):
				# Update zk
				z_i = gamma_k[i] - Wk[:,i] @ D_k
				# Update θk
				theta_k_i = mu[k] / (np.abs(z_i) + epsilon)
				# Update γk+1
				gamma_k[i] = self.eta_threshold(z_i, theta_k_i)
			D_k = A @ gamma_k - y
			gamma_l[k]=gamma_k

		# Final reflectivity profile
		gamma = gamma_k
		return gamma

	def set_W(self,W):
		self.W = W
		return

	def train(self, Y, gamma_labels, epochs, A=None, learning_rate=0.01):
		if A is None:
			A = self.A
		K = self.K
		mu = 0.1*np.zeros(K)
		beta = 0.01*np.zeros(K)
		index=0

		for epoch in range(epochs):
			index=0
			for y, gamma_label in zip(Y, gamma_labels):
				index += 1
				gamma_k = self.run_algorithm(y, A, mu, beta, K)
                
				# Compute loss using labels
				loss = np.linalg.norm(gamma_k - gamma_label)
                
				# Compute gradients using the loss
				grad_mu = np.zeros_like(mu)
				grad_beta = np.zeros_like(beta)

				for k in range(K):
					# Update gradients for mu and beta
					grad_mu[k] = np.sum((gamma_k - gamma_label) * (gamma_k - mu[k])) / (np.abs(gamma_k - gamma_label).sum() + 1e-5)
					grad_beta[k] = np.sum((gamma_k - gamma_label) * (gamma_k - beta[k])) / (np.abs(gamma_k - gamma_label).sum() + 1e-5)

				# Update parameters
				mu -= learning_rate * grad_mu
				beta -= learning_rate * grad_beta
				if index % 100 == 0:
					print(f"At epoch {epoch + 1}/{epochs}, index is {index}/{len(gamma_labels)} || loss is{loss} || gradients are mu {grad_mu} and beta {grad_beta}", end='\r')

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
