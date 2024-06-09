import numpy as np

class AtasiNet : 
	def __init__(self,A) -> None:
		self.A = A
		return

	def compute_W(self,max_iter=1000, tol=1e-6, alpha=0.01):
		A = self.A
		M, N = A.shape
		W = np.random.rand(M, N) + 1j * np.random.rand(M, N)  # Initial guess
		Lambda = np.zeros((M, N), dtype=complex)  # Lagrange multipliers

		for iteration in range(max_iter):
			W_prev = W.copy()

			# Update W
			for i in range(N):
				A_i = A[:, i]
				Lambda_i = Lambda[:, i]

				# Gradient of the objective function
				grad = 2 * (W.T @ A) @ A[:, i] + Lambda_i

				# Update W column by column
				W[:, i] -= alpha * grad
				# Enforce the constraint
				W[:, i] = W[:, i] / (np.dot(W[:, i], A_i.conj()) + 1e-10)  # Normalization to ensure constraint

			# Update Lagrange multipliers
			for i in range(N):
				Lambda[:, i] += alpha * (np.dot(W[:, i].T, A[:, i]) - 1)

			# Check convergence
			if np.linalg.norm(W - W_prev) < tol:
				print(f"Convergence reached after {iteration} iterations.")
				break

		return W


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