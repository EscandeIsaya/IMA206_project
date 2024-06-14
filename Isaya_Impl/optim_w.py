import numpy as np
from scipy.optimize import minimize

class WOptimizer:
    
    def __init__(self, A):
        self.A = A
        self.iteration = 0

    def objective(self, W_flat, A):
        # Reshape W_flat to matrix W
        W = W_flat.reshape(A.shape[0], A.shape[1])
        # Calculate the norm of W.T @ A
        norm_value = np.linalg.norm(W.T @ A)
        return norm_value

    def constraint(self, W_flat, A):
        # Reshape W_flat to matrix W
        W = W_flat.reshape(A.shape[0], A.shape[1])
        # Constraint for each row i: W[i].T @ A[i] == 1
        constraints = [W[i] @ A[i] - 1 for i in range(A.shape[0])]
        return np.array(constraints)
    
    def callback(self, W_flat):
        # Reshape W_flat to matrix W
        self.iteration += 1
        W = W_flat.reshape(self.A.shape[0], self.A.shape[1])
        print(f"Norm of W.T@A at iteration {self.iteration} : {np.linalg.norm(W.T @ self.A)}")

    def minimize_pseudo_inverse(self, A=None):
        if A is None:
            A = self.A
        # Initial guess for W (randomly initialized)
        W_initial = np.random.rand(A.shape[1], A.shape[0]).flatten()

        # Constraints
        cons = [{'type': 'eq', 'fun': self.constraint, 'args': (A,)}]

        # Minimize
        result = minimize(self.objective, W_initial, args=(A,), constraints=cons, method='SLSQP',callback=self.callback)

        # Reshape the solution back to matrix form
        W_optimized = result.x.reshape(A.shape[1], A.shape[0])
        self.iteration = 0
        return W_optimized.T
