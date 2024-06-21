import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from optim_w import WOptimizer

class AtasiNet(nn.Module):
    def __init__(self, A, W=None, K=10, learning_rate=0.01, device=None):
        super(AtasiNet, self).__init__()
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.A = torch.tensor(A, dtype=torch.float32).to(self.device)
        self.W = torch.tensor(W, dtype=torch.float32).to(self.device) if W is not None else None
        self.K = K
        self.mu = nn.Parameter(0.01 * torch.ones(K)).to(self.device)
        self.beta = nn.Parameter(0.01 * torch.ones(K)).to(self.device)

        # Initialize the optimizer and loss function
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def eta_threshold(self, z, theta):
        return torch.sign(z) * torch.maximum(torch.abs(z) - theta, torch.tensor(0.0, device=self.device))

    def compute_w(self, A=None):
        if A is None:
            optimizer = WOptimizer(self.A.cpu().numpy())
        else:
            optimizer = WOptimizer(A)
        self.W = torch.tensor(optimizer.minimize_pseudo_inverse(), dtype=torch.float32).to(self.device)
        return self.W

    def run_algorithm(self, y, A, mu, beta, K):
        # y is expected to be of shape (batch_size, signal_length)
        batch_size = y.size(0)
        epsilon = 0.01
        gamma_k = torch.zeros((batch_size, A.shape[1]), dtype=torch.float32, device=self.device)
        D_k = torch.zeros_like(y, dtype=torch.float32, device=self.device)

        if self.W is None:
            W = self.compute_w()
        else:
            W = self.W

        gamma_l = torch.zeros((batch_size, K, gamma_k.shape[1]), dtype=torch.float32, device=self.device)

        for k in range(K):
            Wk = beta[k] * W
            # print("Wk ", Wk)
            for i in range(gamma_k.shape[1]):
                z_i = gamma_k[:, i] - torch.matmul(D_k, Wk[:, i])  # Shape: (batch_size,)
                # print("z_i ", z_i)
                # print("Wk[:, i] ", Wk[:, i])
                # print("torch.matmul(D_k, Wk[:, i]) ", torch.matmul(D_k, Wk[:, i]) )
                theta_k_i = mu[k] / (torch.abs(z_i) + epsilon)
                # print("theta_i ", theta_k_i)
                gamma_k[:, i] = self.eta_threshold(z_i, theta_k_i)
                # print("gamma_k, i", gamma_k[:, i])
                # print("thresh ", self.eta_threshold(z_i, theta_k_i))
            D_k = torch.matmul(gamma_k, A.T) - y

            gamma_l[:, k, :] = gamma_k

        gamma = gamma_k
        return gamma


    def set_W(self, W):
        self.W = torch.tensor(W, dtype=torch.float32).to(self.device)
        return

    def train(self, train_loader, epochs, A=None):
        if A is None:
            A = self.A
        else:
            A = torch.tensor(A, dtype=torch.float32).to(self.device)

        K = self.K

        for epoch in range(epochs):
            total_loss = 0
            index = 0

            for y, gamma_label in train_loader:
                y = y.to(self.device)
                gamma_label = gamma_label.to(self.device)

                self.optimizer.zero_grad()
                gamma_k = self.run_algorithm(y, A, self.mu, self.beta, K)
                loss = self.loss_fn(gamma_k, gamma_label)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                print(f"At epoch {epoch + 1}/{epochs}, index is {index}/{len(train_loader)} || loss is {loss} ", end='\r')
                index += 1

            avg_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")


    def predict(self, y, A=None):
        if A is None:
            A = self.A
        else:
            A = torch.tensor(A, dtype=torch.float32).to(self.device)

        y = torch.tensor(y, dtype=torch.float32).to(self.device)
        K = self.K
        gamma_k = self.run_algorithm(y, A, self.mu, self.beta, K)
        return gamma_k.detach().cpu().numpy()

    def forward(self, y, A=None):
        return self.predict(y, A)