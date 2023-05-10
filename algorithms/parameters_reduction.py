import torch.nn as nn
import torch
import math
import numpy as np
import torch.nn.functional as F
import torch.nn.utils.prune as prune
from prettytable import PrettyTable
from numpy.linalg import svd
from torch.linalg import matrix_rank
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FractionalConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(FractionalConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Initializing the learnable parameters
        self.A = torch.randn(self.out_channels, self.in_channels)
        # Constrain for the parameter A,
        # as the gamma function is not defined for negative integers or zero.
        self.A = nn.Parameter(torch.abs(self.A) + 1)

        self.sigma = nn.Parameter(torch.randn(self.out_channels, self.in_channels).abs())
        self.x0 = nn.Parameter(torch.randn(self.out_channels, self.in_channels))
        self.y0 = nn.Parameter(torch.randn(self.out_channels, self.in_channels))
        self.a = nn.Parameter(torch.rand(self.out_channels, self.in_channels) * 2)
        self.b = nn.Parameter(torch.rand(self.out_channels, self.in_channels) * 2)

        self.weights = None
        self.compute_weights()

    def compute_weights(self):
        weights = torch.zeros(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
        dx = self._fractional_derivative(self.a, self.A, self.sigma, self.x0)
        dy = self._fractional_derivative(self.a, self.A, self.sigma, self.y0)

        kernel = torch.einsum('abc,abd->acbd', dx, dy)
        weights = kernel.reshape(dx.shape[0], dx.shape[1], dx.shape[2], dy.shape[2])

        self.weights = weights.to(device)

    def forward(self, x):
        self.weights = self.weights.clone().detach()
        x = x.to(device)
        out = F.conv2d(x, self.weights, stride=self.stride, padding=self.padding)
        return out

    def _fractional_derivative(self, alpha, A, sigma, x0):
        N = 15
        h = torch.tensor(1 / self.kernel_size).repeat(self.out_channels, self.in_channels, self.kernel_size)

        def gamma_func(a):
            return torch.exp(torch.lgamma(a))

        def G(x):
            return torch.exp(-(torch.square(x-x0))/torch.square(sigma))
        
        def f(x, n):
            return (gamma_func(alpha + 1) * G(x)) / ((-1)**n * gamma_func(n+1) * gamma_func(1-n+alpha))

        dx = torch.zeros(self.out_channels, self.in_channels, self.kernel_size)
        for x in range(1, self.kernel_size+1):
            x = torch.tensor(x).repeat(self.out_channels, self.in_channels)
            sum_term = 0
            for n in range(N+1):
                n = torch.tensor(n).repeat(self.out_channels, self.in_channels)
                sum_term += f(x, n)
            dx[..., x-1] = sum_term
        
        dx = dx.to(device)
        h = h.to(device)
        A = A.unsqueeze(2).repeat(1, 1, self.kernel_size)
        A = A.to(device)
        dx = (A / h) * dx
        return dx


class PrunedConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True, sparsity=0.5):
        super(PrunedConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        self.sparsity = sparsity

        self.prune()

    def forward(self, x):
        return self.conv(x)

    def prune(self):
        prune.random_unstructured(self.conv, name="weight", amount=0.3)

    def get_sparsity(self):
        return self.sparsity

    def set_sparsity(self, sparsity):
        self.sparsity = sparsity
        self.prune()


class LowRankConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, r=1, method='log', decomposition='cur'):
        super(LowRankConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.r = r

        self.conv = nn.Conv2d(
                in_channels=self.in_channels,              
                out_channels=self.out_channels,            
                kernel_size=self.kernel_size,           
                stride=self.stride,                   
                padding=self.padding,                  
            )

        if decomposition == 'cur':
            if method == 'constant':
                c = 0.99
            elif method == 'log':
                c = np.log(self.conv.weight.shape[-1])
            self.conv.weight.data = torch.from_numpy(self.cur_low_rank(self.conv.weight, c=c, r=r))
        elif decomposition == 'svd':
            self.conv.weight.data = torch.from_numpy(self.traditional_low_rank(self.conv.weight, r=r))

    def forward(self, x):
        return self.conv(x)

    def get_weights(self):
        return self.conv.weight

    def traditional_low_rank(self, A, r):
        A = A.detach().numpy()
        n1, n2, n3, n4 = A.shape
        A = A.reshape((n1*n2*n3, n4))

        # Compute the SVD of A
        U, s, Vt = svd(A, full_matrices=False)

        # Truncate the SVD to the target rank
        U = U[:, :r]
        s = s[:r]
        Vt = Vt[:r, :]

        # Compute the low-rank approximation
        A_approx = np.dot(U, np.dot(np.diag(s), Vt))

        # Reshape the low-rank approximation to a 4D weight matrix
        A_approx = A_approx.reshape((n1, n2, n3, n4))
        return A_approx

    def cur_low_rank(self, A, c, r):
        A = A.detach().numpy()
        n1, n2, n3, n4 = A.shape
        A_2d = A.reshape((n1*n2*n3, n4))
        # m, n = A_2d.shape

        curr_r = np.linalg.matrix_rank(A_2d)
        
        # Computing C
        def choose_col_by_prob(A):
            U, s, Vt = svd(A, full_matrices=False)

            leverage_scores = np.linalg.norm(Vt[:curr_r], axis=0) ** 2 / curr_r
            column_probabilities = np.minimum(c, leverage_scores) / np.sum(np.minimum(c, leverage_scores))
            selected_columns = np.random.choice(A.shape[1], curr_r, replace=False, p=column_probabilities)
            
            return A[:, selected_columns]

        C = choose_col_by_prob(A_2d)
        R = choose_col_by_prob(np.transpose(A_2d))
        C_pinv = np.linalg.pinv(C)
        R_pinv = np.linalg.pinv(R)

        U = C_pinv @ A_2d @ R_pinv

        # Truncate the CUR to the target rank
        C = C[:, :r]
        U = U[:r, :r]
        R = R[:r, :]

        A_approx = C @ U @ R

        # Reshape the low-rank approximation to a 4D weight matrix
        A_approx = A_approx.reshape((n1, n2, n3, n4))
        return A_approx
