from pyKoLesky.cholesky import *
import torch
import math

torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    nu = 5/2
    lengthscale = 0.3  # Lengthscale
    h = 0.05  # Grid spacing
    grid_size = int(1 / h)  # Number of points along one dimension

    # Generate equidistributed points in [0, 1]^2
    x = torch.linspace(0, 1, grid_size)
    x= torch.cartesian_prod(x, x)

    def matern_kernel(X1, X2, nu, lengthscale, sigma=1.0):
        # Compute pairwise distances
        dist = torch.cdist(X1, X2, p=2)  # Euclidean distance

        # Common term sqrt(2p+1) / rho
        p = int(nu - 1 / 2)  # Get p from nu = p + 1/2
        sqrt_term = math.sqrt(2 * p + 1) / lengthscale
        exp_term = torch.exp(-sqrt_term * dist)

        block = sqrt_term * dist

        term1 = 1
        term2 = block
        term3 = block ** 2 / 3
        polynomial = term1 + term2 + term3
        K = sigma ** 2 * polynomial * exp_term
        return K


    # Formulate the covariance matrix K
    Theta = matern_kernel(x, x, nu, lengthscale)


    # Sparsity control parameter
    rho = 8

    Perm, lengths = maximin(x)                              #   Maxmin ordering
    sparsity = sparsity_pattern(x[Perm], lengths, rho)      #   Create Sparsity Pattern
    U_sparse = sparse_cholesky(Theta, Perm, sparsity)       #   torch.sparse_coo_tensor structure sparse matrix

    # Convert sparse tensor to dense for validation
    U_dense = U_sparse.to_dense()

    invPerm = torch.argsort(Perm)                           # Compute the inverse ordering

    Theta_ordered = Theta[Perm][:, Perm]
    mtx = U_dense.T @ Theta_ordered @ U_dense
    klerror = 1 / 2 * (-torch.log(torch.linalg.det(mtx)) + torch.trace(mtx) - len(mtx))

    # Output results
    print("\nSparse Cholesky Factor (U):\n", U_dense)
    print("\nReconstruction KL error of Theta^-1:", klerror)


