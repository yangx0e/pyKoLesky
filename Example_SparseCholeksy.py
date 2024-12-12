from pyKoLesky.cholesky import *
import torch

torch.set_default_dtype(torch.float64)

if __name__ == '__main__':
    A = torch.tensor([[4.0, 1.0, 0.5, 0.0, 0.0],
                      [1.0, 3.0, 0.5, 0.0, 0.0],
                      [0.5, 0.5, 2.0, 1.0, 0.0],
                      [0.0, 0.0, 1.0, 3.0, 0.5],
                      [0.0, 0.0, 0.0, 0.5, 1.0]])
    Theta = A @ A.T  # Ensure positive definiteness

    # Input points
    x = torch.linspace(0, 1, 6)[1:][:, None]

    # Sparsity control parameter
    rho = 3

    Perm, lengths = maximin(x)                              #   Maxmin ordering
    sparsity = sparsity_pattern(x[Perm], lengths, rho)      #   Create Sparsity Pattern
    U_sparse = sparse_cholesky(Theta, Perm, sparsity)       #   torch.sparse_coo_tensor structure sparse matrix

    # Convert sparse tensor to dense for validation
    U_dense = U_sparse.to_dense()

    invPerm = torch.argsort(Perm)                           # Compute the inverse ordering
    # Validate L_sparse satisfies Theta^-1 = UU^T
    Theta_inverse_reconstructed = (U_dense @ U_dense.T)[invPerm][:, invPerm]    # Reconstruct the covariance matrix
    Theta_inverse = torch.inverse(Theta)

    # Condition number before preconditioning
    cond_before = torch.linalg.cond(Theta)

    # Precondition Theta
    Theta_preconditioned = U_dense.T @ Theta[Perm][:, Perm] @ U_dense #L_dense @ Theta @ L_dense.T

    # Condition number after preconditioning
    cond_after = torch.linalg.cond(Theta_preconditioned)

    # Output results
    print("\nSparse Cholesky Factor (U):\n", U_dense)

    # Check correctness
    reconstruction_error = torch.linalg.norm(Theta_inverse - Theta_inverse_reconstructed, ord='fro')
    print("\nReconstruction error of Theta^-1:", reconstruction_error.item())


