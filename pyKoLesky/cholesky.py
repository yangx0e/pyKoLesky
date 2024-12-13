from .ordering import *

def __col(Theta, s, nugget = 1e-12):
    """
    compute \Theta_{s, s}^{-1}e_j / \sqrt{e_j^T\Theta_{s, s}^{-1}e_j}

    A simple implementation is to invert \Theta_{s, s} directly using the following code

    m = torch.inverse(Theta[s][:, s])
    return m[:, -1] / torch.sqrt(m[-1, -1])

    However, it has some stability issues. When, \Theta_{s, s} is ill-conditioned, torch.inverse(Theta[s, s]) may not
    give us the accurate solution and e_j^T\Theta_{s, s}^{-1}e_j may be negative and torch.sqrt(m[-1, -1]) produces NaN value.

    Thus, instead, we use standard Cholesky directly on \Theta_{s, s}^{-1} and also adding a small nugget to prevent numerical issue
    """
    try:
        m = Theta[s][:, s] + nugget * torch.eye(len(s))
        L = torch.linalg.cholesky(m)
        ej = torch.zeros(len(s))  # Create a tensor of zeros
        ej[-1] = 1
        v = torch.cholesky_solve(ej.unsqueeze(-1), L, upper=False).squeeze(-1)
        # print(v.shape)
        if v[-1] < 0:
            raise ValueError(f"Negative value encountered for square root: {v[-1]}")
        vl = torch.sqrt(v[-1])
        return v / vl
    except torch.linalg.LinAlgError as e:
        # Handle Cholesky decomposition failure
        print("When doing sparse Cholesky decomposition, Cholesky decomposition of the submatrix \Theta_{s, s} " + f"failed: {e}")
        raise  # Optionally re-raise the exception after logging
    except ValueError as e:
        # Handle negative square root or other value issues
        print("When doing sparse Cholesky decomposition, \Theta_{s, s}[-1, -1] is negative. " + f"Value error encountered: {e}")
        raise  # Optionally re-raise the exception


def __cholesky(Theta, sparsity):
    n = Theta.size(0)
    indptr = torch.cumsum(torch.tensor([0] + [len(sparsity[i]) for i in range(n)]), dim=0)
    total_nonzeros = indptr[-1].item()

    # Prepare storage for sparse matrix components
    data = torch.zeros(total_nonzeros, dtype=torch.float64)
    row_indices = torch.zeros(total_nonzeros, dtype=torch.int64)
    col_indices = torch.zeros(total_nonzeros, dtype=torch.int64)

    for i in range(n):
        s = sorted(sparsity[i])
        col_data = __col(Theta, s)
        start, end = indptr[i], indptr[i + 1]

        data[start:end] = col_data
        row_indices[start:end] = torch.tensor(s, dtype=torch.int64)
        col_indices[start:end] = i

    # Create sparse COO tensor
    indices = torch.vstack([row_indices, col_indices])
    sparse_cholesky = torch.sparse_coo_tensor(indices, data, size=(n, n))
    return sparse_cholesky

def non_zeros(n, sparsity):
    indptr = torch.cumsum(torch.tensor([0] + [len(sparsity[i]) for i in range(n)]), dim=0)
    total_nonzeros = indptr[-1].item()

    row_indices = torch.zeros(total_nonzeros, dtype=torch.int64)
    col_indices = torch.zeros(total_nonzeros, dtype=torch.int64)

    for i in range(n):
        s = sorted(sparsity[i])
        start, end = indptr[i], indptr[i + 1]
        row_indices[start:end] = torch.tensor(s, dtype=torch.int64)
        col_indices[start:end] = i

    # Create sparse COO tensor
    indices = torch.vstack([row_indices, col_indices])
    return indices

def sparse_cholesky(Theta, Perm, sparsity) -> torch.sparse_coo_tensor:
    reordered_Theta = Theta[Perm][:, Perm]
    return __cholesky(reordered_Theta, sparsity)



