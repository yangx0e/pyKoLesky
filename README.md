# pyKoLesky

`pyKoLesky` is a PyTorch implementation of the sparse Cholesky decomposition algorithm, building upon the algorithms and concepts presented in the Julia package **[KoLesky.jl](https://github.com/f-t-s/KoLesky.jl)**. It implements the sparse Cholesky decomposition algorithm described in the paper *"Sparse Cholesky Factorization by Kullback-Leibler Minimization"* by Florian Schäfer, Matthias Katzfuss, and Houman Owhadi.


The sparse Cholesky algorithm computes a sparse approximation of the inverse Cholesky upper triangular factor $U$ for a dense covariance matrix $\Theta$. It achieves this by minimizing the Kullback-Leibler divergence between the Gaussian distributions $\mathcal{N}(0, \Theta)$ and $\mathcal{N}(0, (UU^T)^{-1})$, while enforcing a sparsity constraint. This method is a generalization of the Vecchia approximation commonly used in spatial statistics. It enables the computation of $\epsilon$-approximate inverse Cholesky factors of $\Theta$ with a space complexity of $\mathcal{O}(N \log(N/\epsilon)^d)$ and a time complexity of $\mathcal{O}(N \log(N/\epsilon)^{2d})$.

## Upcoming Features in Development
- Parallel computing capabilities
- Supernode functionality, as explored in the paper "Sparse Cholesky Factorization by Kullback--Leibler Minimization."

## Development Status

`pyKoLesky` is currently under active development, and new features are being added regularly. While we strive to maintain stability, there may be occasional bugs or incomplete features. 

We greatly appreciate your feedback! If you encounter any issues, please report them by opening an [issue on GitHub](https://github.com/yangx0e/pyKoLesky/issues). Your contributions and suggestions are welcome!



## Related Papers

- Schäfer, Florian, Matthias Katzfuss, and Houman Owhadi. **"Sparse Cholesky Factorization by Kullback--Leibler Minimization."** SIAM Journal on scientific computing 43.3 (2021): A2019-A2046.
- Chen, Yifan, Houman Owhadi, and Florian Schäfer. **"Sparse Cholesky factorization for solving nonlinear PDEs via Gaussian processes."** Mathematics of Computation (2024).

## Installation

You can install `pyKoLesky` from [PyPI](https://pypi.org/project/pyKoLesky/) using `pip`:

```bash
pip install pyKoLesky
```


## Usage

### Basic Usage

Here’s a simple example of how to use `pyKoLesky` to verify the results of maxmin ordering and validate its correctness by labeling the order of reordered points:


```python
from pyKoLesky.ordering import maximin
import torch
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)


n_points = 20
torch.manual_seed(44)  # For reproducibility
x = torch.rand(n_points, 2)  # 10 points in 2D space

# Perform reverse maximin ordering
indexes, _ = maximin(x)

# Plotting the points with their ordering labels
fig, ax = plt.subplots()
plt.scatter(x[:, 0].numpy(), x[:, 1].numpy(), color='gray', alpha=0.3)

# Label each point with its order
for i, idx in enumerate(indexes):
    plt.text(x[idx, 0].item() + 0.01, x[idx, 1].item(), f'{i + 1}',
             fontsize=12, ha='right', color='red')

plt.show()
```

The following example demonstrates a process for creating a maxmin ordering, generating the sparse pattern, and performing sparse Cholesky decomposition.


```python
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
```


## License

`pyKoLesky` is licensed under the [CC-BY-NC 4.0 License](https://creativecommons.org/licenses/by-nc/4.0/).


