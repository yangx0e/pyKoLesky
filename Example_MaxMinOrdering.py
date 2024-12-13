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