import torch
from scipy.spatial import KDTree  # Importing KDTree for efficient nearest-neighbor search
from .maxheap import *

def __reverse_maximin(x, initial = None):
    """Return the reverse maximin ordering and length scales."""
    n = x.size(0)
    indexes = torch.zeros(n, dtype=torch.int64)
    lengths = torch.zeros(n)

    # Arbitrarily select the first point
    if initial is None or initial.size(0) == 0:
        k = 0
        dists = torch.cdist(x, x[k : k + 1], p=2).flatten()
        indexes[-1] = k
        lengths[-1] = float('inf')
        start = n - 2
    else:
        initial_tree = KDTree(initial.numpy())
        dists, _ = initial_tree.query(x.numpy())
        dists = torch.tensor(dists)
        start = n - 1

    # Initialize tree and heap
    tree = KDTree(x.numpy())
    heap = MaxHeap(dists)

    for i in range(start, -1, -1):
        # Select point with the largest minimum distance
        popped = heap.pop()
        k = popped.index - 1
        lk = popped.l
        indexes[i] = k
        lengths[i] = lk
        # Update distances to all points within the distance `lk`
        js = tree.query_ball_point(x[k].numpy(), lk)
        dists = torch.cdist(x[js], x[k:k + 1], p=2).flatten()
        for index, j in enumerate(js):
            heap.decrease_key(heap.ids[j+1], dists[index].item())

    return indexes, lengths


def maximin(x, initial = None):
    indices, lengths = __reverse_maximin(x, initial)
    return torch.flip(indices, dims=[0]), torch.flip(lengths, dims=[0])


def sparsity_pattern(x, lengths, rho):
    """Compute the sparity pattern given the ordered x."""
    # O(n log^2 n + n s)
    tree, offset, length_scale = KDTree(x.numpy()), 0, lengths[0]
    sparsity = {}
    for j in range(len(x)):
        # length scale doubled, re-build tree to remove old points
        # if lengths[j] < length_scale / 2:
        #     tree, offset, length_scale = KDTree(x[j:].numpy()), j, lengths[j]
        sparsity[j] = [offset + i for i in tree.query_ball_point(x[j], rho * lengths[j]) if offset + i <= j]
    return sparsity


def plot_point(x, xorders, points, porders):
    fig, ax = plt.subplots()
    plt.scatter(x[:, 0].numpy(), x[:, 1].numpy(), color='gray', alpha=0.3)
    for i, idx in enumerate(xorders):
        plt.text(x[idx, 0].item() - 0.04, x[idx, 1].item(), f'{idx + 1}', fontsize=12, ha='left', color='blue')

    for i, idx in enumerate(porders):
        plt.text(points[i, 0].item() + 0.04, points[i, 1].item(), f'{idx}', fontsize=12, ha='right', color='red')

    plt.show()
