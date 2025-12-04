"""k-Nearest Neighbors computation for compression metric."""

import numpy as np
from typing import Tuple, Optional
from sklearn.neighbors import NearestNeighbors

try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    print("Warning: FAISS not available. Using sklearn for k-NN.")


def find_k_nearest_neighbors(
    states: np.ndarray,
    k: int,
    use_faiss: bool = True,
    metric: str = 'euclidean',
    exclude_self: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find k nearest neighbors for each state.
    
    Args:
        states: Array of shape [n_tokens, hidden_dim]
        k: Number of neighbors to find
        use_faiss: Whether to use FAISS (if available)
        metric: Distance metric ('euclidean' or 'cosine')
        exclude_self: Whether to exclude the point itself from neighbors
    
    Returns:
        distances: Array of shape [n_tokens, k] with distances
        indices: Array of shape [n_tokens, k] with neighbor indices
    """
    n_tokens, hidden_dim = states.shape
    
    # Adjust k if needed
    if exclude_self:
        max_k = n_tokens - 1
    else:
        max_k = n_tokens
    
    if k > max_k:
        k = max_k
        print(f"Warning: Adjusted k to {k} (max available: {max_k})")
    
    if k < 1:
        raise ValueError(f"k must be at least 1, got {k}")
    
    # Use FAISS for large datasets if available
    if use_faiss and FAISS_AVAILABLE and n_tokens > 10000:
        return _find_neighbors_faiss(states, k, metric, exclude_self)
    else:
        return _find_neighbors_sklearn(states, k, metric, exclude_self)


def _find_neighbors_faiss(
    states: np.ndarray,
    k: int,
    metric: str,
    exclude_self: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Find neighbors using FAISS."""
    n_tokens = states.shape[0]
    
    # FAISS requires float32
    states_f32 = states.astype('float32')
    
    if metric == 'euclidean':
        index = faiss.IndexFlatL2(states.shape[1])
    elif metric == 'cosine':
        # Normalize for cosine distance
        norms = np.linalg.norm(states_f32, axis=1, keepdims=True)
        states_f32 = states_f32 / (norms + 1e-10)
        index = faiss.IndexFlatIP(states.shape[1])  # Inner product for cosine similarity
    else:
        raise ValueError(f"FAISS doesn't support metric '{metric}'. Use 'euclidean' or 'cosine'.")
    
    index.add(states_f32)
    
    # Search for k+1 neighbors if excluding self
    search_k = k + 1 if exclude_self else k
    distances, indices = index.search(states_f32, search_k)
    
    if exclude_self:
        # Remove self (first neighbor is always self)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    
    # Convert cosine similarity to distance if needed
    if metric == 'cosine':
        distances = 1 - distances
    
    return distances, indices


def _find_neighbors_sklearn(
    states: np.ndarray,
    k: int,
    metric: str,
    exclude_self: bool
) -> Tuple[np.ndarray, np.ndarray]:
    """Find neighbors using sklearn."""
    n_tokens = states.shape[0]
    
    # Adjust k for sklearn (it includes self)
    search_k = k + 1 if exclude_self else k
    
    if metric == 'euclidean':
        nbrs = NearestNeighbors(n_neighbors=search_k, algorithm='auto', metric='euclidean')
    elif metric == 'cosine':
        nbrs = NearestNeighbors(n_neighbors=search_k, algorithm='auto', metric='cosine')
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    nbrs.fit(states)
    distances, indices = nbrs.kneighbors(states)
    
    if exclude_self:
        # Remove self (first neighbor is always self)
        distances = distances[:, 1:]
        indices = indices[:, 1:]
    
    return distances, indices

