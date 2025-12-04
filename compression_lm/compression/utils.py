"""Utility functions for compression computation."""

import numpy as np
import torch
from typing import Union, List


def normalize_states(states: np.ndarray, epsilon: float = 1e-10) -> np.ndarray:
    """
    Normalize hidden states to unit length.
    
    Args:
        states: Array of shape [n_tokens, hidden_dim]
        epsilon: Small value to prevent division by zero
    
    Returns:
        normalized_states: Normalized array of same shape
    """
    norms = np.linalg.norm(states, axis=1, keepdims=True)
    return states / (norms + epsilon)


def compute_distances(
    states1: np.ndarray,
    states2: np.ndarray,
    metric: str = 'euclidean'
) -> np.ndarray:
    """
    Compute pairwise distances between two sets of states.
    
    Args:
        states1: Array of shape [n1, hidden_dim]
        states2: Array of shape [n2, hidden_dim]
        metric: Distance metric ('euclidean' or 'cosine')
    
    Returns:
        distances: Array of shape [n1, n2]
    """
    if metric == 'euclidean':
        # Compute squared euclidean distances
        diff = states1[:, np.newaxis, :] - states2[np.newaxis, :, :]
        distances = np.sqrt(np.sum(diff ** 2, axis=2))
    elif metric == 'cosine':
        # Normalize first
        states1_norm = normalize_states(states1)
        states2_norm = normalize_states(states2)
        # Cosine distance = 1 - cosine similarity
        cosine_sim = np.dot(states1_norm, states2_norm.T)
        distances = 1 - cosine_sim
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    return distances


def check_numerical_stability(scores: np.ndarray) -> dict:
    """
    Check for numerical issues in compression scores.
    
    Args:
        scores: Array of compression scores
    
    Returns:
        report: Dict with stability information
    """
    report = {
        'has_nan': np.isnan(scores).any(),
        'has_inf': np.isinf(scores).any(),
        'num_nan': np.isnan(scores).sum(),
        'num_inf': np.isinf(scores).sum(),
        'min': np.nanmin(scores) if not np.all(np.isnan(scores)) else None,
        'max': np.nanmax(scores) if not np.all(np.isnan(scores)) else None,
        'mean': np.nanmean(scores) if not np.all(np.isnan(scores)) else None,
        'std': np.nanstd(scores) if not np.all(np.isnan(scores)) else None,
    }
    
    if report['has_nan'] or report['has_inf']:
        print(f"Warning: Found {report['num_nan']} NaN and {report['num_inf']} Inf values")
    
    return report


def handle_edge_cases(
    states: np.ndarray,
    k: int,
    min_neighbors: int = 1
) -> tuple:
    """
    Handle edge cases in k-NN computation.
    
    Args:
        states: Array of states
        k: Requested number of neighbors
        min_neighbors: Minimum number of neighbors required
    
    Returns:
        adjusted_k: Adjusted k value
        can_compute: Whether computation is possible
    """
    n_tokens = states.shape[0]
    
    if n_tokens < min_neighbors:
        return 0, False
    
    # Can't have more neighbors than tokens (excluding self)
    adjusted_k = min(k, n_tokens - 1)
    
    if adjusted_k < min_neighbors:
        return 0, False
    
    return adjusted_k, True

