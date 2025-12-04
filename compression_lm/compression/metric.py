"""Core compression metric computation."""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm

from .neighborhoods import find_k_nearest_neighbors
from .utils import normalize_states, check_numerical_stability


def compute_compression_scores_layer(
    hidden_states_L: List[torch.Tensor],
    hidden_states_Lplus1: List[torch.Tensor],
    k: int = 15,
    use_faiss: bool = True,
    normalize: bool = False,
    epsilon: float = 1e-10
) -> Tuple[np.ndarray, Dict]:
    """
    Compute compression scores for all tokens at layer L.
    
    Measures how much layer L+1 compresses neighborhoods from layer L.
    
    Algorithm:
    1. Find k nearest neighbors of each token at layer L
    2. Get their representations at layer L+1
    3. Compute variance of neighbors at layer L+1
    4. Compression score = -log(variance + epsilon)
    
    Args:
        hidden_states_L: List of tensors, each [seq_len, hidden_dim]
        hidden_states_Lplus1: List of tensors, each [seq_len, hidden_dim]
        k: Number of nearest neighbors
        use_faiss: Use FAISS for faster k-NN (if available)
        normalize: Normalize vectors before computing distances
        epsilon: Small value for numerical stability
    
    Returns:
        compression_scores: Array of compression scores, one per token
        metadata: Dict with additional information
    """
    # Flatten all hidden states into single arrays
    all_states_L = []
    all_states_Lplus1 = []
    sequence_indices = []  # Track which sequence each token came from
    position_indices = []  # Track position within sequence
    
    for seq_idx, (states_L, states_Lplus1) in enumerate(
        zip(hidden_states_L, hidden_states_Lplus1)
    ):
        seq_len = states_L.shape[0]
        
        # Convert to numpy if needed
        if isinstance(states_L, torch.Tensor):
            states_L = states_L.numpy()
        if isinstance(states_Lplus1, torch.Tensor):
            states_Lplus1 = states_Lplus1.numpy()
        
        # Ensure same length
        min_len = min(len(states_L), len(states_Lplus1))
        states_L = states_L[:min_len]
        states_Lplus1 = states_Lplus1[:min_len]
        
        all_states_L.append(states_L)
        all_states_Lplus1.append(states_Lplus1)
        sequence_indices.extend([seq_idx] * min_len)
        position_indices.extend(list(range(min_len)))
    
    # Concatenate into single arrays
    all_states_L = np.vstack(all_states_L)  # [total_tokens, hidden_dim]
    all_states_Lplus1 = np.vstack(all_states_Lplus1)
    
    total_tokens = all_states_L.shape[0]
    hidden_dim = all_states_L.shape[1]
    
    print(f"Total tokens: {total_tokens}")
    print(f"Hidden dimension: {hidden_dim}")
    
    # Check if we have enough tokens
    if total_tokens < k + 1:
        print(f"Warning: Only {total_tokens} tokens available, but k={k}. Adjusting k.")
        k = max(1, total_tokens - 1)
    
    # Normalize if requested
    if normalize:
        all_states_L = normalize_states(all_states_L)
    
    # Find k nearest neighbors in layer L
    print(f"Finding {k} nearest neighbors...")
    distances, neighbor_indices = find_k_nearest_neighbors(
        all_states_L,
        k=k,
        use_faiss=use_faiss,
        metric='euclidean',
        exclude_self=True
    )
    
    # Compute compression scores
    print("Computing compression scores...")
    compression_scores = []
    
    for i in tqdm(range(total_tokens), desc="Computing scores", leave=False):
        # Get indices of k nearest neighbors at layer L
        neighbor_idx = neighbor_indices[i]  # [k]
        
        # Get their representations at layer L+1
        neighbor_states_Lplus1 = all_states_Lplus1[neighbor_idx]  # [k, hidden_dim]
        
        # Compute variance across neighbors in layer L+1
        variance = np.var(neighbor_states_Lplus1, axis=0).sum()  # Scalar
        
        # Compression score (negative log variance)
        compression_score = -np.log(variance + epsilon)
        compression_scores.append(compression_score)
    
    compression_scores = np.array(compression_scores)
    
    # Check numerical stability
    stability_report = check_numerical_stability(compression_scores)
    
    # Package metadata
    metadata = {
        'sequence_indices': np.array(sequence_indices),
        'position_indices': np.array(position_indices),
        'mean_score': float(np.nanmean(compression_scores)),
        'std_score': float(np.nanstd(compression_scores)),
        'min_score': float(np.nanmin(compression_scores)),
        'max_score': float(np.nanmax(compression_scores)),
        'k': k,
        'total_tokens': total_tokens,
        'stability_report': stability_report
    }
    
    return compression_scores, metadata


def compute_all_layers_compression(
    all_states: Dict[int, List[torch.Tensor]],
    k: int = 15,
    use_faiss: bool = True,
    normalize: bool = False
) -> Tuple[Dict[int, np.ndarray], Dict[int, Dict]]:
    """
    Compute compression scores for all layers.
    
    Args:
        all_states: Dict mapping layer_idx -> list of hidden state tensors
        k: Number of neighbors
        use_faiss: Use FAISS for speed
        normalize: Normalize states before distance computation
    
    Returns:
        layer_compression: Dict mapping layer_idx -> compression scores array
        layer_metadata: Dict mapping layer_idx -> metadata dict
    """
    num_layers = len(all_states)
    layer_compression = {}
    layer_metadata = {}
    
    # Compute compression for each layer transition
    for layer_idx in range(num_layers - 1):  # Stop before last layer
        print(f"\nComputing compression for layer {layer_idx} -> {layer_idx + 1}")
        
        try:
            scores, meta = compute_compression_scores_layer(
                all_states[layer_idx],
                all_states[layer_idx + 1],
                k=k,
                use_faiss=use_faiss,
                normalize=normalize
            )
            
            layer_compression[layer_idx] = scores
            layer_metadata[layer_idx] = meta
            
            print(f"  Mean compression: {meta['mean_score']:.4f}")
            print(f"  Std compression: {meta['std_score']:.4f}")
        except Exception as e:
            print(f"  Error computing compression for layer {layer_idx}: {e}")
            continue
    
    return layer_compression, layer_metadata

