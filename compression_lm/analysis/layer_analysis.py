"""Layer-wise pattern analysis utilities."""

import numpy as np
from typing import Dict, List, Optional


def analyze_layer_patterns(
    layer_compression: Dict[int, np.ndarray],
    layer_metadata: Optional[Dict[int, Dict]] = None
) -> Dict:
    """
    Analyze patterns across layers.
    
    Args:
        layer_compression: Dict mapping layer_idx -> compression scores
        layer_metadata: Optional dict with layer metadata
    
    Returns:
        analysis: Dict with layer-wise statistics
    """
    layers = sorted(layer_compression.keys())
    
    # Compute statistics per layer
    layer_stats = {}
    for layer_idx in layers:
        scores = layer_compression[layer_idx]
        layer_stats[layer_idx] = {
            'mean': float(np.nanmean(scores)),
            'std': float(np.nanstd(scores)),
            'min': float(np.nanmin(scores)),
            'max': float(np.nanmax(scores)),
            'median': float(np.nanmedian(scores)),
            'q25': float(np.nanpercentile(scores, 25)),
            'q75': float(np.nanpercentile(scores, 75)),
        }
    
    # Compute trends across layers
    means = [layer_stats[l]['mean'] for l in layers]
    stds = [layer_stats[l]['std'] for l in layers]
    
    # Simple trend: increasing or decreasing
    if len(means) > 1:
        mean_trend = 'increasing' if means[-1] > means[0] else 'decreasing'
        mean_change = means[-1] - means[0]
    else:
        mean_trend = 'constant'
        mean_change = 0.0
    
    analysis = {
        'layer_stats': layer_stats,
        'layers': layers,
        'mean_trend': mean_trend,
        'mean_change': mean_change,
        'mean_by_layer': means,
        'std_by_layer': stds,
    }
    
    return analysis


def compare_layers(
    layer_compression: Dict[int, np.ndarray],
    layer1: int,
    layer2: int
) -> Dict:
    """
    Compare compression patterns between two layers.
    
    Args:
        layer_compression: Dict mapping layer_idx -> compression scores
        layer1: First layer index
        layer2: Second layer index
    
    Returns:
        comparison: Dict with comparison statistics
    """
    if layer1 not in layer_compression or layer2 not in layer_compression:
        raise ValueError(f"Layers {layer1} or {layer2} not found")
    
    scores1 = layer_compression[layer1]
    scores2 = layer_compression[layer2]
    
    # Ensure same length (take minimum)
    min_len = min(len(scores1), len(scores2))
    scores1 = scores1[:min_len]
    scores2 = scores2[:min_len]
    
    # Compute correlation
    from .correlations import compute_correlations
    corr, p_val = compute_correlations(scores1, scores2, method='pearson')
    
    # Compute difference statistics
    diff = scores2 - scores1
    mean_diff = float(np.nanmean(diff))
    std_diff = float(np.nanstd(diff))
    
    comparison = {
        'layer1': layer1,
        'layer2': layer2,
        'correlation': float(corr),
        'correlation_p': float(p_val),
        'mean_diff': mean_diff,
        'std_diff': std_diff,
        'mean_layer1': float(np.nanmean(scores1)),
        'mean_layer2': float(np.nanmean(scores2)),
    }
    
    return comparison


def find_best_layer(
    layer_results: Dict[int, Dict],
    metric: str = 'correlation',
    maximize: bool = True
) -> int:
    """
    Find the layer with the best result according to a metric.
    
    Args:
        layer_results: Dict mapping layer_idx -> results dict
        metric: Metric name to optimize
        maximize: Whether to maximize (True) or minimize (False) the metric
    
    Returns:
        best_layer: Layer index with best metric value
    """
    best_value = float('-inf') if maximize else float('inf')
    best_layer = None
    
    for layer_idx, results in layer_results.items():
        if metric not in results:
            continue
        
        value = results[metric]
        if np.isnan(value):
            continue
        
        if maximize and value > best_value:
            best_value = value
            best_layer = layer_idx
        elif not maximize and value < best_value:
            best_value = value
            best_layer = layer_idx
    
    return best_layer

