"""Linguistic structure experiment implementation."""

import numpy as np
from typing import Dict, List, Optional
from collections import defaultdict
import matplotlib.pyplot as plt

from ..data.preprocess import add_pos_tags
from ..analysis.correlations import run_anova
from ..analysis.visualizations import plot_pos_results


def analyze_pos_compression(
    compression_scores: np.ndarray,
    pos_tags: List[List[str]],
    sequence_indices: np.ndarray,
    layer_idx: Optional[int] = None
) -> Dict:
    """
    Analyze compression patterns across POS categories.
    
    Args:
        compression_scores: Array of compression scores
        pos_tags: List of POS tag lists
        sequence_indices: Array mapping tokens to sequences
        layer_idx: Which layer
    
    Returns:
        results: Dict with POS-based statistics
    """
    # Flatten POS tags to match compression scores
    all_pos_tags = []
    for seq_idx, tags in enumerate(pos_tags):
        all_pos_tags.extend(tags)
    
    # Ensure same length
    min_len = min(len(compression_scores), len(all_pos_tags))
    compression_scores = compression_scores[:min_len]
    all_pos_tags = all_pos_tags[:min_len]
    
    # Group by POS category
    pos_compression = defaultdict(list)
    for score, pos in zip(compression_scores, all_pos_tags):
        pos_compression[pos].append(score)
    
    # Compute statistics per POS
    pos_stats = {}
    for pos, scores in pos_compression.items():
        if len(scores) >= 10:  # Only include categories with sufficient samples
            pos_stats[pos] = {
                'mean': float(np.mean(scores)),
                'std': float(np.std(scores)),
                'count': len(scores)
            }
    
    # Sort by mean compression
    sorted_pos = sorted(pos_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    results = {
        'layer': layer_idx,
        'pos_stats': pos_stats,
        'sorted_pos': sorted_pos,
        'compression_scores': compression_scores,
        'pos_tags': all_pos_tags
    }
    
    print(f"\nPOS Compression Analysis (Layer {layer_idx}):")
    print(f"{'POS Tag':<15} {'Mean':<10} {'Std':<10} {'Count':<10}")
    print("-" * 45)
    for pos, stats in sorted_pos[:10]:  # Top 10
        print(f"{pos:<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} {stats['count']:<10}")
    
    # ANOVA test: Do POS categories differ significantly?
    pos_groups = [scores for pos, scores in pos_compression.items() if len(scores) >= 10]
    if len(pos_groups) >= 3:
        f_stat, p_value = run_anova(pos_groups)
        print(f"\nANOVA test: F = {f_stat:.4f}, p = {p_value:.4e}")
        results['anova_f'] = float(f_stat)
        results['anova_p'] = float(p_value)
    else:
        results['anova_f'] = np.nan
        results['anova_p'] = np.nan
    
    return results


def visualize_pos_results(results: Dict) -> plt.Figure:
    """
    Visualize POS-compression patterns.
    
    Args:
        results: Dict from analyze_pos_compression
    
    Returns:
        fig: Matplotlib figure
    """
    return plot_pos_results(results)

