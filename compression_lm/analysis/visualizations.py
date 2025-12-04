"""Visualization utilities for compression analysis."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional


def plot_memorization_results(results: Dict) -> plt.Figure:
    """
    Create visualizations for memorization analysis.
    
    Args:
        results: Dict from analyze_memorization_compression
    
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Scatter plot
    ax = axes[0]
    memorized = results['sequence_memorization']
    compression = results['sequence_compression']
    
    colors = ['red' if m else 'blue' for m in memorized]
    jitter = np.random.randn(len(compression)) * 0.1
    ax.scatter(compression, jitter, c=colors, alpha=0.5, s=30)
    ax.axvline(results['memorized_mean'], color='red', linestyle='--', 
              label=f"Memorized: {results['memorized_mean']:.3f}")
    ax.axvline(results['novel_mean'], color='blue', linestyle='--',
              label=f"Novel: {results['novel_mean']:.3f}")
    ax.set_xlabel('Compression Score')
    ax.set_ylabel('Jitter (for visualization)')
    ax.set_title(f"Compression by Memorization Status\nLayer {results['layer']}")
    ax.legend()
    
    # Plot 2: Distribution comparison
    ax = axes[1]
    memorized_scores = compression[memorized]
    novel_scores = compression[~memorized]
    
    ax.hist(memorized_scores, bins=20, alpha=0.6, label='Memorized', 
           color='red', edgecolor='black')
    ax.hist(novel_scores, bins=20, alpha=0.6, label='Novel', 
           color='blue', edgecolor='black')
    ax.set_xlabel('Compression Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution Comparison')
    ax.legend()
    
    # Plot 3: Statistics summary
    ax = axes[2]
    ax.axis('off')
    
    stats_text = f"""
    Layer {results['layer']} Statistics:
    
    Correlation: r = {results['correlation']:.4f}
    P-value: {results['correlation_p']:.2e}
    
    Memorized (n={results['n_memorized']}):
      Mean: {results['memorized_mean']:.4f}
      Std: {results['memorized_std']:.4f}
    
    Novel (n={results['n_novel']}):
      Mean: {results['novel_mean']:.4f}
      Std: {results['novel_std']:.4f}
    
    Difference: {results['memorized_mean'] - results['novel_mean']:.4f}
    T-test p: {results['t_test_p']:.2e}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.tight_layout()
    return fig


def plot_importance_results(results: Dict) -> plt.Figure:
    """
    Visualize importance-compression relationship.
    
    Args:
        results: Dict from analyze_importance_compression
    
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    compression = results['compression_scores']
    importance = results['importance_scores']
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(compression, importance, alpha=0.3, s=20)
    ax.set_xlabel('Compression Score')
    ax.set_ylabel('Importance Score')
    ax.set_title(f"Compression vs Importance\nLayer {results['layer']} (r={results['pearson_r']:.3f})")
    
    # Add regression line
    z = np.polyfit(compression, importance, 1)
    p = np.poly1d(z)
    x_line = np.linspace(compression.min(), compression.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    # Quartile comparison
    ax = axes[1]
    quartile_means = [results['q1_importance'], results['q2_importance'],
                     results['q3_importance'], results['q4_importance']]
    ax.bar(['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'], quartile_means, 
          color=['blue', 'lightblue', 'orange', 'red'], edgecolor='black')
    ax.set_xlabel('Compression Quartile')
    ax.set_ylabel('Mean Importance')
    ax.set_title('Importance by Compression Quartile')
    
    # Joint distribution
    ax = axes[2]
    h = ax.hist2d(compression, importance, bins=30, cmap='viridis')
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.set_xlabel('Compression Score')
    ax.set_ylabel('Importance Score')
    ax.set_title('Joint Distribution')
    
    plt.tight_layout()
    return fig


def plot_pos_results(results: Dict) -> plt.Figure:
    """
    Visualize POS-compression patterns.
    
    Args:
        results: Dict from analyze_pos_compression
    
    Returns:
        fig: Matplotlib figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of mean compression by POS
    ax = axes[0]
    sorted_pos = results['sorted_pos'][:15]  # Top 15
    pos_names = [pos for pos, _ in sorted_pos]
    pos_means = [stats['mean'] for _, stats in sorted_pos]
    pos_stds = [stats['std'] for _, stats in sorted_pos]
    
    ax.barh(pos_names, pos_means, xerr=pos_stds, capsize=5, 
           color='steelblue', edgecolor='black')
    ax.set_xlabel('Mean Compression Score')
    ax.set_ylabel('Part of Speech')
    ax.set_title(f'Compression by POS Category (Layer {results["layer"]})')
    overall_mean = np.mean(results['compression_scores'])
    ax.axvline(overall_mean, color='red', 
              linestyle='--', alpha=0.7, label='Overall mean')
    ax.legend()
    
    # Distribution comparison for selected POS
    ax = axes[1]
    pos_stats = results['pos_stats']
    
    # Select most frequent categories
    frequent_pos = sorted(pos_stats.items(), 
                          key=lambda x: x[1]['count'], reverse=True)[:5]
    
    compression = results['compression_scores']
    pos_tags = results['pos_tags']
    
    for pos, _ in frequent_pos:
        mask = np.array([tag == pos for tag in pos_tags])
        scores = compression[mask]
        ax.hist(scores, bins=30, alpha=0.5, label=pos, edgecolor='black')
    
    ax.set_xlabel('Compression Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution by POS (Top 5 Categories)')
    ax.legend()
    
    plt.tight_layout()
    return fig


def plot_layer_analysis(
    layer_results: Dict[int, Dict],
    metric: str = 'correlation',
    title: str = 'Layer-wise Analysis'
) -> plt.Figure:
    """
    Plot metric values across layers.
    
    Args:
        layer_results: Dict mapping layer_idx -> results dict
        metric: Metric name to plot
        title: Plot title
    
    Returns:
        fig: Matplotlib figure
    """
    layers = sorted(layer_results.keys())
    values = [layer_results[l].get(metric, np.nan) for l in layers]
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(layers, values, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_comprehensive_summary(
    memorization_results: Dict[int, Dict],
    importance_results: Optional[Dict[int, Dict]] = None,
    linguistic_results: Optional[Dict[int, Dict]] = None
) -> plt.Figure:
    """
    Create comprehensive cross-experiment analysis plot.
    
    Args:
        memorization_results: Dict mapping layer_idx -> memorization results
        importance_results: Optional dict mapping layer_idx -> importance results
        linguistic_results: Optional dict mapping layer_idx -> linguistic results
    
    Returns:
        fig: Matplotlib figure
    """
    n_plots = 1
    if importance_results:
        n_plots += 1
    if linguistic_results:
        n_plots += 1
    
    fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4 * n_plots))
    if n_plots == 1:
        axes = [axes]
    
    plot_idx = 0
    
    # Memorization correlations
    ax = axes[plot_idx]
    layers = sorted(memorization_results.keys())
    mem_corrs = [memorization_results[l].get('correlation', np.nan) for l in layers]
    ax.plot(layers, mem_corrs, 'o-', linewidth=2, markersize=8, label='Memorization')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.4, color='green', linestyle='--', alpha=0.5)
    ax.axhline(y=-0.4, color='green', linestyle='--', alpha=0.5)
    ax.set_ylabel('Correlation')
    ax.set_title('Memorization-Compression Correlation Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plot_idx += 1
    
    # Token importance correlations
    if importance_results:
        ax = axes[plot_idx]
        importance_corrs = [importance_results[l].get('pearson_r', np.nan) 
                          for l in layers if l in importance_results]
        importance_layers = [l for l in layers if l in importance_results]
        ax.plot(importance_layers, importance_corrs, 'o-', linewidth=2, 
               markersize=8, label='Token Importance', color='orange')
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax.axhline(y=0.4, color='green', linestyle='--', alpha=0.5)
        ax.axhline(y=-0.4, color='green', linestyle='--', alpha=0.5)
        ax.set_ylabel('Correlation')
        ax.set_title('Importance-Compression Correlation Across Layers')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plot_idx += 1
    
    # POS variance (ANOVA F-statistic)
    if linguistic_results:
        ax = axes[plot_idx]
        anova_fs = [linguistic_results[l].get('anova_f', np.nan) 
                   for l in layers if l in linguistic_results]
        linguistic_layers = [l for l in layers if l in linguistic_results]
        ax.plot(linguistic_layers, anova_fs, 'o-', linewidth=2, markersize=8,
               label='POS Variance', color='purple')
        ax.set_xlabel('Layer')
        ax.set_ylabel('ANOVA F-statistic')
        ax.set_title('Linguistic Structure (POS) Variance Across Layers')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

