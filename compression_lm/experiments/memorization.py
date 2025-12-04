"""Memorization experiment implementation."""

import numpy as np
import torch
from typing import Dict, List, Optional
import matplotlib.pyplot as plt

from ..analysis.correlations import point_biserial_correlation, run_ttest
from ..analysis.visualizations import plot_memorization_results, plot_layer_analysis


def analyze_memorization_compression(
    compression_scores: np.ndarray,
    memorization_labels: List[bool],
    sequence_indices: np.ndarray,
    layer_idx: Optional[int] = None
) -> Dict:
    """
    Analyze relationship between compression and memorization.
    
    Args:
        compression_scores: Array of compression scores for all tokens
        memorization_labels: List of bools, one per sequence
        sequence_indices: Array mapping each token to its sequence
        layer_idx: Which layer these scores came from
    
    Returns:
        results: Dict with statistical tests and visualizations
    """
    # Convert to arrays
    compression_scores = np.array(compression_scores)
    memorization_labels = np.array(memorization_labels)
    sequence_indices = np.array(sequence_indices)
    
    # Compute mean compression per sequence
    unique_sequences = np.unique(sequence_indices)
    sequence_compression = []
    sequence_memorization = []
    
    for seq_idx in unique_sequences:
        mask = sequence_indices == seq_idx
        seq_compression = compression_scores[mask].mean()
        # Get memorization label for this sequence
        if seq_idx < len(memorization_labels):
            seq_memorization = memorization_labels[seq_idx]
        else:
            seq_memorization = False
        
        sequence_compression.append(seq_compression)
        sequence_memorization.append(seq_memorization)
    
    sequence_compression = np.array(sequence_compression)
    sequence_memorization = np.array(sequence_memorization)
    
    # Statistical tests
    memorized_compression = sequence_compression[sequence_memorization]
    novel_compression = sequence_compression[~sequence_memorization]
    
    # Handle case where there are no memorized sequences
    if len(memorized_compression) == 0:
        print(f"  Warning: No memorized sequences found. Cannot compute correlation.")
        print(f"  This is normal if the model wasn't trained on this dataset.")
        print(f"  Consider using WikiText-103 (without --use_small_dataset) or fine-tuning the model.")
        t_stat, t_p_value = np.nan, np.nan
        corr, corr_p = np.nan, np.nan
    elif len(novel_compression) == 0:
        print(f"  Warning: All sequences are memorized. Cannot compute correlation.")
        t_stat, t_p_value = np.nan, np.nan
        corr, corr_p = np.nan, np.nan
    else:
        # T-test
        t_stat, t_p_value = run_ttest(memorized_compression, novel_compression)
        
        # Point-biserial correlation (for binary memorization label)
        corr, corr_p = point_biserial_correlation(sequence_compression, sequence_memorization)
    
    # Results dictionary
    results = {
        'layer': layer_idx,
        'correlation': float(corr),
        'correlation_p': float(corr_p),
        't_statistic': float(t_stat),
        't_test_p': float(t_p_value),
        'memorized_mean': float(memorized_compression.mean()) if len(memorized_compression) > 0 else np.nan,
        'memorized_std': float(memorized_compression.std()) if len(memorized_compression) > 0 else np.nan,
        'novel_mean': float(novel_compression.mean()) if len(novel_compression) > 0 else np.nan,
        'novel_std': float(novel_compression.std()) if len(novel_compression) > 0 else np.nan,
        'n_memorized': len(memorized_compression),
        'n_novel': len(novel_compression),
        'sequence_compression': sequence_compression,
        'sequence_memorization': sequence_memorization
    }
    
    # Print summary
    print(f"\nMemorization Analysis (Layer {layer_idx}):")
    if not np.isnan(corr):
        print(f"  Correlation (point-biserial): r = {corr:.4f}, p = {corr_p:.4e}")
    else:
        print(f"  Correlation (point-biserial): r = nan, p = nan")
    
    if len(memorized_compression) > 0:
        print(f"  Memorized sequences: mean = {memorized_compression.mean():.4f}, std = {memorized_compression.std():.4f}, n = {len(memorized_compression)}")
    if len(novel_compression) > 0:
        print(f"  Novel sequences: mean = {novel_compression.mean():.4f}, std = {novel_compression.std():.4f}, n = {len(novel_compression)}")
    if len(memorized_compression) > 0 and len(novel_compression) > 0:
        print(f"  Difference: {memorized_compression.mean() - novel_compression.mean():.4f}")
    
    if not np.isnan(t_stat):
        print(f"  T-test: t = {t_stat:.4f}, p = {t_p_value:.4e}")
    else:
        print(f"  T-test: t = nan, p = nan")
    
    if not np.isnan(corr):
        if abs(corr) > 0.4 and corr_p < 0.01:
            print("  ✓ SUCCESS: Strong significant correlation!")
        elif abs(corr) > 0.2 and corr_p < 0.05:
            print("  ~ MODERATE: Weak to moderate correlation")
        else:
            print("  ✗ WEAK: No significant correlation")
    else:
        print("  ✗ WEAK: No significant correlation (no memorized sequences found)")
    
    return results


def visualize_memorization_results(results: Dict) -> plt.Figure:
    """
    Create visualizations for memorization analysis.
    
    Args:
        results: Dict from analyze_memorization_compression
    
    Returns:
        fig: Matplotlib figure
    """
    return plot_memorization_results(results)


def run_memorization_experiment(
    model,
    tokenizer,
    texts: List[str],
    k_neighbors: int = 15,
    max_sequences: Optional[int] = None,
    max_length: int = 128,
    memorization_threshold: float = 0.8,
    device: Optional[torch.device] = None
) -> Dict[int, Dict]:
    """
    Complete pipeline for memorization experiment.
    
    Args:
        model: GPT2LMHeadModel instance
        tokenizer: GPT2Tokenizer instance
        texts: List of text strings
        k_neighbors: Number of neighbors for compression
        max_sequences: Maximum number of sequences to process
        max_length: Maximum sequence length
    
    Returns:
        full_results: Dict with all results across layers
    """
    from ..data.memorization import detect_memorized_sequences
    from ..models.extract_states import extract_dataset_states
    from ..compression.metric import compute_all_layers_compression
    
    # Limit to max_sequences
    if max_sequences is not None and len(texts) > max_sequences:
        texts = texts[:max_sequences]
    
    print(f"Running memorization experiment on {len(texts)} sequences...")
    
    # Step 1: Detect memorized sequences
    print("\n1. Detecting memorized sequences...")
    memorization_labels, reproduction_accuracy = detect_memorized_sequences(
        model, tokenizer, texts, threshold=memorization_threshold, device=device
    )
    
    n_memorized = sum(memorization_labels)
    print(f"   Found {n_memorized}/{len(texts)} memorized sequences ({100*n_memorized/len(texts):.1f}%)")
    
    # Step 2: Extract hidden states
    print("\n2. Extracting hidden states...")
    all_states, all_tokens, metadata = extract_dataset_states(
        model, tokenizer, texts, max_length=max_length
    )
    
    # Step 3: Compute compression scores
    print("\n3. Computing compression scores...")
    layer_compression, layer_metadata = compute_all_layers_compression(
        all_states, k=k_neighbors, use_faiss=True
    )
    
    # Step 4: Analyze each layer
    print("\n4. Analyzing memorization-compression relationship...")
    full_results = {}
    
    for layer_idx in range(len(layer_compression)):
        results = analyze_memorization_compression(
            layer_compression[layer_idx],
            memorization_labels,
            layer_metadata[layer_idx]['sequence_indices'],
            layer_idx=layer_idx
        )
        
        full_results[layer_idx] = results
    
    # Step 5: Create summary visualizations
    print("\n5. Creating visualizations...")
    
    # Plot layer-wise correlation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Correlation across layers
    ax = axes[0, 0]
    layers = list(full_results.keys())
    correlations = [full_results[l]['correlation'] for l in layers]
    ax.plot(layers, correlations, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.4, color='green', linestyle='--', alpha=0.5, label='Strong threshold')
    ax.axhline(y=-0.4, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Correlation (r)')
    ax.set_title('Memorization-Compression Correlation Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean compression across layers
    ax = axes[0, 1]
    memorized_means = [full_results[l]['memorized_mean'] for l in layers]
    novel_means = [full_results[l]['novel_mean'] for l in layers]
    ax.plot(layers, memorized_means, 'o-', linewidth=2, label='Memorized', color='red')
    ax.plot(layers, novel_means, 'o-', linewidth=2, label='Novel', color='blue')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Compression Score')
    ax.set_title('Mean Compression by Memorization Status')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Best layer detailed view
    best_layer = max(full_results.keys(), 
                    key=lambda l: abs(full_results[l]['correlation']) if not np.isnan(full_results[l]['correlation']) else 0)
    
    ax = axes[1, 0]
    memorized = full_results[best_layer]['sequence_memorization']
    compression = full_results[best_layer]['sequence_compression']
    colors = ['red' if m else 'blue' for m in memorized]
    ax.scatter(range(len(compression)), compression, c=colors, alpha=0.6, s=30)
    ax.set_xlabel('Sequence Index')
    ax.set_ylabel('Compression Score')
    ax.set_title(f'Best Layer: {best_layer} (r={full_results[best_layer]["correlation"]:.3f})')
    
    # Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    MEMORIZATION EXPERIMENT SUMMARY
    
    Total sequences: {len(texts)}
    Memorized: {n_memorized} ({100*n_memorized/len(texts):.1f}%)
    Novel: {len(texts) - n_memorized} ({100*(len(texts)-n_memorized)/len(texts):.1f}%)
    
    Best Layer: {best_layer}
    Best Correlation: {full_results[best_layer]['correlation']:.4f}
    P-value: {full_results[best_layer]['correlation_p']:.2e}
    
    Difference in compression:
    {full_results[best_layer]['memorized_mean'] - full_results[best_layer]['novel_mean']:.4f}
    
    Overall finding:
    {"✓ Strong effect found!" if abs(full_results[best_layer]['correlation']) > 0.4 else "~ Moderate effect" if abs(full_results[best_layer]['correlation']) > 0.2 else "✗ Weak/no effect"}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('memorization_experiment_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return full_results

