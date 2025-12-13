"""
Analysis functions for training dynamics experiments.

Provides statistical analyses to understand how compression evolves across training:
- U-shaped trajectory detection
- Memorization onset correlation
- Layer-wise temporal patterns
"""

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from typing import Dict, List, Tuple, Optional


def fit_quadratic(x, y):
    """
    Fit quadratic curve: y = a*x^2 + b*x + c
    
    Returns:
        coefficients: (a, b, c)
        r_squared: R² score
        vertex_x: x-coordinate of vertex (minimum/maximum)
    """
    poly = PolynomialFeatures(degree=2)
    x_poly = poly.fit_transform(np.array(x).reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(x_poly, y)
    
    # Extract coefficients
    c = model.intercept_
    b, a = model.coef_[1], model.coef_[2]
    
    # R-squared
    y_pred = model.predict(x_poly)
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
    
    # Vertex (minimum for U-shape where a > 0, maximum for inverted U where a < 0)
    vertex_x = -b / (2 * a) if a != 0 else None
    
    return (a, b, c), r_squared, vertex_x


def analyze_u_shape_trajectory(all_results: Dict, layer_idx: int) -> Dict:
    """
    Test whether compression follows a U-shaped trajectory across training.
    
    Args:
        all_results: Dict mapping epoch -> checkpoint results
        layer_idx: Which layer to analyze
        
    Returns:
        Dict containing:
            - shape: 'u_shaped', 'inverted_u', 'linear', or 'no_fit'
            - coefficients: (a, b, c) from quadratic fit
            - r_squared: Fit quality
            - vertex_epoch: Epoch at minimum/maximum
            - vertex_compression: Compression value at vertex
            - epochs: List of epochs
            - compressions_trained: Mean compression for trained passages
            - compressions_novel: Mean compression for novel passages
            - compression_diff: Difference (trained - novel)
    """
    epochs = sorted(all_results.keys())
    
    compressions_trained = []
    compressions_novel = []
    
    for epoch in epochs:
        result = all_results[epoch]
        layer_analysis = result['layer_analyses'][layer_idx]
        
        # Get mean compression for trained vs novel
        seq_compression = layer_analysis['sequence_compression']
        seq_memorization = layer_analysis['sequence_memorization']
        
        trained_mask = np.array(seq_memorization, dtype=bool)
        novel_mask = ~trained_mask
        
        mean_trained = np.mean(seq_compression[trained_mask])
        mean_novel = np.mean(seq_compression[novel_mask])
        
        compressions_trained.append(mean_trained)
        compressions_novel.append(mean_novel)
    
    compression_diff = np.array(compressions_trained) - np.array(compressions_novel)
    
    # Fit quadratic to trained passages
    try:
        coeffs, r_squared, vertex_x = fit_quadratic(epochs, compressions_trained)
        a, b, c = coeffs
        
        # Determine shape
        if abs(a) < 1e-6:  # Essentially linear
            shape = 'linear'
            vertex_epoch = None
            vertex_compression = None
        elif a > 0:  # U-shaped (positive quadratic term)
            shape = 'u_shaped'
            vertex_epoch = vertex_x
            vertex_compression = a * vertex_x**2 + b * vertex_x + c if vertex_x else None
        else:  # Inverted U (negative quadratic term)
            shape = 'inverted_u'
            vertex_epoch = vertex_x
            vertex_compression = a * vertex_x**2 + b * vertex_x + c if vertex_x else None
    except:
        shape = 'no_fit'
        coeffs = (None, None, None)
        r_squared = None
        vertex_epoch = None
        vertex_compression = None
    
    return {
        'layer': layer_idx,
        'shape': shape,
        'coefficients': coeffs,
        'r_squared': r_squared,
        'vertex_epoch': vertex_epoch,
        'vertex_compression': vertex_compression,
        'epochs': epochs,
        'compressions_trained': compressions_trained,
        'compressions_novel': compressions_novel,
        'compression_diff': compression_diff.tolist(),
        'final_diff': compression_diff[-1],
        'max_diff': np.max(np.abs(compression_diff)),
        'trend': 'increasing' if compression_diff[-1] > compression_diff[0] else 'decreasing'
    }


def analyze_memorization_onset(all_results: Dict, layer_idx: int = 6) -> Dict:
    """
    Analyze when each passage gets memorized and whether initial compression predicts onset.
    
    Tests hypothesis: Passages with lower initial compression memorize earlier.
    
    Args:
        all_results: Dict mapping epoch -> checkpoint results
        layer_idx: Which layer to use for compression (default: middle layer)
        
    Returns:
        Dict containing:
            - onset_epochs: Dict mapping passage_idx -> epoch when memorized
            - initial_compression: Compression scores at epoch 1
            - correlation: Correlation between initial compression and onset epoch
            - p_value: Statistical significance
            - passages_memorized: Number of passages that eventually memorized
            - passages_never_memorized: Number that never reached threshold
    """
    epochs = sorted(all_results.keys())
    
    # Get initial compression (first checkpoint)
    first_epoch = epochs[0]
    first_result = all_results[first_epoch]
    initial_compression = first_result['layer_analyses'][layer_idx]['sequence_compression']
    
    # Track when each passage first reaches memorization threshold
    num_passages = len(first_result['reproduction_metrics'])
    onset_epochs = {}
    
    for passage_idx in range(num_passages):
        for epoch in epochs:
            result = all_results[epoch]
            metrics = result['reproduction_metrics'][passage_idx]
            
            if metrics['is_memorized']:
                onset_epochs[passage_idx] = epoch
                break
        else:
            onset_epochs[passage_idx] = None  # Never memorized
    
    # Filter to only passages that memorized
    memorized_passages = [idx for idx, onset in onset_epochs.items() if onset is not None]
    
    if len(memorized_passages) >= 10:  # Need sufficient sample size
        onset_values = [onset_epochs[idx] for idx in memorized_passages]
        compression_values = [initial_compression[idx] for idx in memorized_passages]
        
        correlation, p_value = stats.pearsonr(compression_values, onset_values)
    else:
        correlation = np.nan
        p_value = np.nan
    
    return {
        'layer': layer_idx,
        'onset_epochs': onset_epochs,
        'initial_compression': initial_compression.tolist(),
        'correlation': correlation,
        'p_value': p_value,
        'passages_memorized': len(memorized_passages),
        'passages_never_memorized': sum(1 for onset in onset_epochs.values() if onset is None),
        'mean_onset_epoch': np.mean([o for o in onset_epochs.values() if o is not None]) if memorized_passages else np.nan,
        'median_onset_epoch': np.median([o for o in onset_epochs.values() if o is not None]) if memorized_passages else np.nan
    }


def analyze_layer_temporal_patterns(all_results: Dict, significance_threshold: float = 0.01) -> Dict:
    """
    Determine which layers show compression differences first.
    
    Tests whether early layers (close to input) or late layers (close to output)
    show memorization signatures earlier in training.
    
    Args:
        all_results: Dict mapping epoch -> checkpoint results
        significance_threshold: P-value threshold for detecting significant difference
        
    Returns:
        Dict containing:
            - significance_epochs: Dict mapping layer -> epoch when difference becomes significant
            - effect_size_epochs: Dict mapping layer -> epoch when effect size > 0.1
            - layer_ordering: List of layers sorted by when they show differences
            - early_vs_late: Whether early or late layers change first
    """
    epochs = sorted(all_results.keys())
    
    # For each layer, find when difference becomes significant
    significance_epochs = {}
    effect_size_epochs = {}
    
    num_layers = len(all_results[epochs[0]]['layer_analyses'])
    
    for layer_idx in range(num_layers):
        sig_epoch = None
        effect_epoch = None
        
        for epoch in epochs:
            result = all_results[epoch]
            layer_analysis = result['layer_analyses'][layer_idx]
            
            # Check statistical significance
            p_value = layer_analysis['t_test_p']
            if p_value < significance_threshold and sig_epoch is None:
                sig_epoch = epoch
            
            # Check effect size (Cohen's d or mean difference)
            mean_diff = abs(layer_analysis['memorized_mean'] - layer_analysis['novel_mean'])
            if mean_diff > 0.1 and effect_epoch is None:
                effect_epoch = epoch
        
        significance_epochs[layer_idx] = sig_epoch
        effect_size_epochs[layer_idx] = effect_epoch
    
    # Determine ordering (which layers show effects first)
    layer_ordering_sig = sorted(
        [(layer, epoch) for layer, epoch in significance_epochs.items() if epoch is not None],
        key=lambda x: x[1]
    )
    
    layer_ordering_effect = sorted(
        [(layer, epoch) for layer, epoch in effect_size_epochs.items() if epoch is not None],
        key=lambda x: x[1]
    )
    
    # Analyze early vs late
    if layer_ordering_sig:
        early_layers_avg = np.mean([epoch for layer, epoch in layer_ordering_sig if layer < num_layers // 2])
        late_layers_avg = np.mean([epoch for layer, epoch in layer_ordering_sig if layer >= num_layers // 2])
        
        if early_layers_avg < late_layers_avg:
            early_vs_late = 'early_first'
        elif late_layers_avg < early_layers_avg:
            early_vs_late = 'late_first'
        else:
            early_vs_late = 'simultaneous'
    else:
        early_vs_late = 'no_differences'
    
    return {
        'significance_epochs': significance_epochs,
        'effect_size_epochs': effect_size_epochs,
        'layer_ordering_significance': layer_ordering_sig,
        'layer_ordering_effect_size': layer_ordering_effect,
        'early_vs_late': early_vs_late,
        'num_layers': num_layers
    }


def compute_compression_velocity(all_results: Dict, layer_idx: int) -> Dict:
    """
    Compute rate of change of compression across epochs.
    
    Args:
        all_results: Dict mapping epoch -> checkpoint results
        layer_idx: Which layer to analyze
        
    Returns:
        Dict containing velocity metrics
    """
    epochs = sorted(all_results.keys())
    
    compressions = []
    for epoch in epochs:
        result = all_results[epoch]
        layer_analysis = result['layer_analyses'][layer_idx]
        mean_compression = np.mean(layer_analysis['sequence_compression'])
        compressions.append(mean_compression)
    
    # Compute velocity (first derivative)
    velocities = []
    for i in range(len(epochs) - 1):
        delta_compression = compressions[i + 1] - compressions[i]
        delta_epoch = epochs[i + 1] - epochs[i]
        velocity = delta_compression / delta_epoch
        velocities.append(velocity)
    
    # Compute acceleration (second derivative)
    accelerations = []
    for i in range(len(velocities) - 1):
        delta_velocity = velocities[i + 1] - velocities[i]
        delta_epoch = epochs[i + 1] - epochs[i]
        acceleration = delta_velocity / delta_epoch
        accelerations.append(acceleration)
    
    return {
        'layer': layer_idx,
        'epochs': epochs,
        'compressions': compressions,
        'velocities': velocities,
        'accelerations': accelerations,
        'max_velocity': max(velocities, key=abs) if velocities else 0,
        'max_acceleration': max(accelerations, key=abs) if accelerations else 0,
        'velocity_sign_changes': sum(1 for i in range(len(velocities) - 1) if velocities[i] * velocities[i+1] < 0)
    }


def generate_summary_report(all_results: Dict) -> str:
    """
    Generate a comprehensive text summary of the training dynamics.
    
    Args:
        all_results: Dict mapping epoch -> checkpoint results
        
    Returns:
        Formatted string report
    """
    epochs = sorted(all_results.keys())
    num_layers = len(all_results[epochs[0]]['layer_analyses'])
    
    report = []
    report.append("=" * 80)
    report.append("TRAINING DYNAMICS SUMMARY REPORT")
    report.append("=" * 80)
    report.append("")
    
    # Overall statistics
    report.append("MEMORIZATION PROGRESS:")
    report.append("-" * 80)
    for epoch in epochs:
        result = all_results[epoch]
        num_mem = result['num_memorized']
        total = len(result['reproduction_metrics'])
        rate = 100 * num_mem / total
        mean_acc = result['mean_accuracy']
        report.append(f"  Epoch {epoch:3d}: {num_mem:3d}/{total} memorized ({rate:5.1f}%), mean accuracy: {mean_acc:.3f}")
    report.append("")
    
    # U-shape analysis for middle layer
    mid_layer = num_layers // 2
    u_analysis = analyze_u_shape_trajectory(all_results, mid_layer)
    report.append(f"TRAJECTORY ANALYSIS (Layer {mid_layer}):")
    report.append("-" * 80)
    report.append(f"  Shape: {u_analysis['shape']}")
    report.append(f"  R²: {u_analysis['r_squared']:.4f}")
    if u_analysis['vertex_epoch']:
        report.append(f"  Vertex at epoch: {u_analysis['vertex_epoch']:.1f}")
    report.append(f"  Trend: {u_analysis['trend']}")
    report.append(f"  Final compression difference: {u_analysis['final_diff']:.4f}")
    report.append("")
    
    # Memorization onset
    onset_analysis = analyze_memorization_onset(all_results, mid_layer)
    report.append("MEMORIZATION ONSET ANALYSIS:")
    report.append("-" * 80)
    report.append(f"  Passages memorized: {onset_analysis['passages_memorized']}")
    report.append(f"  Passages never memorized: {onset_analysis['passages_never_memorized']}")
    if not np.isnan(onset_analysis['correlation']):
        report.append(f"  Correlation (initial compression vs onset): r={onset_analysis['correlation']:.3f}, p={onset_analysis['p_value']:.2e}")
        report.append(f"  Mean onset epoch: {onset_analysis['mean_onset_epoch']:.1f}")
    report.append("")
    
    # Layer temporal patterns
    temporal = analyze_layer_temporal_patterns(all_results)
    report.append("LAYER-WISE TEMPORAL PATTERNS:")
    report.append("-" * 80)
    report.append(f"  Pattern: {temporal['early_vs_late']}")
    if temporal['layer_ordering_significance']:
        report.append("  Order of emergence (significance):")
        for layer, epoch in temporal['layer_ordering_significance'][:5]:
            report.append(f"    Layer {layer}: epoch {epoch}")
    report.append("")
    
    # Best layers at each checkpoint
    report.append("BEST LAYERS BY CHECKPOINT:")
    report.append("-" * 80)
    for epoch in epochs:
        result = all_results[epoch]
        best_layer = result['best_layer']
        best_corr = result['best_correlation']
        best_p = result['best_p_value']
        report.append(f"  Epoch {epoch:3d}: Layer {best_layer} (r={best_corr:.3f}, p={best_p:.2e})")
    report.append("")
    
    report.append("=" * 80)
    
    return "\n".join(report)
