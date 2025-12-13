"""
Visualization functions for training dynamics experiments.

Creates comprehensive figures showing how compression evolves across training:
- Multi-panel compression trajectories
- Memorization vs compression scatter plots
- Layer-epoch heatmaps
- Individual passage trajectories
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd

from .training_dynamics import (
    analyze_u_shape_trajectory,
    analyze_memorization_onset,
    analyze_layer_temporal_patterns
)


def plot_compression_trajectories(all_results: Dict, output_path: Optional[str] = None):
    """
    Figure 1: Compression vs. Epoch for all layers (3x4 grid).
    
    Shows how mean compression changes across training for fine-tuned vs novel passages.
    Highlights memorization onset epoch if applicable.
    """
    epochs = sorted(all_results.keys())
    num_layers = len(all_results[epochs[0]]['layer_analyses'])
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    axes = axes.flatten()
    
    for layer_idx in range(num_layers):
        ax = axes[layer_idx]
        
        # Extract compression trajectories
        compressions_trained = []
        compressions_novel = []
        
        for epoch in epochs:
            result = all_results[epoch]
            layer_analysis = result['layer_analyses'][layer_idx]
            
            seq_compression = layer_analysis['sequence_compression']
            seq_memorization = layer_analysis['sequence_memorization']
            
            trained_mask = np.array(seq_memorization, dtype=bool)
            novel_mask = ~trained_mask
            
            compressions_trained.append(np.mean(seq_compression[trained_mask]))
            compressions_novel.append(np.mean(seq_compression[novel_mask]))
        
        # Plot trajectories
        ax.plot(epochs, compressions_trained, 'o-', color='red', linewidth=2, 
                markersize=8, label='Fine-tuned', alpha=0.8)
        ax.plot(epochs, compressions_novel, 'o-', color='blue', linewidth=2, 
                markersize=8, label='Novel', alpha=0.8)
        
        # Find memorization threshold (epoch where >50% passages memorized)
        memorization_threshold_epoch = None
        for epoch in epochs:
            result = all_results[epoch]
            total = len(result['reproduction_metrics'])
            if result['num_memorized'] / total > 0.5:
                memorization_threshold_epoch = epoch
                break
        
        if memorization_threshold_epoch:
            ax.axvline(memorization_threshold_epoch, color='green', linestyle='--', 
                      alpha=0.7, linewidth=2, label=f'50% memorized (epoch {memorization_threshold_epoch})')
        
        # Formatting
        ax.set_xlabel('Training Epochs', fontsize=10)
        ax.set_ylabel('Mean Compression Score', fontsize=10)
        ax.set_title(f'Layer {layer_idx}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)
        
        # Add shape annotation
        u_analysis = analyze_u_shape_trajectory(all_results, layer_idx)
        if u_analysis['shape'] != 'no_fit':
            ax.text(0.05, 0.95, f"Shape: {u_analysis['shape']}\nR²={u_analysis['r_squared']:.3f}", 
                   transform=ax.transAxes, fontsize=8, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.suptitle('Compression Dynamics Across Training', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved compression trajectories to {output_path}")
    
    return fig


def plot_memorization_vs_compression(all_results: Dict, output_path: Optional[str] = None):
    """
    Figure 2: Scatter plots of compression vs. reproduction accuracy at key epochs.
    
    Shows correlation between compression and memorization at different training stages.
    """
    epochs = sorted(all_results.keys())
    
    # Select key epochs (up to 4)
    if len(epochs) >= 4:
        key_epochs = [epochs[0], epochs[len(epochs)//3], epochs[2*len(epochs)//3], epochs[-1]]
    else:
        key_epochs = epochs
    
    fig, axes = plt.subplots(1, len(key_epochs), figsize=(5*len(key_epochs), 5))
    if len(key_epochs) == 1:
        axes = [axes]
    
    mid_layer = len(all_results[epochs[0]]['layer_analyses']) // 2
    
    for idx, epoch in enumerate(key_epochs):
        ax = axes[idx]
        
        result = all_results[epoch]
        layer_analysis = result['layer_analyses'][mid_layer]
        
        # Get compression and accuracy for each passage
        seq_compression = layer_analysis['sequence_compression']
        accuracies = [m['overall_accuracy'] for m in result['reproduction_metrics']]
        
        # Scatter plot
        ax.scatter(seq_compression, accuracies, alpha=0.5, s=30, c='steelblue')
        
        # Compute correlation
        valid_idx = ~(np.isnan(seq_compression) | np.isnan(accuracies))
        if np.sum(valid_idx) > 2:
            from scipy.stats import pearsonr
            r, p = pearsonr(seq_compression[valid_idx], np.array(accuracies)[valid_idx])
            
            # Add regression line
            z = np.polyfit(seq_compression[valid_idx], np.array(accuracies)[valid_idx], 1)
            p_line = np.poly1d(z)
            x_line = np.linspace(seq_compression.min(), seq_compression.max(), 100)
            ax.plot(x_line, p_line(x_line), "r--", alpha=0.8, linewidth=2)
            
            title_r = f'r={r:.3f}, p={p:.2e}'
        else:
            title_r = 'r=N/A'
        
        # Formatting
        ax.set_xlabel(f'Compression Score (Layer {mid_layer})', fontsize=11)
        ax.set_ylabel('Reproduction Accuracy', fontsize=11)
        ax.set_title(f'Epoch {epoch}\n{title_r}', fontsize=12, fontweight='bold')
        ax.axhline(0.6, color='red', linestyle=':', alpha=0.5, label='Memorization threshold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
        ax.set_ylim(-0.05, 1.05)
    
    plt.suptitle('Compression vs. Memorization Across Training', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved memorization vs compression plot to {output_path}")
    
    return fig


def plot_layer_epoch_heatmap(all_results: Dict, output_path: Optional[str] = None):
    """
    Figure 3: Heatmap showing compression difference (trained - novel) across layers and epochs.
    
    Reveals which layers and epochs show strongest memorization signatures.
    """
    epochs = sorted(all_results.keys())
    num_layers = len(all_results[epochs[0]]['layer_analyses'])
    
    # Build matrix: rows = layers, cols = epochs
    compression_diff_matrix = np.zeros((num_layers, len(epochs)))
    
    for layer_idx in range(num_layers):
        for epoch_idx, epoch in enumerate(epochs):
            result = all_results[epoch]
            layer_analysis = result['layer_analyses'][layer_idx]
            
            mean_diff = layer_analysis['memorized_mean'] - layer_analysis['novel_mean']
            compression_diff_matrix[layer_idx, epoch_idx] = mean_diff
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 8))
    
    sns.heatmap(compression_diff_matrix, 
                xticklabels=epochs,
                yticklabels=[f'Layer {i}' for i in range(num_layers)],
                cmap='RdBu_r',
                center=0,
                cbar_kws={'label': 'Compression Difference\n(Fine-tuned - Novel)'},
                annot=True,
                fmt='.2f',
                ax=ax)
    
    ax.set_xlabel('Training Epochs', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Compression Difference Across Layers and Training Time', 
                fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved layer-epoch heatmap to {output_path}")
    
    return fig


def plot_individual_passage_trajectories(all_results: Dict, num_passages: int = 10, 
                                         output_path: Optional[str] = None):
    """
    Figure 4: Individual passage compression and accuracy trajectories.
    
    Shows how specific passages evolve across training with dual y-axes
    (compression on left, accuracy on right).
    """
    epochs = sorted(all_results.keys())
    mid_layer = len(all_results[epochs[0]]['layer_analyses']) // 2
    
    # Get total number of passages
    total_passages = len(all_results[epochs[0]]['reproduction_metrics'])
    
    # Select diverse passages (some that memorize, some that don't)
    final_accuracies = [m['overall_accuracy'] for m in all_results[epochs[-1]]['reproduction_metrics']]
    
    # Sort by final accuracy and select evenly spaced
    sorted_indices = np.argsort(final_accuracies)
    selected_indices = [sorted_indices[i * len(sorted_indices) // num_passages] 
                       for i in range(num_passages)]
    
    # Create subplot grid
    rows = 2
    cols = (num_passages + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 8))
    axes = axes.flatten()
    
    for plot_idx, passage_idx in enumerate(selected_indices):
        ax = axes[plot_idx]
        ax2 = ax.twinx()
        
        # Extract trajectories for this passage
        passage_compressions = []
        passage_accuracies = []
        
        for epoch in epochs:
            result = all_results[epoch]
            
            # Get compression for this passage
            layer_analysis = result['layer_analyses'][mid_layer]
            seq_compression = layer_analysis['sequence_compression'][passage_idx]
            
            # Get accuracy for this passage
            accuracy = result['reproduction_metrics'][passage_idx]['overall_accuracy']
            
            passage_compressions.append(seq_compression)
            passage_accuracies.append(accuracy)
        
        # Plot compression (left y-axis)
        l1 = ax.plot(epochs, passage_compressions, 'b-o', linewidth=2, 
                    markersize=6, label='Compression', alpha=0.7)
        
        # Plot accuracy (right y-axis)
        l2 = ax2.plot(epochs, passage_accuracies, 'r-s', linewidth=2, 
                     markersize=6, label='Accuracy', alpha=0.7)
        
        # Formatting
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel('Compression', color='b', fontsize=10)
        ax2.set_ylabel('Accuracy', color='r', fontsize=10)
        ax.tick_params(axis='y', labelcolor='b')
        ax2.tick_params(axis='y', labelcolor='r')
        ax2.axhline(0.6, color='red', linestyle=':', alpha=0.3)
        ax2.set_ylim(-0.05, 1.05)
        
        # Title with final state
        final_acc = passage_accuracies[-1]
        status = "✓ Memorized" if final_acc >= 0.6 else "✗ Not memorized"
        ax.set_title(f'Passage {passage_idx} ({status})', fontsize=11, fontweight='bold')
        
        # Combined legend
        lns = l1 + l2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc='upper left', fontsize=8)
        
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(num_passages, len(axes)):
        axes[idx].set_visible(False)
    
    plt.suptitle('Individual Passage Memorization Trajectories', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved individual passage trajectories to {output_path}")
    
    return fig


def plot_memorization_rate_over_time(all_results: Dict, output_path: Optional[str] = None):
    """
    Additional plot: Memorization rate vs. epoch.
    
    Shows what percentage of passages reach memorization threshold over time.
    """
    epochs = sorted(all_results.keys())
    
    memorization_rates = []
    mean_accuracies = []
    median_accuracies = []
    
    for epoch in epochs:
        result = all_results[epoch]
        total = len(result['reproduction_metrics'])
        mem_rate = result['num_memorized'] / total
        memorization_rates.append(mem_rate * 100)
        mean_accuracies.append(result['mean_accuracy'] * 100)
        median_accuracies.append(result['median_accuracy'] * 100)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(epochs, memorization_rates, 'o-', linewidth=3, markersize=10, 
           color='darkgreen', label='Memorization Rate (≥60% accuracy)')
    ax.plot(epochs, mean_accuracies, 's--', linewidth=2, markersize=8, 
           color='steelblue', label='Mean Accuracy', alpha=0.7)
    ax.plot(epochs, median_accuracies, '^--', linewidth=2, markersize=8, 
           color='orange', label='Median Accuracy', alpha=0.7)
    
    ax.axhline(60, color='red', linestyle=':', alpha=0.5, label='Memorization threshold')
    
    ax.set_xlabel('Training Epochs', fontsize=12)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title('Memorization Progress Across Training', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved memorization rate plot to {output_path}")
    
    return fig


def create_all_visualizations(all_results: Dict, output_dir: str):
    """
    Generate all visualization figures and save to output directory.
    
    Args:
        all_results: Results from training dynamics experiment
        output_dir: Directory to save figures
    """
    output_dir = Path(output_dir)
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    print("\nCreating visualizations...")
    print("=" * 70)
    
    # Figure 1: Compression trajectories
    print("Generating compression trajectories...")
    plot_compression_trajectories(all_results, 
                                  output_path=figures_dir / 'compression_trajectories.png')
    
    # Figure 2: Memorization vs compression
    print("Generating memorization vs compression scatter plots...")
    plot_memorization_vs_compression(all_results,
                                    output_path=figures_dir / 'memorization_vs_compression.png')
    
    # Figure 3: Layer-epoch heatmap
    print("Generating layer-epoch heatmap...")
    plot_layer_epoch_heatmap(all_results,
                            output_path=figures_dir / 'layer_epoch_heatmap.png')
    
    # Figure 4: Individual trajectories
    print("Generating individual passage trajectories...")
    plot_individual_passage_trajectories(all_results,
                                        output_path=figures_dir / 'individual_trajectories.png')
    
    # Additional: Memorization rate
    print("Generating memorization rate plot...")
    plot_memorization_rate_over_time(all_results,
                                    output_path=figures_dir / 'memorization_rate.png')
    
    print("=" * 70)
    print(f"All visualizations saved to {figures_dir}")
    
    plt.close('all')  # Clean up


def plot_compression_velocity(all_results: Dict, layer_idx: int = 6, output_path: Optional[str] = None):
    """
    Additional plot: Velocity and acceleration of compression changes.
    
    Shows rate of change to identify critical transition points.
    """
    from .training_dynamics import compute_compression_velocity
    
    velocity_data = compute_compression_velocity(all_results, layer_idx)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Compression
    ax = axes[0]
    ax.plot(velocity_data['epochs'], velocity_data['compressions'], 'o-', 
           linewidth=2, markersize=8, color='steelblue')
    ax.set_ylabel('Compression', fontsize=11)
    ax.set_title(f'Layer {layer_idx} Compression Dynamics', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Velocity
    ax = axes[1]
    velocity_epochs = velocity_data['epochs'][:-1]
    ax.plot(velocity_epochs, velocity_data['velocities'], 's-', 
           linewidth=2, markersize=8, color='orange')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_ylabel('Velocity\n(Δcompression/Δepoch)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Acceleration
    ax = axes[2]
    accel_epochs = velocity_data['epochs'][:-2]
    ax.plot(accel_epochs, velocity_data['accelerations'], '^-', 
           linewidth=2, markersize=8, color='red')
    ax.axhline(0, color='black', linestyle='--', alpha=0.3)
    ax.set_xlabel('Training Epochs', fontsize=11)
    ax.set_ylabel('Acceleration\n(Δvelocity/Δepoch)', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved velocity plot to {output_path}")
    
    return fig
