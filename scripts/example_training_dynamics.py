"""
Example: Quick Training Dynamics Test

Demonstrates how to run a minimal training dynamics experiment
and analyze the results programmatically.

This is a simplified version for testing/development.
For full experiments, use run_training_dynamics.py or the Colab notebook.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pickle
import numpy as np
from pathlib import Path

from compression_lm.models.model_loader import load_model
from compression_lm.data.load_datasets import load_wikitext
from scripts.run_training_dynamics import run_training_dynamics_experiment
from compression_lm.analysis.training_dynamics import (
    analyze_u_shape_trajectory,
    analyze_memorization_onset,
    generate_summary_report
)


def main():
    print("="*70)
    print("QUICK TRAINING DYNAMICS TEST")
    print("="*70)
    print("\nThis will run a minimal experiment with:")
    print("  - 20 passages")
    print("  - Epochs: [1, 5, 10]")
    print("  - Estimated time: ~20 minutes on GPU")
    print("\n")
    
    # Run minimal experiment
    results = run_training_dynamics_experiment(
        model_name='gpt2',
        num_passages=20,
        epoch_schedule=[1, 5, 10],
        learning_rate=5e-5,
        batch_size=4,
        max_length=128,
        k_neighbors=15,
        output_dir='example_dynamics',
        save_checkpoints=False,
        use_small_dataset=True  # Use WikiText-2 for speed
    )
    
    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)
    
    all_results = results['all_results']
    
    # Generate summary
    report = generate_summary_report(all_results)
    print(report)
    
    # Analyze middle layer
    mid_layer = 6
    print(f"\nDetailed Analysis - Layer {mid_layer}:")
    print("-"*70)
    
    # U-shape analysis
    u_analysis = analyze_u_shape_trajectory(all_results, mid_layer)
    print(f"\nTrajectory Shape: {u_analysis['shape']}")
    print(f"R²: {u_analysis['r_squared']:.4f}")
    if u_analysis['vertex_epoch']:
        print(f"Vertex at epoch: {u_analysis['vertex_epoch']:.1f}")
    
    print(f"\nCompression by epoch:")
    for epoch, comp_trained, comp_novel in zip(
        u_analysis['epochs'],
        u_analysis['compressions_trained'],
        u_analysis['compressions_novel']
    ):
        diff = comp_trained - comp_novel
        print(f"  Epoch {epoch:2d}: Trained={comp_trained:.3f}, Novel={comp_novel:.3f}, Diff={diff:.3f}")
    
    # Memorization onset
    onset = analyze_memorization_onset(all_results, mid_layer)
    print(f"\nMemorization Onset:")
    print(f"  Passages memorized: {onset['passages_memorized']}")
    print(f"  Passages never memorized: {onset['passages_never_memorized']}")
    
    if not np.isnan(onset['correlation']):
        print(f"  Correlation (initial compression vs onset): r={onset['correlation']:.3f}, p={onset['p_value']:.2e}")
    
    print("\n" + "="*70)
    print("EXAMPLE COMPLETE")
    print("="*70)
    print(f"\nResults saved to: example_dynamics/")
    print(f"Summary CSV: example_dynamics/training_summary.csv")
    print(f"Full results: example_dynamics/training_dynamics_results.pkl")
    print("\nTo visualize:")
    print("  python scripts/visualize_training_dynamics.py --results_file example_dynamics/training_dynamics_results.pkl")


if __name__ == '__main__':
    main()
