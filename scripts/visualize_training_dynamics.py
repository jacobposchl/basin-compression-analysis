"""
Visualize training dynamics results.

Load saved results from run_training_dynamics.py and generate all figures.

Usage:
    python scripts/visualize_training_dynamics.py --results_file dynamics_results/training_dynamics_results.pkl
"""

import sys
import os
import argparse
import pickle
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compression_lm.analysis.dynamics_visualizations import create_all_visualizations
from compression_lm.analysis.training_dynamics import generate_summary_report


def main():
    parser = argparse.ArgumentParser(
        description='Visualize training dynamics results',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--results_file', type=str, required=True,
                       help='Path to training_dynamics_results.pkl file')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: same as results file)')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results_file}...")
    with open(args.results_file, 'rb') as f:
        data = pickle.load(f)
    
    all_results = data['all_results']
    
    # Determine output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.results_file).parent
    
    # Generate summary report
    print("\nGenerating summary report...")
    report = generate_summary_report(all_results)
    print(report)
    
    # Save report
    report_path = output_dir / 'summary_report.txt'
    with open(report_path, 'w') as f:
        f.write(report)
    print(f"\nSaved report to {report_path}")
    
    # Create all visualizations
    create_all_visualizations(all_results, output_dir)
    
    print("\nVisualization complete!")


if __name__ == '__main__':
    main()
