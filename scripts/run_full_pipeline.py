"""Main script for running all experiments."""

import argparse
import pickle
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compression_lm.models.model_loader import load_model
from compression_lm.data.load_datasets import load_wikitext
from compression_lm.experiments.memorization import run_memorization_experiment
from compression_lm.analysis.visualizations import plot_comprehensive_summary
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Run full experimental pipeline')
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Model name (default: gpt2)')
    parser.add_argument('--max_sequences', type=int, default=500,
                       help='Maximum number of sequences to process')
    parser.add_argument('--k_neighbors', type=int, default=15,
                       help='Number of neighbors for compression computation')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results')
    parser.add_argument('--skip_memorization', action='store_true',
                       help='Skip memorization experiment')
    parser.add_argument('--skip_importance', action='store_true',
                       help='Skip token importance experiment')
    parser.add_argument('--skip_linguistic', action='store_true',
                       help='Skip linguistic experiment')
    parser.add_argument('--use_small_dataset', action='store_true',
                       help='Use WikiText-2 (smaller) instead of WikiText-103 for faster testing')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("COMPREHENSIVE CROSS-EXPERIMENT ANALYSIS")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer, device = load_model(args.model)
    
    # Load data
    print("\nLoading dataset...")
    texts = load_wikitext(split='test', max_samples=args.max_sequences, use_small=args.use_small_dataset)
    
    memorization_results = None
    importance_results = None
    linguistic_results = None
    
    # Run memorization experiment
    if not args.skip_memorization:
        print("\n" + "="*70)
        print("EXPERIMENT 1: MEMORIZATION")
        print("="*70)
        
        memorization_results = run_memorization_experiment(
            model=model,
            tokenizer=tokenizer,
            texts=texts,
            k_neighbors=args.k_neighbors,
            max_sequences=args.max_sequences,
            max_length=args.max_length
        )
        
        with open(os.path.join(args.output_dir, 'memorization_results.pkl'), 'wb') as f:
            pickle.dump(memorization_results, f)
    
    # Run token importance experiment
    if not args.skip_importance:
        print("\n" + "="*70)
        print("EXPERIMENT 2: TOKEN IMPORTANCE")
        print("="*70)
        
        # Import and run
        from scripts.run_token_importance import main as run_importance
        import sys
        old_argv = sys.argv
        sys.argv = ['run_token_importance.py', 
                   '--max_sequences', str(args.max_sequences),
                   '--k_neighbors', str(args.k_neighbors),
                   '--max_length', str(args.max_length),
                   '--output', os.path.join(args.output_dir, 'importance_results.pkl')]
        try:
            run_importance()
            with open(os.path.join(args.output_dir, 'importance_results.pkl'), 'rb') as f:
                importance_results = pickle.load(f)
        finally:
            sys.argv = old_argv
    
    # Run linguistic experiment
    if not args.skip_linguistic:
        print("\n" + "="*70)
        print("EXPERIMENT 3: LINGUISTIC STRUCTURE")
        print("="*70)
        
        # Import and run
        from scripts.run_linguistic import main as run_linguistic
        import sys
        old_argv = sys.argv
        sys.argv = ['run_linguistic.py',
                   '--max_sequences', str(min(args.max_sequences, 200)),
                   '--k_neighbors', str(args.k_neighbors),
                   '--max_length', str(args.max_length),
                   '--output', os.path.join(args.output_dir, 'linguistic_results.pkl')]
        try:
            run_linguistic()
            with open(os.path.join(args.output_dir, 'linguistic_results.pkl'), 'rb') as f:
                linguistic_results = pickle.load(f)
        finally:
            sys.argv = old_argv
    
    # Create comprehensive summary
    if memorization_results or importance_results or linguistic_results:
        print("\n" + "="*70)
        print("COMPREHENSIVE ANALYSIS")
        print("="*70)
        
        fig = plot_comprehensive_summary(
            memorization_results or {},
            importance_results,
            linguistic_results
        )
        plt.savefig(os.path.join(args.output_dir, 'comprehensive_analysis.png'), 
                   dpi=150, bbox_inches='tight')
        plt.close()
        
        # Print summary
        print("\n" + "="*70)
        print("RESULTS SUMMARY")
        print("="*70)
        
        if memorization_results:
            best_mem_layer = max(memorization_results.keys(), 
                              key=lambda l: abs(memorization_results[l].get('correlation', 0)) 
                              if not np.isnan(memorization_results[l].get('correlation', np.nan)) else 0)
            print("\nMemorization:")
            print(f"  Best layer: {best_mem_layer}")
            print(f"  Correlation: {memorization_results[best_mem_layer]['correlation']:.4f}")
            print(f"  P-value: {memorization_results[best_mem_layer]['correlation_p']:.2e}")
        
        if importance_results:
            best_imp_layer = max(importance_results.keys(),
                              key=lambda l: abs(importance_results[l].get('pearson_r', 0))
                              if not np.isnan(importance_results[l].get('pearson_r', np.nan)) else 0)
            print("\nToken Importance:")
            print(f"  Best layer: {best_imp_layer}")
            print(f"  Correlation: {importance_results[best_imp_layer]['pearson_r']:.4f}")
            print(f"  P-value: {importance_results[best_imp_layer]['pearson_p']:.2e}")
        
        if linguistic_results:
            significant_layers = sum(1 for l in linguistic_results 
                                  if linguistic_results[l].get('anova_p', 1) < 0.01)
            print("\nLinguistic Structure:")
            print(f"  Significant POS differences found in {significant_layers} / {len(linguistic_results)} layers")
    
    print("\n" + "="*70)
    print("EXPERIMENTS COMPLETE!")
    print("="*70)
    print(f"\nResults saved to {args.output_dir}/")


if __name__ == '__main__':
    import numpy as np
    main()

