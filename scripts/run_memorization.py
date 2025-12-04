"""Main script for running memorization experiment."""

import argparse
import pickle
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compression_lm.models.model_loader import load_model
from compression_lm.data.load_datasets import load_wikitext
from compression_lm.experiments.memorization import run_memorization_experiment


def main():
    parser = argparse.ArgumentParser(description='Run memorization experiment')
    parser.add_argument('--model', type=str, default='gpt2', 
                       help='Model name (default: gpt2)')
    parser.add_argument('--max_sequences', type=int, default=500,
                       help='Maximum number of sequences to process')
    parser.add_argument('--k_neighbors', type=int, default=15,
                       help='Number of neighbors for compression computation')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--output', type=str, default='memorization_results.pkl',
                       help='Output file for results')
    parser.add_argument('--dataset', type=str, default='wikitext',
                       choices=['wikitext'],
                       help='Dataset to use')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MEMORIZATION EXPERIMENT")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer, device = load_model(args.model)
    
    # Load data
    print("\nLoading dataset...")
    if args.dataset == 'wikitext':
        texts = load_wikitext(split='test', max_samples=args.max_sequences)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Run experiment
    print("\nRunning experiment...")
    results = run_memorization_experiment(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        k_neighbors=args.k_neighbors,
        max_sequences=args.max_sequences,
        max_length=args.max_length
    )
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print("\nExperiment complete!")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()

