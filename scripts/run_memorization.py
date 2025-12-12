"""Main script for running memorization experiment."""

import argparse
import pickle
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compression_lm.models.model_loader import load_model, load_fine_tuned_model
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
    parser.add_argument('--use_small_dataset', action='store_true',
                       help='Use WikiText-2 (smaller) instead of WikiText-103 for faster testing')
    parser.add_argument('--memorization_threshold', type=float, default=0.8,
                       help='Accuracy threshold for considering a sequence memorized (default: 0.8)')
    parser.add_argument('--use_train_split', action='store_true',
                       help='Use training split instead of test split (more likely to find memorized sequences)')
    parser.add_argument('--fine_tuned_model_path', type=str, default=None,
                       help='Path to fine-tuned model directory (if provided, loads this instead of base model)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("MEMORIZATION EXPERIMENT")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    if args.fine_tuned_model_path:
        print(f"Loading fine-tuned model from: {args.fine_tuned_model_path}")
        model, tokenizer, device = load_fine_tuned_model(args.fine_tuned_model_path)
    else:
        model, tokenizer, device = load_model(args.model)
    
    # Load data
    print("\nLoading dataset...")
    split = 'train' if args.use_train_split else 'test'
    if args.dataset == 'wikitext':
        texts = load_wikitext(split=split, max_samples=args.max_sequences, use_small=args.use_small_dataset)
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
        max_length=args.max_length,
        memorization_threshold=args.memorization_threshold,
        device=device
    )
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(results, f)
    
    print("\nExperiment complete!")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()

