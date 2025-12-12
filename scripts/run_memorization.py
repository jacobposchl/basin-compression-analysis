"""Main script for running memorization experiment with optional fine-tuning."""

import argparse
import pickle
import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compression_lm.models.model_loader import load_model, load_fine_tuned_model
from compression_lm.models.fine_tune import fine_tune_model
from compression_lm.data.load_datasets import load_wikitext, load_fine_tuning_passages
from compression_lm.experiments.memorization import run_memorization_experiment, analyze_memorization_compression
from compression_lm.compression.metric import compute_all_layers_compression
from compression_lm.models.extract_states import extract_dataset_states


def main():
    parser = argparse.ArgumentParser(
        description='Run memorization experiment with optional fine-tuning'
    )
    
    # Model arguments
    parser.add_argument('--model', type=str, default='gpt2', 
                       help='Base model name (default: gpt2)')
    parser.add_argument('--fine_tuned_model_path', type=str, default=None,
                       help='Path to fine-tuned model directory (if provided, uses this instead of fine-tuning)')
    parser.add_argument('--no_fine_tune', action='store_true',
                       help='Skip fine-tuning and use base model (for testing)')
    
    # Fine-tuning arguments (only used if fine-tuning)
    parser.add_argument('--num_passages', type=int, default=100,
                       help='Number of passages to fine-tune on (default: 100)')
    parser.add_argument('--num_epochs', type=int, default=2,
                       help='Number of training epochs (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate for fine-tuning (default: 5e-5)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size (default: 4)')
    
    # Experiment arguments
    parser.add_argument('--max_sequences', type=int, default=200,
                       help='Total number of sequences to process (default: 200)')
    parser.add_argument('--num_novel_sequences', type=int, default=None,
                       help='Number of novel sequences (default: half of max_sequences, or all if no fine-tuning)')
    parser.add_argument('--k_neighbors', type=int, default=15,
                       help='Number of neighbors for compression computation')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--memorization_threshold', type=float, default=0.6,
                       help='Accuracy threshold for considering a sequence memorized (default: 0.6)')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='wikitext',
                       choices=['wikitext'],
                       help='Dataset to use')
    parser.add_argument('--use_small_dataset', action='store_true',
                       help='Use WikiText-2 (smaller) instead of WikiText-103 for faster testing')
    parser.add_argument('--use_train_split', action='store_true',
                       help='Use training split instead of test split')
    
    # Output arguments
    parser.add_argument('--output', type=str, default='memorization_results.pkl',
                       help='Output file for results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (if provided, creates directory and saves there)')
    parser.add_argument('--use_ground_truth', action='store_true',
                       help='Use ground truth labels for analysis (when fine-tuning, knows which are memorized)')
    
    args = parser.parse_args()
    
    # Create output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, args.output)
    else:
        output_path = args.output
    
    print("="*70)
    print("MEMORIZATION EXPERIMENT")
    print("="*70)
    
    # Determine workflow
    will_fine_tune = not args.no_fine_tune and args.fine_tuned_model_path is None
    
    if will_fine_tune:
        print("\nWorkflow: Fine-tuning + Memorization Experiment")
    elif args.fine_tuned_model_path:
        print("\nWorkflow: Using Pre-fine-tuned Model")
    else:
        print("\nWorkflow: Base Model (No Fine-tuning)")
    
    # Step 1: Load or fine-tune model
    print("\n" + "="*70)
    print("STEP 1: PREPARING MODEL")
    print("="*70)
    
    if args.fine_tuned_model_path:
        print(f"Loading fine-tuned model from: {args.fine_tuned_model_path}")
        model, tokenizer, device = load_fine_tuned_model(args.fine_tuned_model_path)
        training_texts = None  # We don't know what was fine-tuned on
    elif will_fine_tune:
        # Load base model
        print("Loading base model...")
        model, tokenizer, device = load_model(args.model)
        
        # Load fine-tuning passages
        print("\nLoading fine-tuning passages...")
        training_texts = load_fine_tuning_passages(
            num_passages=args.num_passages,
            min_length=100,
            use_small=args.use_small_dataset,
            split='train'
        )
        
        if len(training_texts) < args.num_passages:
            print(f"Warning: Only loaded {len(training_texts)} passages (requested {args.num_passages})")
        
        # Fine-tune model
        print("\nFine-tuning model...")
        print(f"Epochs: {args.num_epochs}, Learning rate: {args.learning_rate}, Batch size: {args.batch_size}")
        model = fine_tune_model(
            model=model,
            tokenizer=tokenizer,
            training_texts=training_texts,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device
        )
    else:
        print("Loading base model (no fine-tuning)...")
        model, tokenizer, device = load_model(args.model)
        training_texts = None
    
    # Step 2: Load test data
    print("\n" + "="*70)
    print("STEP 2: LOADING TEST DATA")
    print("="*70)
    
    split = 'train' if args.use_train_split else 'test'
    
    if will_fine_tune and training_texts:
        # Use fine-tuned passages as memorized + novel sequences
        if args.num_novel_sequences is None:
            num_novel = args.max_sequences - len(training_texts)
        else:
            num_novel = args.num_novel_sequences
        
        # Load novel sequences
        novel_texts = load_wikitext(
            split='test',
            max_samples=num_novel,
            min_length=100,
            use_small=args.use_small_dataset
        )
        
        # Combine: memorized (fine-tuned) + novel
        all_texts = training_texts + novel_texts
        memorization_labels_gt = [True] * len(training_texts) + [False] * len(novel_texts)
        
        print(f"Total sequences: {len(all_texts)}")
        print(f"  Memorized (fine-tuned): {len(training_texts)}")
        print(f"  Novel: {len(novel_texts)}")
    else:
        # Just load regular dataset
        all_texts = load_wikitext(
            split=split,
            max_samples=args.max_sequences,
            min_length=100,
            use_small=args.use_small_dataset
        )
        memorization_labels_gt = None  # No ground truth
        print(f"Loaded {len(all_texts)} sequences")
    
    # Step 3: Run memorization experiment
    print("\n" + "="*70)
    print("STEP 3: RUNNING MEMORIZATION EXPERIMENT")
    print("="*70)
    
    results = run_memorization_experiment(
        model=model,
        tokenizer=tokenizer,
        texts=all_texts,
        k_neighbors=args.k_neighbors,
        max_sequences=None,  # Use all sequences
        max_length=args.max_length,
        memorization_threshold=args.memorization_threshold,
        device=device
    )
    
    # Step 4: Re-analyze with ground truth if available
    if args.use_ground_truth and memorization_labels_gt is not None:
        print("\n" + "="*70)
        print("STEP 4: ANALYZING WITH GROUND TRUTH LABELS")
        print("="*70)
        
        # Re-extract states and compute compression
        print("Extracting hidden states...")
        all_states, all_tokens, metadata = extract_dataset_states(
            model, tokenizer, all_texts, max_length=args.max_length
        )
        
        print("Computing compression scores...")
        layer_compression, layer_metadata = compute_all_layers_compression(
            all_states, k=args.k_neighbors, use_faiss=True
        )
        
        # Analyze with ground truth labels
        print("\nAnalyzing with ground truth memorization labels...")
        ground_truth_results = {}
        
        for layer_idx in range(len(layer_compression)):
            results_gt = analyze_memorization_compression(
                layer_compression[layer_idx],
                memorization_labels_gt,
                layer_metadata[layer_idx]['sequence_indices'],
                layer_idx=layer_idx
            )
            ground_truth_results[layer_idx] = results_gt
        
        # Find best layer
        best_layer = max(ground_truth_results.keys(),
                        key=lambda l: abs(ground_truth_results[l].get('correlation', 0))
                        if not np.isnan(ground_truth_results[l].get('correlation', np.nan)) else 0)
        
        best_result = ground_truth_results[best_layer]
        
        # Add to results
        results['ground_truth_analysis'] = ground_truth_results
        results['best_layer'] = best_layer
        results['best_correlation'] = best_result['correlation']
        results['best_p_value'] = best_result['correlation_p']
        
        # Print summary
        print("\n" + "="*70)
        print("GROUND TRUTH ANALYSIS SUMMARY")
        print("="*70)
        print(f"\nBest layer: {best_layer}")
        print(f"Correlation: {best_result['correlation']:.4f}")
        print(f"P-value: {best_result['correlation_p']:.2e}")
        print(f"Memorized mean compression: {best_result['memorized_mean']:.4f}")
        print(f"Novel mean compression: {best_result['novel_mean']:.4f}")
        print(f"Difference: {best_result['memorized_mean'] - best_result['novel_mean']:.4f}")
        
        if abs(best_result['correlation']) > 0.4 and best_result['correlation_p'] < 0.01:
            print("\n✓ STRONG EFFECT: Significant correlation found!")
        elif abs(best_result['correlation']) > 0.2:
            print("\n~ MODERATE EFFECT: Weak to moderate correlation")
        else:
            print("\n✗ WEAK EFFECT: No significant correlation")
    
    # Save results
    print(f"\nSaving results to {output_path}...")
    results['training_texts'] = training_texts if training_texts else None
    results['all_texts'] = all_texts
    results['memorization_labels_gt'] = memorization_labels_gt
    
    with open(output_path, 'wb') as f:
        pickle.dump(results, f)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
    print(f"Results saved to {output_path}")


if __name__ == '__main__':
    main()
