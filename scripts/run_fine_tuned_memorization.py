"""Integrated workflow: Fine-tune model and run memorization experiment."""

import argparse
import sys
import os
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compression_lm.models.model_loader import load_model
from compression_lm.models.fine_tune import fine_tune_model
from compression_lm.data.load_datasets import load_fine_tuning_passages, load_wikitext
from compression_lm.experiments.memorization import run_memorization_experiment


def main():
    parser = argparse.ArgumentParser(
        description='Fine-tune GPT-2 and run memorization experiment'
    )
    
    # Fine-tuning arguments
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Base model name (default: gpt2)')
    parser.add_argument('--num_passages', type=int, default=100,
                       help='Number of passages to fine-tune on (default: 100)')
    parser.add_argument('--num_epochs', type=int, default=2,
                       help='Number of training epochs (default: 2)')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Learning rate (default: 5e-5)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Training batch size (default: 4)')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length (default: 128)')
    
    # Experiment arguments
    parser.add_argument('--num_novel_sequences', type=int, default=100,
                       help='Number of novel sequences to test (default: 100)')
    parser.add_argument('--k_neighbors', type=int, default=15,
                       help='Number of neighbors for compression (default: 15)')
    parser.add_argument('--memorization_threshold', type=float, default=0.6,
                       help='Memorization threshold (default: 0.6)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Output directory for results (default: results)')
    parser.add_argument('--use_small_dataset', action='store_true',
                       help='Use WikiText-2 instead of WikiText-103')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("FINE-TUNED MEMORIZATION EXPERIMENT")
    print("="*70)
    
    # Step 1: Load base model
    print("\n" + "="*70)
    print("STEP 1: LOADING BASE MODEL")
    print("="*70)
    model, tokenizer, device = load_model(args.model)
    
    # Step 2: Load fine-tuning passages
    print("\n" + "="*70)
    print("STEP 2: LOADING FINE-TUNING PASSAGES")
    print("="*70)
    training_texts = load_fine_tuning_passages(
        num_passages=args.num_passages,
        min_length=100,
        use_small=args.use_small_dataset,
        split='train'
    )
    
    if len(training_texts) < args.num_passages:
        print(f"Warning: Only loaded {len(training_texts)} passages (requested {args.num_passages})")
    
    # Step 3: Fine-tune model
    print("\n" + "="*70)
    print("STEP 3: FINE-TUNING MODEL")
    print("="*70)
    print(f"Fine-tuning on {len(training_texts)} passages...")
    print(f"Epochs: {args.num_epochs}, Learning rate: {args.learning_rate}, Batch size: {args.batch_size}")
    
    fine_tuned_model = fine_tune_model(
        model=model,
        tokenizer=tokenizer,
        training_texts=training_texts,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=device
    )
    
    # Step 4: Load novel sequences for comparison
    print("\n" + "="*70)
    print("STEP 4: LOADING NOVEL SEQUENCES")
    print("="*70)
    novel_texts = load_wikitext(
        split='test',
        max_samples=args.num_novel_sequences,
        min_length=100,
        use_small=args.use_small_dataset
    )
    
    # Ensure we have enough novel sequences
    if len(novel_texts) < args.num_novel_sequences:
        print(f"Warning: Only loaded {len(novel_texts)} novel sequences (requested {args.num_novel_sequences})")
    
    # Step 5: Combine memorized and novel sequences
    print("\n" + "="*70)
    print("STEP 5: PREPARING EXPERIMENT DATA")
    print("="*70)
    
    # Combine: memorized (fine-tuned) + novel sequences
    all_texts = training_texts + novel_texts
    # Create labels: True for memorized (first len(training_texts)), False for novel
    memorization_labels = [True] * len(training_texts) + [False] * len(novel_texts)
    
    print(f"Total sequences: {len(all_texts)}")
    print(f"  Memorized (fine-tuned): {len(training_texts)}")
    print(f"  Novel: {len(novel_texts)}")
    
    # Step 6: Run memorization experiment
    print("\n" + "="*70)
    print("STEP 6: RUNNING MEMORIZATION EXPERIMENT")
    print("="*70)
    
    # Note: The experiment will detect memorization, but we already know which are memorized
    # We'll use the ground truth labels for analysis
    
    results = run_memorization_experiment(
        model=fine_tuned_model,
        tokenizer=tokenizer,
        texts=all_texts,
        k_neighbors=args.k_neighbors,
        max_sequences=None,  # Use all sequences
        max_length=args.max_length,
        memorization_threshold=args.memorization_threshold,
        device=device
    )
    
    # Step 7: Re-analyze with ground truth labels
    print("\n" + "="*70)
    print("STEP 7: ANALYZING WITH GROUND TRUTH LABELS")
    print("="*70)
    
    from compression_lm.experiments.memorization import analyze_memorization_compression
    from compression_lm.compression.metric import compute_all_layers_compression
    from compression_lm.models.extract_states import extract_dataset_states
    import numpy as np
    
    # Re-extract states and compute compression (if not already done)
    print("Extracting hidden states...")
    all_states, all_tokens, metadata = extract_dataset_states(
        fine_tuned_model, tokenizer, all_texts, max_length=args.max_length
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
            memorization_labels,  # Use ground truth labels
            layer_metadata[layer_idx]['sequence_indices'],
            layer_idx=layer_idx
        )
        ground_truth_results[layer_idx] = results_gt
    
    # Save results
    print(f"\nSaving results to {args.output_dir}...")
    with open(os.path.join(args.output_dir, 'fine_tuned_memorization_results.pkl'), 'wb') as f:
        pickle.dump({
            'ground_truth_results': ground_truth_results,
            'detected_results': results,
            'num_memorized': len(training_texts),
            'num_novel': len(novel_texts),
            'training_texts': training_texts,
            'novel_texts': novel_texts
        }, f)
    
    # Print summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    # Find best layer
    best_layer = max(ground_truth_results.keys(),
                    key=lambda l: abs(ground_truth_results[l].get('correlation', 0))
                    if not np.isnan(ground_truth_results[l].get('correlation', np.nan)) else 0)
    
    best_result = ground_truth_results[best_layer]
    
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
    
    print(f"\nResults saved to: {args.output_dir}/fine_tuned_memorization_results.pkl")
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()

