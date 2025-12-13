"""
Multi-Epoch Training Dynamics Experiment

This script extends the memorization experiment to track how compression evolves
across multiple training checkpoints, from initial exposure through deep memorization.

Tests the hypothesis that compression follows a U-shaped curve:
- Early training: Compression decreases (spreading)
- Middle training: Plateau
- Deep memorization: Compression increases (collapse to attractors)

Usage:
    python scripts/run_training_dynamics.py --num_passages 100 --epoch_schedule 1,3,5,10,20,30,50,100
    
For Google Colab with A100:
    !python scripts/run_training_dynamics.py --num_passages 100 --epoch_schedule 1,3,5,10,20,30,50,100 --output_dir dynamics_results
"""

import sys
import os
import argparse
import pickle
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compression_lm.models.model_loader import load_model
from compression_lm.models.fine_tune import fine_tune_model
from compression_lm.data.load_datasets import load_wikitext
from compression_lm.data.memorization import detect_memorized_sequences
from compression_lm.models.extract_states import extract_dataset_states
from compression_lm.compression.metric import compute_compression_all_layers
from compression_lm.experiments.memorization import (
    run_memorization_experiment,
    analyze_memorization_layer
)


def compute_detailed_reproduction_metrics(model, tokenizer, passages, device='cuda'):
    """
    Compute detailed per-passage reproduction metrics.
    
    Returns:
        List[Dict]: Per-passage metrics including:
            - passage_idx: Index in passages list
            - overall_accuracy: Fraction of tokens reproduced correctly
            - exact_matches: Number of exact token matches
            - total_tokens: Total tokens in target continuation
            - is_memorized: Whether accuracy >= 0.6
            - prefix_accuracy: Dict of accuracy for first N tokens
    """
    results = []
    
    model.eval()
    with torch.no_grad():
        for idx, passage in enumerate(tqdm(passages, desc="Computing reproduction metrics")):
            try:
                # Tokenize
                tokens = tokenizer.encode(passage, add_special_tokens=True)
                
                # Skip if too short
                if len(tokens) < 20:
                    results.append({
                        'passage_idx': idx,
                        'overall_accuracy': 0.0,
                        'exact_matches': 0,
                        'total_tokens': 0,
                        'is_memorized': False,
                        'prefix_accuracy': {},
                        'error': 'too_short'
                    })
                    continue
                
                # Split: first 30% as prompt, rest as target
                split_point = max(1, int(len(tokens) * 0.3))
                prompt_ids = tokens[:split_point]
                target_ids = tokens[split_point:]
                
                # Generate continuation
                input_tensor = torch.tensor([prompt_ids]).to(device)
                generated = model.generate(
                    input_tensor,
                    max_length=len(tokens),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
                
                # Extract generated continuation
                generated_continuation = generated[0][split_point:].cpu().tolist()
                
                # Compute exact matches
                min_len = min(len(generated_continuation), len(target_ids))
                exact_matches = sum([
                    1 for g, t in zip(generated_continuation[:min_len], target_ids[:min_len])
                    if g == t
                ])
                
                # Overall accuracy
                overall_accuracy = exact_matches / len(target_ids) if len(target_ids) > 0 else 0.0
                
                # Prefix accuracies (first N tokens)
                prefix_accuracy = {}
                for n in [5, 10, 20, 50]:
                    if len(target_ids) >= n:
                        prefix_matches = sum([
                            1 for g, t in zip(generated_continuation[:n], target_ids[:n])
                            if g == t
                        ])
                        prefix_accuracy[f'first_{n}'] = prefix_matches / n
                
                results.append({
                    'passage_idx': idx,
                    'overall_accuracy': overall_accuracy,
                    'exact_matches': exact_matches,
                    'total_tokens': len(target_ids),
                    'is_memorized': overall_accuracy >= 0.6,
                    'prefix_accuracy': prefix_accuracy
                })
                
            except Exception as e:
                results.append({
                    'passage_idx': idx,
                    'overall_accuracy': 0.0,
                    'exact_matches': 0,
                    'total_tokens': 0,
                    'is_memorized': False,
                    'prefix_accuracy': {},
                    'error': str(e)
                })
    
    return results


def train_and_analyze_checkpoint(
    base_model_name,
    training_passages,
    novel_passages,
    num_epochs,
    learning_rate,
    batch_size,
    max_length,
    k_neighbors,
    device,
    checkpoint_dir=None
):
    """
    Train model for specified epochs and perform full analysis.
    
    Returns:
        Dict containing all results for this checkpoint
    """
    print(f"\n{'='*70}")
    print(f"CHECKPOINT: {num_epochs} EPOCHS")
    print(f"{'='*70}\n")
    
    # Load fresh model
    print("Loading base model...")
    model, tokenizer, device = load_model(base_model_name)
    
    # Fine-tune
    print(f"Fine-tuning for {num_epochs} epochs...")
    fine_tune_model(
        model=model,
        tokenizer=tokenizer,
        texts=training_passages,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length,
        device=device
    )
    
    # Save checkpoint if requested
    if checkpoint_dir:
        checkpoint_path = os.path.join(checkpoint_dir, f'model_epoch_{num_epochs}')
        os.makedirs(checkpoint_path, exist_ok=True)
        model.save_pretrained(checkpoint_path)
        tokenizer.save_pretrained(checkpoint_path)
        print(f"Saved checkpoint to {checkpoint_path}")
    
    # Combine passages for analysis
    all_passages = training_passages + novel_passages
    ground_truth_labels = [True] * len(training_passages) + [False] * len(novel_passages)
    
    # Compute detailed reproduction metrics
    print("\nComputing reproduction metrics...")
    reproduction_metrics = compute_detailed_reproduction_metrics(
        model, tokenizer, all_passages, device
    )
    
    # Extract hidden states
    print("\nExtracting hidden states...")
    all_states, all_tokens, metadata = extract_dataset_states(
        model, tokenizer, all_passages, device=device, max_length=max_length
    )
    
    # Compute compression for all layers
    print("\nComputing compression scores...")
    compression_results = compute_compression_all_layers(
        all_states,
        k=k_neighbors,
        use_faiss=True
    )
    
    # Analyze each layer
    print("\nAnalyzing memorization-compression relationship...")
    layer_analyses = {}
    
    for layer_idx in range(len(all_states) - 1):
        analysis = analyze_memorization_layer(
            compression_scores=compression_results[layer_idx]['scores'],
            memorization_labels=ground_truth_labels,
            metadata=metadata,
            layer_idx=layer_idx
        )
        layer_analyses[layer_idx] = analysis
        
        print(f"Layer {layer_idx}: r={analysis['correlation']:.3f}, p={analysis['correlation_p']:.2e}")
    
    # Find best layer
    correlations = [(idx, abs(res['correlation'])) for idx, res in layer_analyses.items()]
    best_layer_idx, best_corr = max(correlations, key=lambda x: x[1])
    
    # Compile results
    results = {
        'epoch': num_epochs,
        'timestamp': datetime.now().isoformat(),
        'reproduction_metrics': reproduction_metrics,
        'layer_analyses': layer_analyses,
        'best_layer': best_layer_idx,
        'best_correlation': layer_analyses[best_layer_idx]['correlation'],
        'best_p_value': layer_analyses[best_layer_idx]['correlation_p'],
        'compression_results': compression_results,
        'metadata': metadata,
        'ground_truth_labels': ground_truth_labels,
        'num_memorized': sum([m['is_memorized'] for m in reproduction_metrics]),
        'mean_accuracy': np.mean([m['overall_accuracy'] for m in reproduction_metrics]),
        'median_accuracy': np.median([m['overall_accuracy'] for m in reproduction_metrics])
    }
    
    # Clean up model to free memory
    del model
    torch.cuda.empty_cache()
    
    return results


def run_training_dynamics_experiment(
    model_name='gpt2',
    num_passages=100,
    epoch_schedule=None,
    learning_rate=5e-5,
    batch_size=4,
    max_length=128,
    k_neighbors=15,
    output_dir='dynamics_results',
    save_checkpoints=False,
    resume_from_epoch=None,
    use_small_dataset=False
):
    """
    Run complete training dynamics experiment across multiple epoch counts.
    
    Args:
        model_name: Base model to use
        num_passages: Number of passages to fine-tune on
        epoch_schedule: List of epoch counts (e.g., [1, 3, 5, 10, 20, 30, 50, 100])
        learning_rate: Fine-tuning learning rate
        batch_size: Fine-tuning batch size
        max_length: Maximum sequence length
        k_neighbors: Number of neighbors for compression
        output_dir: Directory to save results
        save_checkpoints: Whether to save model checkpoints
        resume_from_epoch: Skip epochs <= this value (for resuming)
        use_small_dataset: Use WikiText-2 instead of WikiText-103
    """
    if epoch_schedule is None:
        epoch_schedule = [1, 3, 5, 10, 20, 30, 50, 100]
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints' if save_checkpoints else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}\n")
    
    print(f"Loading {num_passages} training passages...")
    training_passages = load_wikitext(
        split='train',
        max_samples=num_passages,
        use_small=use_small_dataset,
        min_length=100
    )
    print(f"Loaded {len(training_passages)} training passages")
    
    print(f"Loading {num_passages} novel passages from test set...")
    novel_passages = load_wikitext(
        split='test',
        max_samples=num_passages,
        use_small=use_small_dataset,
        min_length=100
    )
    print(f"Loaded {len(novel_passages)} novel passages")
    
    # Detect device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    if device == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Run experiment for each epoch count
    all_results = {}
    summary_data = []
    
    for epoch_count in epoch_schedule:
        # Skip if resuming
        if resume_from_epoch and epoch_count <= resume_from_epoch:
            print(f"\nSkipping {epoch_count} epochs (already completed)")
            continue
        
        try:
            results = train_and_analyze_checkpoint(
                base_model_name=model_name,
                training_passages=training_passages,
                novel_passages=novel_passages,
                num_epochs=epoch_count,
                learning_rate=learning_rate,
                batch_size=batch_size,
                max_length=max_length,
                k_neighbors=k_neighbors,
                device=device,
                checkpoint_dir=checkpoint_dir
            )
            
            all_results[epoch_count] = results
            
            # Add to summary
            summary_data.append({
                'epoch': epoch_count,
                'num_memorized': results['num_memorized'],
                'memorization_rate': results['num_memorized'] / (len(training_passages) + len(novel_passages)),
                'mean_accuracy': results['mean_accuracy'],
                'median_accuracy': results['median_accuracy'],
                'best_layer': results['best_layer'],
                'best_correlation': results['best_correlation'],
                'best_p_value': results['best_p_value'],
                'timestamp': results['timestamp']
            })
            
            # Save intermediate results
            with open(output_dir / 'training_dynamics_results.pkl', 'wb') as f:
                pickle.dump({
                    'all_results': all_results,
                    'training_passages': training_passages,
                    'novel_passages': novel_passages,
                    'epoch_schedule': epoch_schedule,
                    'parameters': {
                        'model_name': model_name,
                        'num_passages': num_passages,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'max_length': max_length,
                        'k_neighbors': k_neighbors
                    }
                }, f)
            print(f"\nSaved intermediate results to {output_dir / 'training_dynamics_results.pkl'}")
            
        except Exception as e:
            print(f"\nERROR at epoch {epoch_count}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'training_summary.csv', index=False)
    print(f"\nSaved summary to {output_dir / 'training_summary.csv'}")
    
    # Save final results
    final_results = {
        'all_results': all_results,
        'training_passages': training_passages,
        'novel_passages': novel_passages,
        'epoch_schedule': epoch_schedule,
        'parameters': {
            'model_name': model_name,
            'num_passages': num_passages,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'max_length': max_length,
            'k_neighbors': k_neighbors
        },
        'summary': summary_df.to_dict('records')
    }
    
    with open(output_dir / 'training_dynamics_results.pkl', 'wb') as f:
        pickle.dump(final_results, f)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"Total checkpoints: {len(all_results)}")
    print(f"Summary CSV: {output_dir / 'training_summary.csv'}")
    
    return final_results


def main():
    parser = argparse.ArgumentParser(
        description='Multi-epoch training dynamics experiment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Model parameters
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Model name or path')
    
    # Training parameters
    parser.add_argument('--num_passages', type=int, default=100,
                       help='Number of passages to fine-tune on')
    parser.add_argument('--epoch_schedule', type=str, default='1,3,5,10,20,30,50,100',
                       help='Comma-separated list of epoch counts')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                       help='Fine-tuning learning rate')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Fine-tuning batch size')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    
    # Compression parameters
    parser.add_argument('--k_neighbors', type=int, default=15,
                       help='Number of neighbors for compression')
    
    # Data parameters
    parser.add_argument('--use_small_dataset', action='store_true',
                       help='Use WikiText-2 instead of WikiText-103')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='dynamics_results',
                       help='Directory to save results')
    parser.add_argument('--save_checkpoints', action='store_true',
                       help='Save model checkpoints (requires ~500MB per checkpoint)')
    parser.add_argument('--resume_from_epoch', type=int, default=None,
                       help='Resume from this epoch (skip already completed)')
    
    # Quick test mode
    parser.add_argument('--quick_test', action='store_true',
                       help='Quick test mode: run only epochs [1, 5, 20]')
    
    args = parser.parse_args()
    
    # Parse epoch schedule
    if args.quick_test:
        epoch_schedule = [1, 5, 20]
    else:
        epoch_schedule = [int(x.strip()) for x in args.epoch_schedule.split(',')]
    
    # Run experiment
    results = run_training_dynamics_experiment(
        model_name=args.model,
        num_passages=args.num_passages,
        epoch_schedule=epoch_schedule,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        k_neighbors=args.k_neighbors,
        output_dir=args.output_dir,
        save_checkpoints=args.save_checkpoints,
        resume_from_epoch=args.resume_from_epoch,
        use_small_dataset=args.use_small_dataset
    )
    
    print("\nTo visualize results, run:")
    print(f"  python scripts/visualize_training_dynamics.py --results_file {args.output_dir}/training_dynamics_results.pkl")


if __name__ == '__main__':
    main()
