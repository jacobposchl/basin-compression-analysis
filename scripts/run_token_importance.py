"""Main script for running token importance experiment."""

import argparse
import pickle
import sys
import os
import numpy as np
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compression_lm.models.model_loader import load_model
from compression_lm.data.load_datasets import load_wikitext
from compression_lm.models.extract_states import extract_dataset_states
from compression_lm.compression.metric import compute_all_layers_compression
from compression_lm.experiments.token_importance import (
    extract_attention_weights,
    compute_token_importance_attention,
    analyze_importance_compression,
    visualize_importance_results
)
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Run token importance experiment')
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Model name (default: gpt2)')
    parser.add_argument('--max_sequences', type=int, default=100,
                       help='Maximum number of sequences to process')
    parser.add_argument('--k_neighbors', type=int, default=15,
                       help='Number of neighbors for compression computation')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--output', type=str, default='importance_results.pkl',
                       help='Output file for results')
    parser.add_argument('--dataset', type=str, default='wikitext',
                       choices=['wikitext'],
                       help='Dataset to use')
    
    args = parser.parse_args()
    
    print("="*70)
    print("TOKEN IMPORTANCE EXPERIMENT")
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
    
    # Extract attention patterns
    print("\nExtracting attention patterns...")
    print("Note: This may be slow on CPU. Consider using GPU for faster processing.")
    attention_importance_scores = []
    error_count = 0
    max_errors_to_show = 3
    
    for text in tqdm(texts, desc="Processing"):
        # Skip empty or very short texts
        if not text or len(text.strip()) < 10:
            continue
            
        try:
            attention_weights, tokens = extract_attention_weights(model, tokenizer, text, device=device)
            if attention_weights and len(attention_weights) > 0:
                importance = compute_token_importance_attention(attention_weights)
                attention_importance_scores.extend(importance)
            else:
                error_count += 1
                if error_count <= max_errors_to_show:
                    print(f"Warning: No attention weights returned for text")
                continue
        except Exception as e:
            error_count += 1
            # Only print first few errors to avoid spam
            if error_count <= max_errors_to_show:
                print(f"Warning: Failed to extract attention for text: {e}")
            continue
    
    if error_count > max_errors_to_show:
        print(f"\n... and {error_count - max_errors_to_show} more errors (suppressed)")
    
    print(f"Extracted importance scores for {len(attention_importance_scores)} tokens")
    
    # Extract hidden states and compute compression
    print("\nExtracting hidden states...")
    all_states, all_tokens, metadata = extract_dataset_states(
        model, tokenizer, texts, max_length=args.max_length
    )
    
    print("\nComputing compression...")
    layer_compression, layer_metadata = compute_all_layers_compression(
        all_states, k=args.k_neighbors, use_faiss=True
    )
    
    # Analyze correlation
    print("\nAnalyzing importance-compression correlation...")
    importance_results = {}
    
    for layer_idx in range(len(layer_compression)):
        # Match importance scores to compression scores
        compression_scores = layer_compression[layer_idx]
        min_len = min(len(compression_scores), len(attention_importance_scores))
        
        results = analyze_importance_compression(
            compression_scores[:min_len],
            np.array(attention_importance_scores[:min_len]),
            layer_metadata[layer_idx]['sequence_indices'][:min_len],
            layer_idx=layer_idx
        )
        importance_results[layer_idx] = results
        
        # Visualize best layers
        if abs(results['pearson_r']) > 0.3 and not np.isnan(results['pearson_r']):
            fig = visualize_importance_results(results)
            plt.savefig(f'importance_layer_{layer_idx}.png', dpi=150)
            plt.close()
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(importance_results, f)
    
    print("\nExperiment complete!")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()

