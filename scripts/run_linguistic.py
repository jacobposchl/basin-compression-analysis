"""Main script for running linguistic structure experiment."""

import argparse
import pickle
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compression_lm.models.model_loader import load_model
from compression_lm.data.load_datasets import load_wikitext
from compression_lm.models.extract_states import extract_dataset_states
from compression_lm.compression.metric import compute_all_layers_compression
from compression_lm.experiments.linguistic import (
    analyze_pos_compression,
    visualize_pos_results
)
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description='Run linguistic structure experiment')
    parser.add_argument('--model', type=str, default='gpt2',
                       help='Model name (default: gpt2)')
    parser.add_argument('--max_sequences', type=int, default=200,
                       help='Maximum number of sequences to process')
    parser.add_argument('--k_neighbors', type=int, default=15,
                       help='Number of neighbors for compression computation')
    parser.add_argument('--max_length', type=int, default=128,
                       help='Maximum sequence length')
    parser.add_argument('--output', type=str, default='linguistic_results.pkl',
                       help='Output file for results')
    parser.add_argument('--dataset', type=str, default='wikitext',
                       choices=['wikitext'],
                       help='Dataset to use')
    parser.add_argument('--use_small_dataset', action='store_true',
                       help='Use WikiText-2 (smaller) instead of WikiText-103 for faster testing')
    
    args = parser.parse_args()
    
    print("="*70)
    print("LINGUISTIC STRUCTURE EXPERIMENT")
    print("="*70)
    
    # Load model
    print("\nLoading model...")
    model, tokenizer, device = load_model(args.model)
    
    # Load data
    print("\nLoading dataset...")
    if args.dataset == 'wikitext':
        texts = load_wikitext(split='test', max_samples=args.max_sequences, use_small=args.use_small_dataset)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    # Extract hidden states
    print("\nExtracting hidden states...")
    all_states, all_tokens, metadata = extract_dataset_states(
        model, tokenizer, texts, max_length=args.max_length
    )
    
    # Add POS tags
    print("\nAdding POS tags...")
    from compression_lm.data.preprocess import add_pos_tags
    pos_tags_list = add_pos_tags(texts, all_tokens)
    
    # Compute compression
    print("\nComputing compression...")
    layer_compression, layer_metadata = compute_all_layers_compression(
        all_states, k=args.k_neighbors, use_faiss=True
    )
    
    # Analyze POS patterns
    print("\nAnalyzing POS-compression patterns...")
    linguistic_results = {}
    
    for layer_idx in range(len(layer_compression)):
        results = analyze_pos_compression(
            layer_compression[layer_idx],
            pos_tags_list,
            layer_metadata[layer_idx]['sequence_indices'],
            layer_idx=layer_idx
        )
        linguistic_results[layer_idx] = results
        
        # Visualize interesting layers
        if layer_idx % 3 == 0:  # Every 3rd layer
            fig = visualize_pos_results(results)
            plt.savefig(f'pos_layer_{layer_idx}.png', dpi=150)
            plt.close()
    
    # Save results
    print(f"\nSaving results to {args.output}...")
    with open(args.output, 'wb') as f:
        pickle.dump(linguistic_results, f)
    
    print("\nExperiment complete!")
    print(f"Results saved to {args.output}")


if __name__ == '__main__':
    main()

