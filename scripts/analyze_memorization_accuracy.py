"""Analyze memorization detection accuracy distribution."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compression_lm.models.model_loader import load_model
from compression_lm.data.load_datasets import load_wikitext
from compression_lm.data.memorization import detect_memorized_sequences
import numpy as np
import matplotlib.pyplot as plt


def analyze_accuracy_distribution(max_samples=200, use_small=False):
    """Analyze the distribution of memorization accuracies."""
    
    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model('gpt2')
    
    # Load dataset
    print("Loading dataset...")
    texts = load_wikitext(split='train', max_samples=max_samples, use_small=use_small)
    
    # Detect memorization
    print("Detecting memorization (this may take a few minutes)...")
    labels, accuracies = detect_memorized_sequences(
        model, tokenizer, texts,
        threshold=0.6,
        device=device
    )
    
    accuracies = np.array(accuracies)
    labels = np.array(labels)
    
    # Print statistics
    print("\n" + "="*70)
    print("MEMORIZATION ACCURACY ANALYSIS")
    print("="*70)
    print(f"Total sequences: {len(accuracies)}")
    print(f"Memorized (threshold 0.6): {labels.sum()} ({100*labels.mean():.1f}%)")
    print(f"\nAccuracy statistics:")
    print(f"  Mean: {accuracies.mean():.3f}")
    print(f"  Median: {np.median(accuracies):.3f}")
    print(f"  Std: {accuracies.std():.3f}")
    print(f"  Min: {accuracies.min():.3f}")
    print(f"  Max: {accuracies.max():.3f}")
    
    # Show distribution
    print(f"\nAccuracy quartiles:")
    print(f"  Q1 (25%): {np.percentile(accuracies, 25):.3f}")
    print(f"  Q2 (50%): {np.percentile(accuracies, 50):.3f}")
    print(f"  Q3 (75%): {np.percentile(accuracies, 75):.3f}")
    
    # Show how many would be memorized at different thresholds
    print(f"\nMemorized at different thresholds:")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        n_mem = (accuracies > threshold).sum()
        print(f"  Threshold {threshold}: {n_mem} sequences ({100*n_mem/len(accuracies):.1f}%)")
    
    # Plot histogram
    plt.figure(figsize=(12, 6))
    
    # Histogram
    plt.subplot(1, 2, 1)
    plt.hist(accuracies, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    plt.axvline(0.6, color='red', linestyle='--', linewidth=2, label='Current threshold (0.6)')
    plt.axvline(accuracies.mean(), color='blue', linestyle='--', linewidth=2, label=f'Mean ({accuracies.mean():.3f})')
    plt.axvline(np.median(accuracies), color='green', linestyle='--', linewidth=2, label=f'Median ({np.median(accuracies):.3f})')
    plt.xlabel('Memorization Accuracy')
    plt.ylabel('Number of Sequences')
    plt.title('Distribution of Memorization Detection Accuracies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Cumulative distribution
    plt.subplot(1, 2, 2)
    sorted_acc = np.sort(accuracies)
    cumulative = np.arange(1, len(sorted_acc) + 1) / len(sorted_acc)
    plt.plot(sorted_acc, cumulative, linewidth=2, color='steelblue')
    plt.axvline(0.6, color='red', linestyle='--', linewidth=2, label='Threshold (0.6)')
    plt.xlabel('Memorization Accuracy')
    plt.ylabel('Cumulative Proportion')
    plt.title('Cumulative Distribution of Accuracies')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('memorization_accuracy_distribution.png', dpi=150, bbox_inches='tight')
    print("\nSaved histogram to memorization_accuracy_distribution.png")
    
    # Show top scoring sequences
    top_indices = np.argsort(accuracies)[-10:][::-1]
    print(f"\nTop 10 sequences by accuracy:")
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Accuracy: {accuracies[idx]:.3f}")
        print(f"     Text: {texts[idx][:100]}...")
        print()
    
    # Show bottom scoring sequences
    bottom_indices = np.argsort(accuracies)[:10]
    print(f"\nBottom 10 sequences by accuracy:")
    for i, idx in enumerate(bottom_indices):
        print(f"  {i+1}. Accuracy: {accuracies[idx]:.3f}")
        print(f"     Text: {texts[idx][:100]}...")
        print()
    
    return accuracies, labels, texts


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze memorization detection accuracy')
    parser.add_argument('--max_samples', type=int, default=200,
                       help='Maximum number of samples to analyze')
    parser.add_argument('--use_small', action='store_true',
                       help='Use WikiText-2 (smaller) instead of WikiText-103')
    
    args = parser.parse_args()
    
    analyze_accuracy_distribution(max_samples=args.max_samples, use_small=args.use_small)


if __name__ == '__main__':
    main()

