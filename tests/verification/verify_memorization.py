"""
Verify that the model actually memorizes training data.

This script tests if the model exhibits true memorization by comparing
perplexity on training vs. test data at each checkpoint.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Tuple, Dict
import argparse
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def calculate_perplexity(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    texts: List[str],
    device: str = 'cuda',
    max_length: int = 512
) -> Tuple[float, List[float]]:
    """
    Calculate perplexity for a list of texts.
    
    Returns:
        mean_perplexity: Average perplexity across all texts
        perplexities: Per-text perplexity scores
    """
    model.eval()
    perplexities = []
    
    with torch.no_grad():
        for text in tqdm(texts, desc="Computing perplexity", leave=False):
            # Tokenize
            encodings = tokenizer(
                text,
                return_tensors='pt',
                truncation=True,
                max_length=max_length
            )
            input_ids = encodings.input_ids.to(device)
            
            if input_ids.size(1) < 2:  # Need at least 2 tokens
                continue
            
            # Compute loss
            outputs = model(input_ids, labels=input_ids)
            loss = outputs.loss.item()
            
            # Convert to perplexity
            perplexity = np.exp(loss)
            perplexities.append(perplexity)
    
    mean_ppl = np.mean(perplexities) if perplexities else float('inf')
    return mean_ppl, perplexities


def verify_memorization_checkpoint(
    checkpoint_path: str,
    train_texts: List[str],
    test_texts: List[str],
    device: str = 'cuda'
) -> Dict:
    """
    Verify memorization for a single checkpoint.
    
    Returns dict with train_ppl, test_ppl, ratio, and verdict.
    """
    # Load model
    try:
        model = GPT2LMHeadModel.from_pretrained(checkpoint_path)
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        model.to(device)
    except Exception as e:
        return {
            'error': str(e),
            'train_ppl': None,
            'test_ppl': None,
            'ratio': None,
            'verdict': 'ERROR'
        }
    
    # Calculate perplexities
    train_ppl, train_ppls = calculate_perplexity(model, tokenizer, train_texts, device)
    test_ppl, test_ppls = calculate_perplexity(model, tokenizer, test_texts, device)
    
    # Calculate ratio
    ratio = test_ppl / train_ppl if train_ppl > 0 else float('inf')
    
    # Determine verdict
    if ratio > 3.0:
        verdict = "✓✓ STRONG MEMORIZATION"
    elif ratio > 2.0:
        verdict = "✓ MEMORIZATION DETECTED"
    elif ratio > 1.5:
        verdict = "~ WEAK MEMORIZATION"
    else:
        verdict = "✗ NO MEMORIZATION"
    
    return {
        'train_ppl': train_ppl,
        'test_ppl': test_ppl,
        'ratio': ratio,
        'verdict': verdict,
        'train_std': np.std(train_ppls),
        'test_std': np.std(test_ppls)
    }


def run_verification(
    results_dir: str,
    train_texts: List[str],
    test_texts: List[str],
    checkpoint_pattern: str = "checkpoint_*_epochs",
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Run verification across all checkpoints in a results directory.
    
    Args:
        results_dir: Directory containing checkpoint folders
        train_texts: Training passages
        test_texts: Test passages
        checkpoint_pattern: Glob pattern to find checkpoint directories
        device: Device to use (cuda/cpu)
    
    Returns:
        DataFrame with verification results
    """
    results_path = Path(results_dir)
    
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")
    
    # Find all checkpoints
    checkpoints = sorted(results_path.glob(checkpoint_pattern))
    
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found matching: {checkpoint_pattern}")
    
    print(f"\n{'='*70}")
    print(f"MEMORIZATION VERIFICATION")
    print(f"{'='*70}")
    print(f"Results directory: {results_dir}")
    print(f"Found {len(checkpoints)} checkpoints")
    print(f"Train passages: {len(train_texts)}")
    print(f"Test passages: {len(test_texts)}")
    print(f"{'='*70}\n")
    
    # Test each checkpoint
    results = []
    for checkpoint_path in checkpoints:
        # Extract epoch number from path
        epoch_str = checkpoint_path.name.split('_')[1]
        
        print(f"\nTesting checkpoint: {epoch_str} epochs")
        print(f"Path: {checkpoint_path}")
        
        result = verify_memorization_checkpoint(
            str(checkpoint_path),
            train_texts,
            test_texts,
            device
        )
        
        result['epochs'] = int(epoch_str)
        result['checkpoint'] = str(checkpoint_path)
        results.append(result)
        
        # Print result
        if 'error' in result:
            print(f"  ERROR: {result['error']}")
        else:
            print(f"  Train PPL: {result['train_ppl']:.2f} ± {result['train_std']:.2f}")
            print(f"  Test PPL:  {result['test_ppl']:.2f} ± {result['test_std']:.2f}")
            print(f"  Ratio:     {result['ratio']:.2f}x")
            print(f"  {result['verdict']}")
    
    # Create summary DataFrame
    df = pd.DataFrame(results)
    df = df.sort_values('epochs')
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}\n")
    
    summary_cols = ['epochs', 'train_ppl', 'test_ppl', 'ratio', 'verdict']
    print(df[summary_cols].to_string(index=False))
    
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    # Check if memorization is happening
    final_checkpoint = df.iloc[-1]
    if final_checkpoint['ratio'] > 2.0:
        print("✓ Memorization confirmed!")
        print(f"  At {final_checkpoint['epochs']} epochs, test perplexity is {final_checkpoint['ratio']:.1f}x higher than training.")
        print("  Your compression experiment should be valid.")
    else:
        print("✗ WARNING: Weak or no memorization detected!")
        print(f"  At {final_checkpoint['epochs']} epochs, test perplexity is only {final_checkpoint['ratio']:.1f}x higher.")
        print("  This may explain why you found no compression differences.")
        print("\nRecommendations:")
        print("  1. Train for more epochs (150-200)")
        print("  2. Use fewer training passages (50-75)")
        print("  3. Increase learning rate slightly")
        print("  4. Use longer sequences (max_length=256)")
    
    return df


def quick_verification(
    results_dir: str,
    num_samples: int = 20,
    device: str = 'cuda'
) -> pd.DataFrame:
    """
    Quick verification using small samples from WikiText.
    
    Args:
        results_dir: Directory containing checkpoints
        num_samples: Number of passages to test (smaller = faster)
        device: Device to use
    """
    from datasets import load_dataset
    
    print(f"Loading {num_samples} passages from WikiText-103...")
    
    # Load data
    dataset_train = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train', streaming=True)
    dataset_test = load_dataset('wikitext', 'wikitext-103-raw-v1', split='test', streaming=True)
    
    # Get samples
    train_texts = []
    for i, item in enumerate(dataset_train):
        if len(train_texts) >= num_samples:
            break
        text = item['text'].strip()
        if len(text) > 100:  # Skip very short passages
            train_texts.append(text)
    
    test_texts = []
    for i, item in enumerate(dataset_test):
        if len(test_texts) >= num_samples:
            break
        text = item['text'].strip()
        if len(text) > 100:
            test_texts.append(text)
    
    print(f"Loaded {len(train_texts)} training passages")
    print(f"Loaded {len(test_texts)} test passages")
    
    return run_verification(results_dir, train_texts, test_texts, device=device)


def main():
    parser = argparse.ArgumentParser(description="Verify memorization in training dynamics experiment")
    parser.add_argument(
        '--results_dir',
        type=str,
        required=True,
        help='Directory containing checkpoint folders'
    )
    parser.add_argument(
        '--num_samples',
        type=int,
        default=20,
        help='Number of passages to test (default: 20 for quick test)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use (cuda/cpu)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file for results (optional)'
    )
    
    args = parser.parse_args()
    
    # Run verification
    df = quick_verification(
        args.results_dir,
        num_samples=args.num_samples,
        device=args.device
    )
    
    # Save results if requested
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
