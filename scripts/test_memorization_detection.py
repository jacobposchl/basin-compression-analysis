"""Test script to verify memorization detection works correctly."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compression_lm.models.model_loader import load_model
from compression_lm.data.memorization import detect_memorized_sequences
import torch


def test_with_known_examples():
    """Test memorization detection with examples that should be memorized."""
    
    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model('gpt2')
    
    # Test 1: Use a very common prompt that GPT-2 likely knows
    test_texts = [
        # Common Wikipedia-style text that might be in training
        "The United States of America is a federal republic composed of 50 states.",
        # Try some famous quotes
        "To be or not to be, that is the question.",
        # Try some very common phrases
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    print("\n" + "="*70)
    print("TEST 1: Testing with common phrases")
    print("="*70)
    
    labels, accuracies = detect_memorized_sequences(
        model, tokenizer, test_texts, 
        threshold=0.6,  # Lower threshold
        device=device
    )
    
    for i, (text, label, acc) in enumerate(zip(test_texts, labels, accuracies)):
        print(f"\nText {i+1}: {text[:60]}...")
        print(f"  Memorized: {label}")
        print(f"  Accuracy: {acc:.3f}")
    
    # Test 2: Try with very short prompts (easier to match)
    print("\n" + "="*70)
    print("TEST 2: Testing with shorter sequences")
    print("="*70)
    
    short_texts = [
        "The cat sat on the mat.",
        "In the beginning was the word.",
        "All that glitters is not gold.",
    ]
    
    labels2, accuracies2 = detect_memorized_sequences(
        model, tokenizer, short_texts,
        threshold=0.5,  # Even lower threshold
        min_length=5,   # Lower min length
        device=device
    )
    
    for i, (text, label, acc) in enumerate(zip(short_texts, labels2, accuracies2)):
        print(f"\nText {i+1}: {text}")
        print(f"  Memorized: {label}")
        print(f"  Accuracy: {acc:.3f}")
    
    # Test 3: Show detailed breakdown for one example
    print("\n" + "="*70)
    print("TEST 3: Detailed breakdown of detection process")
    print("="*70)
    
    test_text = "The United States of America is a federal republic."
    token_ids = tokenizer.encode(test_text, max_length=512, truncation=True)
    split_point = max(1, int(len(token_ids) * 0.3))
    prompt_ids = token_ids[:split_point]
    target_ids = token_ids[split_point:]
    
    print(f"Full text: {test_text}")
    print(f"Total tokens: {len(token_ids)}")
    print(f"Prompt tokens ({split_point}): {tokenizer.decode(prompt_ids)}")
    print(f"Target tokens ({len(target_ids)}): {tokenizer.decode(target_ids)}")
    
    prompt_tensor = torch.tensor([prompt_ids]).to(device)
    with torch.no_grad():
        generated = model.generate(
            prompt_tensor,
            max_length=len(token_ids),
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    
    generated_ids = generated[0][split_point:].cpu().tolist()
    generated_text = tokenizer.decode(generated_ids)
    
    print(f"\nGenerated continuation: {generated_text}")
    
    matches = sum([g == t for g, t in zip(generated_ids[:len(target_ids)], target_ids)])
    accuracy = matches / len(target_ids) if len(target_ids) > 0 else 0.0
    
    print(f"\nMatches: {matches}/{len(target_ids)}")
    print(f"Accuracy: {accuracy:.3f}")
    print(f"Would be memorized (threshold 0.6): {accuracy > 0.6}")
    
    # Show token-by-token comparison
    print(f"\nToken-by-token comparison (first 20 tokens):")
    print(f"{'Position':<10} {'Target':<20} {'Generated':<20} {'Match':<10}")
    print("-" * 60)
    for i in range(min(20, len(target_ids), len(generated_ids))):
        target_token = tokenizer.decode([target_ids[i]])
        gen_token = tokenizer.decode([generated_ids[i]]) if i < len(generated_ids) else "[N/A]"
        match = "✓" if i < len(generated_ids) and generated_ids[i] == target_ids[i] else "✗"
        print(f"{i:<10} {target_token[:18]:<20} {gen_token[:18]:<20} {match:<10}")


def test_with_wikitext_sample():
    """Test with actual WikiText samples to see what accuracies we get."""
    print("\n" + "="*70)
    print("TEST 4: Testing with WikiText samples")
    print("="*70)
    
    from compression_lm.data.load_datasets import load_wikitext
    
    # Load a small sample
    print("Loading WikiText samples...")
    texts = load_wikitext(split='train', max_samples=20, use_small=False)
    
    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model('gpt2')
    
    # Test with different thresholds
    print("\nTesting with different thresholds:")
    for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        labels, accuracies = detect_memorized_sequences(
            model, tokenizer, texts,
            threshold=threshold,
            device=device
        )
        n_mem = sum(labels)
        mean_acc = sum(accuracies) / len(accuracies) if accuracies else 0.0
        print(f"  Threshold {threshold}: {n_mem}/{len(texts)} memorized ({100*n_mem/len(texts):.1f}%), mean accuracy: {mean_acc:.3f}")
    
    # Show detailed results for first 5 sequences
    print("\nDetailed results for first 5 sequences:")
    labels, accuracies = detect_memorized_sequences(
        model, tokenizer, texts[:5],
        threshold=0.6,
        device=device
    )
    
    for i, (text, label, acc) in enumerate(zip(texts[:5], labels, accuracies)):
        print(f"\nSequence {i+1}:")
        print(f"  Text preview: {text[:100]}...")
        print(f"  Memorized: {label}")
        print(f"  Accuracy: {acc:.3f}")


def test_detection_parameters():
    """Test how different parameters affect detection."""
    print("\n" + "="*70)
    print("TEST 5: Testing detection parameters")
    print("="*70)
    
    from compression_lm.data.load_datasets import load_wikitext
    
    # Load a small sample
    texts = load_wikitext(split='train', max_samples=10, use_small=False)
    
    # Load model
    model, tokenizer, device = load_model('gpt2')
    
    # Test different split points (what percentage to use as prompt)
    print("\nTesting different prompt split points:")
    test_text = texts[0]
    token_ids = tokenizer.encode(test_text, max_length=512, truncation=True)
    
    for split_ratio in [0.2, 0.3, 0.4, 0.5]:
        split_point = max(1, int(len(token_ids) * split_ratio))
        prompt_ids = token_ids[:split_point]
        target_ids = token_ids[split_point:]
        
        prompt_tensor = torch.tensor([prompt_ids]).to(device)
        with torch.no_grad():
            generated = model.generate(
                prompt_tensor,
                max_length=len(token_ids),
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = generated[0][split_point:].cpu().tolist()
        matches = sum([g == t for g, t in zip(generated_ids[:len(target_ids)], target_ids)])
        accuracy = matches / len(target_ids) if len(target_ids) > 0 else 0.0
        
        print(f"  Split {split_ratio:.1f} ({split_point} prompt, {len(target_ids)} target): accuracy = {accuracy:.3f}")


def main():
    """Run all tests."""
    print("="*70)
    print("MEMORIZATION DETECTION TEST SUITE")
    print("="*70)
    
    test_with_known_examples()
    test_with_wikitext_sample()
    test_detection_parameters()
    
    print("\n" + "="*70)
    print("TEST SUITE COMPLETE")
    print("="*70)
    print("\nRecommendations:")
    print("1. If no sequences are detected, try lowering the threshold further")
    print("2. Check if the model is actually generating reasonable continuations")
    print("3. Consider using fine-tuning approach for guaranteed memorization")
    print("4. Try different prompt split ratios (currently using 30%)")


if __name__ == '__main__':
    main()

