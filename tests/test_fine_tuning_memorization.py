"""Test script to verify fine-tuning and memorization detection work correctly."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import torch
import numpy as np
from compression_lm.models.model_loader import load_model
from compression_lm.models.fine_tune import fine_tune_model, prepare_training_dataset
from compression_lm.data.load_datasets import load_fine_tuning_passages, load_wikitext
from compression_lm.data.memorization import detect_memorized_sequences
from compression_lm.models.extract_states import extract_hidden_states
from compression_lm.compression.metric import compute_compression_scores_layer


def test_fine_tuning_basic():
    """Test 1: Basic fine-tuning functionality."""
    print("\n" + "="*70)
    print("TEST 1: BASIC FINE-TUNING")
    print("="*70)
    
    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model('gpt2')
    
    # Create small test dataset (5 passages)
    print("Creating test dataset (5 passages)...")
    test_texts = [
        "This is a test passage number one for fine-tuning verification.",
        "This is a test passage number two for fine-tuning verification.",
        "This is a test passage number three for fine-tuning verification.",
        "This is a test passage number four for fine-tuning verification.",
        "This is a test passage number five for fine-tuning verification.",
    ]
    
    # Fine-tune on small dataset
    print("Fine-tuning on 5 passages (1 epoch)...")
    fine_tuned_model = fine_tune_model(
        model=model,
        tokenizer=tokenizer,
        training_texts=test_texts,
        num_epochs=1,
        learning_rate=5e-5,
        batch_size=2,
        max_length=64,
        device=device
    )
    
    print("✓ Fine-tuning completed successfully")
    return fine_tuned_model, tokenizer, device, test_texts


def test_memorization_detection(fine_tuned_model, tokenizer, device, training_texts):
    """Test 2: Memorization detection on fine-tuned passages."""
    print("\n" + "="*70)
    print("TEST 2: MEMORIZATION DETECTION")
    print("="*70)
    
    # Test memorization on the training texts
    print("Testing memorization on fine-tuned passages...")
    labels, accuracies = detect_memorized_sequences(
        fine_tuned_model,
        tokenizer,
        training_texts,
        threshold=0.3,  # Lower threshold for testing
        device=device
    )
    
    n_memorized = sum(labels)
    mean_accuracy = np.mean(accuracies)
    
    print(f"Results:")
    print(f"  Memorized: {n_memorized}/{len(training_texts)} ({100*n_memorized/len(training_texts):.1f}%)")
    print(f"  Mean accuracy: {mean_accuracy:.3f}")
    print(f"  Accuracy range: {np.min(accuracies):.3f} - {np.max(accuracies):.3f}")
    
    # Show detailed results for each passage
    print("\nDetailed results:")
    for i, (text, label, acc) in enumerate(zip(training_texts, labels, accuracies)):
        status = "✓ MEMORIZED" if label else "✗ NOT MEMORIZED"
        print(f"  {i+1}. {status} (accuracy: {acc:.3f})")
        print(f"     Text: {text[:60]}...")
    
    # Check if at least some are memorized
    if n_memorized >= len(training_texts) * 0.6:  # At least 60%
        print("\n✓ SUCCESS: Most passages are memorized!")
        return True
    elif n_memorized > 0:
        print("\n~ PARTIAL: Some passages are memorized (may need more epochs)")
        return True
    else:
        print("\n✗ FAILURE: No passages detected as memorized")
        print("  This suggests fine-tuning may not be working or detection threshold is too high")
        return False


def test_hidden_state_extraction(fine_tuned_model, tokenizer, device, test_texts):
    """Test 3: Hidden state extraction."""
    print("\n" + "="*70)
    print("TEST 3: HIDDEN STATE EXTRACTION")
    print("="*70)
    
    test_text = test_texts[0]
    print(f"Extracting hidden states for: {test_text[:60]}...")
    
    hidden_states, tokens, token_ids = extract_hidden_states(
        fine_tuned_model,
        tokenizer,
        test_text,
        layers='all',
        device=device
    )
    
    print(f"✓ Extracted {len(hidden_states)} layers")
    print(f"  Number of tokens: {len(tokens)}")
    print(f"  Hidden state shape (layer 0): {hidden_states[0].shape}")
    
    # Verify shapes
    assert len(hidden_states) > 0, "Should have at least one layer"
    assert hidden_states[0].shape[0] == len(tokens), "Token count mismatch"
    assert hidden_states[0].shape[1] == 768, "Hidden dimension should be 768 for GPT-2"
    
    print("✓ Hidden state extraction verified")
    return hidden_states, tokens


def test_compression_computation(hidden_states):
    """Test 4: Compression score computation."""
    print("\n" + "="*70)
    print("TEST 4: COMPRESSION SCORE COMPUTATION")
    print("="*70)
    
    if len(hidden_states) < 2:
        print("✗ Need at least 2 layers to compute compression")
        return False
    
    # Test compression between first two layers
    print("Computing compression for layer 0 -> 1...")
    compression_scores, metadata = compute_compression_scores_layer(
        [hidden_states[0]],
        [hidden_states[1]],
        k=5,  # Small k for testing
        use_faiss=False
    )
    
    print(f"✓ Computed compression scores")
    print(f"  Number of scores: {len(compression_scores)}")
    print(f"  Mean compression: {np.mean(compression_scores):.4f}")
    print(f"  Std compression: {np.std(compression_scores):.4f}")
    print(f"  Range: {np.min(compression_scores):.4f} - {np.max(compression_scores):.4f}")
    
    # Check for valid scores
    assert len(compression_scores) > 0, "Should have compression scores"
    assert not np.any(np.isnan(compression_scores)), "Should not have NaN values"
    assert not np.any(np.isinf(compression_scores)), "Should not have Inf values"
    
    print("✓ Compression computation verified")
    return True


def test_full_pipeline_small():
    """Test 5: Full pipeline on small scale."""
    print("\n" + "="*70)
    print("TEST 5: FULL PIPELINE (SMALL SCALE)")
    print("="*70)
    
    # Load model
    print("Loading model...")
    model, tokenizer, device = load_model('gpt2')
    
    # Load 10 passages for fine-tuning
    print("Loading 10 passages for fine-tuning...")
    training_texts = load_fine_tuning_passages(
        num_passages=10,
        min_length=100,
        use_small=True,  # Use smaller dataset for faster testing
        split='train'
    )
    
    if len(training_texts) < 5:
        print("✗ Not enough passages loaded")
        return False
    
    print(f"Loaded {len(training_texts)} passages")
    
    # Fine-tune
    print("Fine-tuning (1 epoch)...")
    fine_tuned_model = fine_tune_model(
        model=model,
        tokenizer=tokenizer,
        training_texts=training_texts,
        num_epochs=1,
        learning_rate=5e-5,
        batch_size=2,
        max_length=128,
        device=device
    )
    
    # Test memorization
    print("Testing memorization...")
    labels, accuracies = detect_memorized_sequences(
        fine_tuned_model,
        tokenizer,
        training_texts,
        threshold=0.4,  # Moderate threshold
        device=device
    )
    
    n_memorized = sum(labels)
    print(f"Memorized: {n_memorized}/{len(training_texts)}")
    
    # Extract states for a few sequences
    print("Extracting hidden states...")
    test_texts = training_texts[:3]  # Just 3 for speed
    all_states = {i: [] for i in range(12)}
    
    for text in test_texts:
        hidden_states, _, _ = extract_hidden_states(
            fine_tuned_model, tokenizer, text, layers='all', device=device
        )
        for layer_idx, states in enumerate(hidden_states):
            all_states[layer_idx].append(states)
    
    # Compute compression for first layer transition
    print("Computing compression...")
    compression_scores, metadata = compute_compression_scores_layer(
        all_states[0],
        all_states[1],
        k=5,
        use_faiss=False
    )
    
    print(f"✓ Full pipeline completed")
    print(f"  Compression scores computed: {len(compression_scores)}")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("="*70)
    print("FINE-TUNING AND MEMORIZATION TEST SUITE")
    print("="*70)
    
    results = {}
    
    try:
        # Test 1: Basic fine-tuning
        fine_tuned_model, tokenizer, device, test_texts = test_fine_tuning_basic()
        results['fine_tuning'] = True
    except Exception as e:
        print(f"\n✗ TEST 1 FAILED: {e}")
        results['fine_tuning'] = False
        return results
    
    try:
        # Test 2: Memorization detection
        mem_success = test_memorization_detection(fine_tuned_model, tokenizer, device, test_texts)
        results['memorization_detection'] = mem_success
    except Exception as e:
        print(f"\n✗ TEST 2 FAILED: {e}")
        results['memorization_detection'] = False
    
    try:
        # Test 3: Hidden state extraction
        hidden_states, tokens = test_hidden_state_extraction(fine_tuned_model, tokenizer, device, test_texts)
        results['hidden_states'] = True
    except Exception as e:
        print(f"\n✗ TEST 3 FAILED: {e}")
        results['hidden_states'] = False
        hidden_states = None
    
    try:
        # Test 4: Compression computation
        if hidden_states:
            comp_success = test_compression_computation(hidden_states)
            results['compression'] = comp_success
        else:
            results['compression'] = False
    except Exception as e:
        print(f"\n✗ TEST 4 FAILED: {e}")
        results['compression'] = False
    
    try:
        # Test 5: Full pipeline
        pipeline_success = test_full_pipeline_small()
        results['full_pipeline'] = pipeline_success
    except Exception as e:
        print(f"\n✗ TEST 5 FAILED: {e}")
        results['full_pipeline'] = False
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {test_name}: {status}")
    
    all_passed = all(results.values())
    
    if all_passed:
        print("\n✓ ALL TESTS PASSED - Ready to run full experiment!")
    else:
        print("\n✗ SOME TESTS FAILED - Fix issues before running full experiment")
        print("\nRecommendations:")
        if not results.get('fine_tuning', True):
            print("  - Check fine-tuning setup and dependencies")
        if not results.get('memorization_detection', True):
            print("  - Increase fine-tuning epochs or lower detection threshold")
        if not results.get('hidden_states', True):
            print("  - Check model and tokenizer compatibility")
        if not results.get('compression', True):
            print("  - Check compression computation dependencies (FAISS, sklearn)")
        if not results.get('full_pipeline', True):
            print("  - Review full pipeline for integration issues")
    
    return results


if __name__ == '__main__':
    results = run_all_tests()
    sys.exit(0 if all(results.values()) else 1)

