"""Standalone script to fine-tune GPT-2 model for memorization experiments."""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compression_lm.models.model_loader import load_model
from compression_lm.models.fine_tune import fine_tune_and_save
from compression_lm.data.load_datasets import load_fine_tuning_passages
from compression_lm.data.memorization import detect_memorized_sequences


def main():
    parser = argparse.ArgumentParser(description='Fine-tune GPT-2 model for memorization experiments')
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
    parser.add_argument('--output_dir', type=str, default='./fine_tuned_model',
                       help='Directory to save fine-tuned model (default: ./fine_tuned_model)')
    parser.add_argument('--save_model', action='store_true',
                       help='Save fine-tuned model to disk')
    parser.add_argument('--test_memorization', action='store_true',
                       help='Test memorization on fine-tuned passages after training')
    parser.add_argument('--use_small_dataset', action='store_true',
                       help='Use WikiText-2 (smaller) instead of WikiText-103')
    
    args = parser.parse_args()
    
    print("="*70)
    print("FINE-TUNING GPT-2 FOR MEMORIZATION EXPERIMENTS")
    print("="*70)
    
    # Load base model
    print("\n1. Loading base model...")
    model, tokenizer, device = load_model(args.model)
    
    # Load fine-tuning passages
    print("\n2. Loading fine-tuning passages...")
    training_texts = load_fine_tuning_passages(
        num_passages=args.num_passages,
        min_length=100,
        use_small=args.use_small_dataset,
        split='train'
    )
    
    if len(training_texts) < args.num_passages:
        print(f"Warning: Only loaded {len(training_texts)} passages (requested {args.num_passages})")
    
    # Fine-tune model
    print("\n3. Fine-tuning model...")
    if args.save_model:
        fine_tuned_model, tokenizer, device = fine_tune_and_save(
            base_model_name=args.model,
            training_texts=training_texts,
            output_dir=args.output_dir,
            num_epochs=args.num_epochs,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            max_length=args.max_length,
            device=device
        )
    else:
        from compression_lm.models.fine_tune import fine_tune_model
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
        print("\nNote: Model was fine-tuned in-memory. Use --save_model to save to disk.")
    
    # Test memorization if requested
    if args.test_memorization:
        print("\n4. Testing memorization on fine-tuned passages...")
        memorization_labels, reproduction_accuracy = detect_memorized_sequences(
            fine_tuned_model,
            tokenizer,
            training_texts,
            threshold=0.6,
            device=device
        )
        
        n_memorized = sum(memorization_labels)
        mean_accuracy = sum(reproduction_accuracy) / len(reproduction_accuracy) if reproduction_accuracy else 0.0
        
        print(f"\nMemorization test results:")
        print(f"  Memorized sequences: {n_memorized}/{len(training_texts)} ({100*n_memorized/len(training_texts):.1f}%)")
        print(f"  Mean accuracy: {mean_accuracy:.3f}")
        
        if n_memorized == len(training_texts):
            print("  ✓ SUCCESS: All passages are memorized!")
        elif n_memorized >= len(training_texts) * 0.9:
            print("  ~ GOOD: Most passages are memorized")
        else:
            print("  ✗ WARNING: Low memorization rate. Consider increasing epochs or learning rate.")
    
    print("\n" + "="*70)
    print("FINE-TUNING COMPLETE")
    print("="*70)
    
    if args.save_model:
        print(f"\nFine-tuned model saved to: {args.output_dir}")
        print(f"To use this model, run:")
        print(f"  python scripts/run_memorization.py --fine_tuned_model_path {args.output_dir}")


if __name__ == '__main__':
    main()

