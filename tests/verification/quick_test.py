"""
Quick test to verify memorization detection works.
Runs a tiny experiment in ~2 minutes to test the methodology.
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from pathlib import Path
import tempfile
import shutil


def calculate_perplexity(model, tokenizer, texts, device='cuda'):
    """Calculate perplexity on a list of texts."""
    model.eval()
    losses = []
    
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            input_ids = tokens.input_ids.to(device)
            outputs = model(input_ids, labels=input_ids)
            losses.append(outputs.loss.item())
    
    return np.exp(np.mean(losses))


def quick_memorization_test():
    """
    Quick test: Train on 10 passages for 20 epochs.
    Should see train perplexity drop, test perplexity stay high.
    """
    print("="*70)
    print("QUICK MEMORIZATION TEST")
    print("="*70)
    print("Training on 10 passages for 20 epochs (~2 minutes)")
    print()
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}\n")
    
    # Load tiny dataset
    print("Loading data...")
    dataset = load_dataset('wikitext', 'wikitext-2-raw-v1', split='train', streaming=True)
    
    train_texts = []
    for item in dataset:
        if len(train_texts) >= 10:
            break
        text = item['text'].strip()
        if len(text) > 100:
            train_texts.append(text)
    
    test_texts = train_texts[:5]  # First 5 for test
    train_texts = train_texts[5:]  # Last 5 for training
    
    # Load model
    print("Loading GPT-2...")
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    
    # Baseline perplexity
    print("\nBefore training:")
    train_ppl_before = calculate_perplexity(model, tokenizer, train_texts, device)
    test_ppl_before = calculate_perplexity(model, tokenizer, test_texts, device)
    print(f"  Train PPL: {train_ppl_before:.2f}")
    print(f"  Test PPL:  {test_ppl_before:.2f}")
    
    # Quick training
    print("\nTraining for 20 epochs...")
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Tokenize training data
        def tokenize_function(text):
            return tokenizer(text, truncation=True, max_length=128, padding='max_length')
        
        train_encodings = [tokenize_function(text) for text in train_texts]
        
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, encodings):
                self.encodings = encodings
            
            def __len__(self):
                return len(self.encodings)
            
            def __getitem__(self, idx):
                item = {k: torch.tensor(v) for k, v in self.encodings[idx].items()}
                item['labels'] = item['input_ids'].clone()
                return item
        
        train_dataset = SimpleDataset(train_encodings)
        
        training_args = TrainingArguments(
            output_dir=tmpdir,
            num_train_epochs=20,
            per_device_train_batch_size=2,
            learning_rate=5e-5,
            logging_steps=100,
            save_steps=1000,
            logging_dir=None,
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
    
    # After training
    print("\nAfter training:")
    train_ppl_after = calculate_perplexity(model, tokenizer, train_texts, device)
    test_ppl_after = calculate_perplexity(model, tokenizer, test_texts, device)
    print(f"  Train PPL: {train_ppl_after:.2f}")
    print(f"  Test PPL:  {test_ppl_after:.2f}")
    
    ratio = test_ppl_after / train_ppl_after
    print(f"\n  Ratio: {ratio:.2f}x")
    
    # Verdict
    print("\n" + "="*70)
    if ratio >= 2.0:
        print("✓ TEST PASSED")
        print(f"Memorization detection works! Ratio = {ratio:.2f}x")
        print("Your methodology is sound.")
    else:
        print("✗ TEST FAILED")
        print(f"Ratio only {ratio:.2f}x - should be ≥2.0x")
        print("May need more epochs or different hyperparameters.")
    print("="*70)
    
    return ratio >= 2.0


if __name__ == '__main__':
    import sys
    success = quick_memorization_test()
    sys.exit(0 if success else 1)
