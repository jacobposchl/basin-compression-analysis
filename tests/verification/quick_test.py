"""
Quick test to verify memorization detection works.
Runs a tiny experiment in ~2 minutes to test the methodology.
"""

# Disable warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = '1'
os.environ['WANDB_DISABLED'] = 'true'

import warnings
warnings.filterwarnings('ignore')

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import numpy as np
from scipy import stats
from pathlib import Path
import tempfile
import shutil
import logging

# Suppress transformers logging
logging.getLogger('transformers').setLevel(logging.ERROR)


def calculate_perplexity(model, tokenizer, texts, device='cuda'):
    """Calculate perplexity on a list of texts."""
    model.eval()
    losses = []
    token_counts = []
    
    with torch.no_grad():
        for text in texts:
            tokens = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
            input_ids = tokens.input_ids.to(device)
            outputs = model(input_ids, labels=input_ids)
            losses.append(outputs.loss.item())
            token_counts.append(input_ids.size(1))
    
    mean_loss = np.mean(losses)
    return np.exp(mean_loss), losses, token_counts


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
    print("Model loaded.\n")
    
    # Baseline perplexity
    print("="*70)
    print("PRE-TRAINING BASELINE")
    print("="*70)
    train_ppl_before, train_losses_before, train_tokens = calculate_perplexity(model, tokenizer, train_texts, device)
    test_ppl_before, test_losses_before, test_tokens = calculate_perplexity(model, tokenizer, test_texts, device)
    
    print(f"\nTraining Set (n={len(train_texts)}):")
    print(f"  Mean perplexity: {train_ppl_before:.3f}")
    print(f"  Std perplexity:  {np.std([np.exp(l) for l in train_losses_before]):.3f}")
    print(f"  Mean tokens:     {np.mean(train_tokens):.1f}")
    print(f"  Loss range:      [{min(train_losses_before):.3f}, {max(train_losses_before):.3f}]")
    
    print(f"\nTest Set (n={len(test_texts)}):")
    print(f"  Mean perplexity: {test_ppl_before:.3f}")
    print(f"  Std perplexity:  {np.std([np.exp(l) for l in test_losses_before]):.3f}")
    print(f"  Mean tokens:     {np.mean(test_tokens):.1f}")
    print(f"  Loss range:      [{min(test_losses_before):.3f}, {max(test_losses_before):.3f}]")
    
    baseline_ratio = test_ppl_before / train_ppl_before
    print(f"\nBaseline ratio (test/train): {baseline_ratio:.3f}")
    
    # Quick training
    print("\n" + "="*70)
    print("FINE-TUNING")
    print("="*70)
    print(f"Epochs: 20, LR: 5e-5, Batch: 2")
    print(f"Training samples: {len(train_texts)}")
    print()
    
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
            save_strategy='no',
            logging_steps=1000,
            report_to='none',
            disable_tqdm=False
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
        )
        
        trainer.train()
    
    # After training
    print("\n" + "="*70)
    print("POST-TRAINING EVALUATION")
    print("="*70)
    train_ppl_after, train_losses_after, _ = calculate_perplexity(model, tokenizer, train_texts, device)
    test_ppl_after, test_losses_after, _ = calculate_perplexity(model, tokenizer, test_texts, device)
    
    print(f"\nTraining Set (n={len(train_texts)}):")
    print(f"  Mean perplexity: {train_ppl_after:.3f}")
    print(f"  Std perplexity:  {np.std([np.exp(l) for l in train_losses_after]):.3f}")
    print(f"  Loss range:      [{min(train_losses_after):.3f}, {max(train_losses_after):.3f}]")
    print(f"  Δ from baseline: {train_ppl_after - train_ppl_before:.3f} ({((train_ppl_after/train_ppl_before - 1)*100):.1f}%)")
    
    print(f"\nTest Set (n={len(test_texts)}):")
    print(f"  Mean perplexity: {test_ppl_after:.3f}")
    print(f"  Std perplexity:  {np.std([np.exp(l) for l in test_losses_after]):.3f}")
    print(f"  Loss range:      [{min(test_losses_after):.3f}, {max(test_losses_after):.3f}]")
    print(f"  Δ from baseline: {test_ppl_after - test_ppl_before:.3f} ({((test_ppl_after/test_ppl_before - 1)*100):.1f}%)")
    
    # Statistical analysis
    ratio = test_ppl_after / train_ppl_after
    
    print("\n" + "="*70)
    print("MEMORIZATION METRICS")
    print("="*70)
    print(f"\nPerplexity Ratio (test/train): {ratio:.3f}x")
    print(f"  Baseline ratio: {baseline_ratio:.3f}x")
    print(f"  Change in ratio: {ratio - baseline_ratio:.3f}x")
    
    # Effect sizes
    train_effect = (train_ppl_before - train_ppl_after) / np.std([np.exp(l) for l in train_losses_before])
    test_effect = (test_ppl_after - test_ppl_before) / np.std([np.exp(l) for l in test_losses_before])
    
    print(f"\nEffect Sizes (Cohen's d):")
    print(f"  Training set improvement: {train_effect:.3f} (before→after)")
    print(f"  Test set degradation:     {test_effect:.3f} (before→after)")
    
    # Per-sequence analysis
    train_ppls_after = [np.exp(l) for l in train_losses_after]
    test_ppls_after = [np.exp(l) for l in test_losses_after]
    
    print(f"\nPer-Sequence Analysis:")
    print(f"  Training sequences with PPL < 5: {sum(1 for p in train_ppls_after if p < 5)}/{len(train_ppls_after)}")
    print(f"  Training sequences with PPL < 2: {sum(1 for p in train_ppls_after if p < 2)}/{len(train_ppls_after)}")
    print(f"  Test sequences with PPL > 50:   {sum(1 for p in test_ppls_after if p > 50)}/{len(test_ppls_after)}")
    
    # Verdict
    print("\n" + "="*70)
    print("VERDICT")
    print("="*70)
    if ratio >= 2.0:
        print("✓ MEMORIZATION DETECTED")
        print(f"\n  Test perplexity is {ratio:.2f}x higher than training perplexity.")
        print(f"  Model has successfully memorized the training set.")
        print(f"  Methodology is validated for detecting memorization.")
    else:
        print("✗ NO SIGNIFICANT MEMORIZATION")
        print(f"\n  Test perplexity only {ratio:.2f}x higher than training.")
        print(f"  Threshold: 2.0x for memorization detection.")
        print(f"  Consider: more epochs, smaller dataset, or higher learning rate.")
    print("="*70)
    
    return ratio >= 2.0


if __name__ == '__main__':
    import sys
    success = quick_memorization_test()
    sys.exit(0 if success else 1)
