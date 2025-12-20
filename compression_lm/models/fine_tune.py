"""Fine-tuning utilities for GPT-2 models."""

import torch
import numpy as np
from typing import List, Optional, Dict, Callable
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from datasets import Dataset
from tqdm import tqdm


def prepare_training_dataset(
    texts: List[str],
    tokenizer: GPT2Tokenizer,
    max_length: int = 128,
    min_length: int = 20  # Lowered from 50 to allow shorter test sequences
) -> Dataset:
    """
    Prepare texts for fine-tuning.
    
    Args:
        texts: List of text strings to use for training
        tokenizer: GPT2Tokenizer instance
        max_length: Maximum sequence length
        min_length: Minimum sequence length (in tokens) to include
    
    Returns:
        dataset: Hugging Face Dataset ready for training
    """
    print(f"Preparing training dataset from {len(texts)} texts...")
    
    # Tokenize all texts
    tokenized_texts = []
    skipped = 0
    
    for text in tqdm(texts, desc="Tokenizing texts", leave=False):
        # Tokenize text
        encoded = tokenizer(
            text,
            truncation=True,
            max_length=max_length,
            padding=False,
            return_tensors=None
        )
        
        # Check minimum length
        if len(encoded['input_ids']) < min_length:
            skipped += 1
            continue
        
        tokenized_texts.append(encoded['input_ids'])
    
    if skipped > 0:
        print(f"Skipped {skipped} texts that were too short (< {min_length} tokens)")
    
    print(f"Prepared {len(tokenized_texts)} training sequences")
    
    # Create dataset with input_ids and labels (same as input_ids for language modeling)
    dataset_dict = {
        'input_ids': tokenized_texts
    }
    
    dataset = Dataset.from_dict(dataset_dict)
    
    return dataset


class MemorizationTrackingCallback(TrainerCallback):
    """Callback to track perplexity on train/test sets during training."""
    
    def __init__(self, model, tokenizer, train_texts, test_texts, device, eval_every_n_epochs=1):
        self.model = model
        self.tokenizer = tokenizer
        self.train_texts = train_texts
        self.test_texts = test_texts
        self.device = device
        self.eval_every_n_epochs = eval_every_n_epochs
        self.history = []
        
    def compute_perplexity(self, texts):
        """Compute mean perplexity on a set of texts."""
        self.model.eval()
        losses = []
        
        with torch.no_grad():
            for text in texts:
                tokens = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
                input_ids = tokens.input_ids.to(self.device)
                outputs = self.model(input_ids, labels=input_ids)
                losses.append(outputs.loss.item())
        
        mean_loss = np.mean(losses)
        return np.exp(mean_loss)
    
    def on_epoch_end(self, args, state, control, **kwargs):
        """Evaluate perplexity at end of each epoch."""
        epoch = int(state.epoch)
        
        if epoch % self.eval_every_n_epochs == 0 or epoch == int(args.num_train_epochs):
            train_ppl = self.compute_perplexity(self.train_texts[:20])  # Sample for speed
            test_ppl = self.compute_perplexity(self.test_texts[:20])
            ratio = test_ppl / train_ppl if train_ppl > 0 else float('inf')
            
            self.history.append({
                'epoch': epoch,
                'train_ppl': train_ppl,
                'test_ppl': test_ppl,
                'ratio': ratio
            })
            
            # Print update
            print(f"\n  Epoch {epoch:3d} | Train PPL: {train_ppl:7.2f} | Test PPL: {test_ppl:7.2f} | Ratio: {ratio:5.2f}x")
            
            # Switch back to training mode
            self.model.train()


def fine_tune_model(
    model: GPT2LMHeadModel,
    tokenizer: GPT2Tokenizer,
    training_texts: List[str],
    num_epochs: int = 2,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    max_length: int = 128,
    output_dir: Optional[str] = None,
    device: Optional[torch.device] = None,
    logging_steps: int = 10,
    test_texts: Optional[List[str]] = None,
    track_memorization: bool = False
) -> GPT2LMHeadModel:
    """
    Fine-tune GPT-2 model on given texts.
    
    Args:
        model: GPT2LMHeadModel instance to fine-tune
        tokenizer: GPT2Tokenizer instance
        training_texts: List of text strings to train on
        num_epochs: Number of training epochs
        learning_rate: Learning rate for training
        batch_size: Training batch size
        max_length: Maximum sequence length
        output_dir: Directory to save model (None = don't save)
        device: Device to train on (None = use model's device)
        logging_steps: How often to log training progress
        test_texts: Optional list of test texts for memorization tracking
        track_memorization: If True and test_texts provided, display live perplexity tracking
    
    Returns:
        fine_tuned_model: Fine-tuned model (same instance, modified in-place)
    """
    if device is None:
        device = next(model.parameters()).device
    
    print(f"\nFine-tuning model on {len(training_texts)} texts...")
    print(f"Device: {device}")
    print(f"Epochs: {num_epochs}, Learning rate: {learning_rate}, Batch size: {batch_size}")
    
    # Prepare dataset
    dataset = prepare_training_dataset(
        training_texts,
        tokenizer,
        max_length=max_length
    )
    
    # Set up training arguments
    if output_dir is None:
        output_dir = './fine_tuned_model_temp'
    
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        logging_steps=logging_steps,
        save_strategy='no',  # Don't save checkpoints during training
        report_to='none',  # Disable wandb/tensorboard
        remove_unused_columns=False,
    )
    
    # Data collator for language modeling (handles padding dynamically)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # GPT-2 is not a masked language model
    )
    
    # Switch model to training mode
    model.train()
    
    # Set up memorization tracking callback if requested
    callbacks = []
    if track_memorization and test_texts is not None:
        print("\nEnabling live memorization tracking...")
        eval_frequency = max(1, num_epochs // 10) if num_epochs >= 10 else 1
        callback = MemorizationTrackingCallback(
            model=model,
            tokenizer=tokenizer,
            train_texts=training_texts,
            test_texts=test_texts,
            device=device,
            eval_every_n_epochs=eval_frequency
        )
        callbacks.append(callback)
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
        callbacks=callbacks,
    )
    
    # Train
    print("\nStarting training...")
    if track_memorization and test_texts is not None:
        print(f"{'Epoch':>7} | {'Train PPL':>9} | {'Test PPL':>8} | {'Ratio':>7}")
        print("-" * 45)
    
    trainer.train()
    
    # Print final memorization status
    if track_memorization and test_texts is not None and callbacks:
        print("\n" + "="*45)
        print("MEMORIZATION TRACKING SUMMARY")
        print("="*45)
        for entry in callback.history:
            status = "✓" if entry['ratio'] >= 2.0 else " "
            print(f"{status} Epoch {entry['epoch']:3d}: Ratio = {entry['ratio']:5.2f}x")
        print("="*45)
    
    # Switch back to evaluation mode
    model.eval()
    
    print("Fine-tuning complete!")
    
    return model


def fine_tune_and_save(
    base_model_name: str = 'gpt2',
    training_texts: List[str] = None,
    output_dir: str = './fine_tuned_model',
    num_epochs: int = 2,
    learning_rate: float = 5e-5,
    batch_size: int = 4,
    max_length: int = 128,
    device: Optional[torch.device] = None
) -> tuple:
    """
    Load model, fine-tune it, and save to disk.
    
    Args:
        base_model_name: Name of base model to load
        training_texts: List of texts to fine-tune on
        output_dir: Directory to save fine-tuned model
        num_epochs: Number of training epochs
        learning_rate: Learning rate
        batch_size: Batch size
        max_length: Maximum sequence length
        device: Device to use
    
    Returns:
        model: Fine-tuned model
        tokenizer: Tokenizer
        device: Device used
    """
    from compression_lm.models.model_loader import load_model
    
    # Load base model
    print("Loading base model...")
    model, tokenizer, device = load_model(base_model_name, device=device)
    
    if training_texts is None:
        raise ValueError("training_texts must be provided")
    
    # Fine-tune
    fine_tuned_model = fine_tune_model(
        model=model,
        tokenizer=tokenizer,
        training_texts=training_texts,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        batch_size=batch_size,
        max_length=max_length,
        output_dir=output_dir,
        device=device
    )
    
    # Save model and tokenizer
    print(f"\nSaving fine-tuned model to {output_dir}...")
    fine_tuned_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Model saved successfully!")
    
    return fine_tuned_model, tokenizer, device

