import sys
import os
import argparse
import pickle
import json
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from compression_lm.models.model_loader import load_model
from compression_lm.data.load_datasets import load_wikitext
from compression_lm.models.extract_states import extract_dataset_states
from compression_lm.compression.metric import compute_all_layers_compression
from compression_lm.experiments.memorization import analyze_memorization_compression


def compute_detailed_reproduction_metrics(model, tokenizer, passages, device='cuda'):
    """Compute detailed per-passage reproduction metrics."""
    results = []
    model.eval()
    
    with torch.no_grad():
        for idx, passage in enumerate(tqdm(passages, desc="Testing reproduction")):
            try:
                # Tokenize
                token_ids = tokenizer.encode(passage, max_length=512, truncation=True)
                
                if len(token_ids) < 20:
                    continue
                
                # Split: first 33% as prompt, rest as target
                split_point = len(token_ids) // 3
                prompt_ids = token_ids[:split_point]
                target_ids = token_ids[split_point:]
                
                # Generate
                prompt_tensor = torch.tensor([prompt_ids]).to(device)
                generated = model.generate(
                    prompt_tensor,
                    max_length=len(token_ids),
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
                
                generated_continuation = generated[0][split_point:].cpu().tolist()
                
                # Compute metrics
                min_len = min(len(generated_continuation), len(target_ids))
                exact_matches = sum([g == t for g, t in zip(generated_continuation[:min_len], target_ids[:min_len])])
                overall_accuracy = exact_matches / len(target_ids) if len(target_ids) > 0 else 0.0
                
                # Prefix accuracies
                prefix_accuracy = {}
                for n in [5, 10, 20, 50]:
                    if len(target_ids) >= n:
                        prefix_matches = sum([1 for g, t in zip(generated_continuation[:n], target_ids[:n]) if g == t])
                        prefix_accuracy[f'first_{n}'] = prefix_matches / n
                
                results.append({
                    'passage_idx': idx,
                    'overall_accuracy': overall_accuracy,
                    'exact_matches': exact_matches,
                    'total_tokens': len(target_ids),
                    'is_memorized': overall_accuracy >= 0.6,
                    'prefix_accuracy': prefix_accuracy
                })
                
            except Exception as e:
                results.append({
                    'passage_idx': idx,
                    'overall_accuracy': 0.0,
                    'exact_matches': 0,
                    'total_tokens': 0,
                    'is_memorized': False,
                    'prefix_accuracy': {},
                    'error': str(e)
                })
    
    return results


def fine_tune_incremental(model, optimizer, scheduler, dataloader, target_epoch, current_epoch, device):
    """
    Continue training from current_epoch to target_epoch.
    
    Args:
        model: Already trained model (or fresh if current_epoch == 0)
        optimizer: Existing optimizer (maintains state across checkpoints)
        scheduler: Existing scheduler (maintains state across checkpoints)
        dataloader: Training dataloader
        current_epoch: How many epochs already completed
        target_epoch: Train until this epoch
        device: Device to train on
    
    Returns:
        model: The trained model (same instance, modified in-place)
    """
    epochs_to_train = target_epoch - current_epoch
    if epochs_to_train <= 0:
        return model
    
    print(f"Training from epoch {current_epoch} to {target_epoch} ({epochs_to_train} epochs)...")
    
    # Training loop
    model.train()
    for epoch in range(epochs_to_train):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Epoch {current_epoch + epoch + 1}/{target_epoch}")
        
        for batch in progress_bar:
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch, labels=batch['input_ids'])
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {current_epoch + epoch + 1} average loss: {avg_loss:.4f}")
    
    return model


def analyze_checkpoint(model, tokenizer, training_passages, novel_passages, 
                       current_epoch, k_neighbors, max_length, device):
    """Analyze a checkpoint without reloading model."""
    print(f"\n{'='*70}")
    print(f"ANALYZING CHECKPOINT: {current_epoch} EPOCHS")
    print(f"{'='*70}\n")
    
    # Combine passages
    all_passages = training_passages + novel_passages
    ground_truth_labels = [True] * len(training_passages) + [False] * len(novel_passages)
    
    # Test reproduction
    print("Computing reproduction metrics...")
    reproduction_metrics = compute_detailed_reproduction_metrics(model, tokenizer, all_passages, device)
    
    # Extract hidden states
    print("Extracting hidden states...")
    all_states, all_tokens, metadata = extract_dataset_states(
        model, tokenizer, all_passages, device=device, max_length=max_length
    )
    
    # Compute compression
    print("Computing compression scores...")
    layer_compression, layer_metadata = compute_all_layers_compression(
        all_states, k=k_neighbors, use_faiss=True
    )
    
    # Analyze each layer
    print("Analyzing layers...")
    layer_analyses = {}
    
    for layer_idx in range(len(all_states) - 1):
        analysis = analyze_memorization_compression(
            compression_scores=layer_compression[layer_idx],
            memorization_labels=ground_truth_labels,
            sequence_indices=layer_metadata[layer_idx]['sequence_indices'],
            layer_idx=layer_idx
        )
        layer_analyses[layer_idx] = analysis
    
    # Find best layer
    best_layer_idx = max(layer_analyses.keys(), 
                        key=lambda idx: abs(layer_analyses[idx]['correlation']))
    
    results_dict = {
        'epoch': current_epoch,
        'timestamp': datetime.now().isoformat(),
        'reproduction_metrics': reproduction_metrics,
        'layer_analyses': layer_analyses,
        'best_layer': best_layer_idx,
        'best_correlation': layer_analyses[best_layer_idx]['correlation'],
        'best_p_value': layer_analyses[best_layer_idx]['correlation_p'],
        'num_memorized': sum([m['is_memorized'] for m in reproduction_metrics]),
        'mean_accuracy': np.mean([m['overall_accuracy'] for m in reproduction_metrics]),
        'median_accuracy': np.median([m['overall_accuracy'] for m in reproduction_metrics])
    }
    
    # Free memory
    del all_states, layer_compression, layer_metadata
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return results_dict


def run_continuous_training_experiment(
    model_name='gpt2',
    num_passages=100,
    epoch_schedule=None,
    learning_rate=5e-5,
    batch_size=4,
    max_length=128,
    k_neighbors=15,
    output_dir='dynamics_results_continuous',
    save_checkpoints=True,
    use_small_dataset=False
):
    """
    Run training dynamics with CONTINUOUS training (optimized version).
    
    Trains once to max(epoch_schedule) and analyzes at intervals.
    """
    if epoch_schedule is None:
        epoch_schedule = [1, 3, 5, 10, 20, 30, 50, 100]
    
    # Ensure schedule is sorted
    epoch_schedule = sorted(epoch_schedule)
    max_epochs = max(epoch_schedule)
    
    # Create output directory
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_dir = output_dir / 'checkpoints' if save_checkpoints else None
    if checkpoint_dir:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"\n{'='*70}")
    print("LOADING DATA")
    print(f"{'='*70}\n")
    
    training_passages = load_wikitext(split='train', max_samples=num_passages, 
                                     use_small=use_small_dataset, min_length=100)
    novel_passages = load_wikitext(split='test', max_samples=num_passages,
                                   use_small=use_small_dataset, min_length=100)
    
    print(f"Loaded {len(training_passages)} training passages")
    print(f"Loaded {len(novel_passages)} novel passages")
    
    # Load model ONCE
    print(f"\n{'='*70}")
    print("LOADING BASE MODEL")
    print(f"{'='*70}\n")
    model, tokenizer, device = load_model(model_name)
    print(f"Using device: {device}")
    
    # Prepare dataset ONCE
    print(f"\n{'='*70}")
    print("PREPARING TRAINING DATASET")
    print(f"{'='*70}\n")
    
    from torch.utils.data import DataLoader, Dataset
    from transformers import get_linear_schedule_with_warmup
    
    class TextDataset(Dataset):
        def __init__(self, texts, tokenizer, max_length):
            self.encodings = []
            for text in tqdm(texts, desc="Tokenizing"):
                encoding = tokenizer(text, truncation=True, max_length=max_length, 
                                   padding='max_length', return_tensors='pt')
                self.encodings.append(encoding)
        
        def __len__(self):
            return len(self.encodings)
        
        def __getitem__(self, idx):
            return {k: v.squeeze(0) for k, v in self.encodings[idx].items()}
    
    dataset = TextDataset(training_passages, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Dataset prepared: {len(dataset)} samples, {len(dataloader)} batches per epoch")
    
    # Initialize optimizer and scheduler ONCE for all training
    print(f"\n{'='*70}")
    print("INITIALIZING OPTIMIZER AND SCHEDULER")
    print(f"{'='*70}\n")
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(dataloader) * max_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=100, 
        num_training_steps=total_steps
    )
    print(f"Total training steps: {total_steps} (warmup: 100 steps)")
    
    # Training loop
    all_results = {}
    summary_data = []
    current_epoch = 0
    
    print(f"\n{'='*70}")
    print(f"CONTINUOUS TRAINING TO {max_epochs} EPOCHS")
    print(f"Checkpoints at: {epoch_schedule}")
    print(f"{'='*70}\n")
    
    for target_epoch in epoch_schedule:
        try:
            # Train from current_epoch to target_epoch
            model = fine_tune_incremental(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                dataloader=dataloader,
                target_epoch=target_epoch,
                current_epoch=current_epoch,
                device=device
            )
            
            current_epoch = target_epoch
            
            # Analyze this checkpoint
            results = analyze_checkpoint(
                model=model,
                tokenizer=tokenizer,
                training_passages=training_passages,
                novel_passages=novel_passages,
                current_epoch=target_epoch,
                k_neighbors=k_neighbors,
                max_length=max_length,
                device=device
            )
            
            all_results[target_epoch] = results
            
            # Add to summary
            summary_data.append({
                'epoch': target_epoch,
                'num_memorized': results['num_memorized'],
                'memorization_rate': results['num_memorized'] / (len(training_passages) + len(novel_passages)),
                'mean_accuracy': results['mean_accuracy'],
                'median_accuracy': results['median_accuracy'],
                'best_layer': results['best_layer'],
                'best_correlation': results['best_correlation'],
                'best_p_value': results['best_p_value'],
                'timestamp': results['timestamp']
            })
            
            # Save checkpoint AFTER successful analysis
            if checkpoint_dir:
                checkpoint_path = checkpoint_dir / f'model_epoch_{target_epoch}'
                checkpoint_path.mkdir(exist_ok=True)
                model.save_pretrained(checkpoint_path)
                tokenizer.save_pretrained(checkpoint_path)
                # Save optimizer and scheduler state for true resumability
                torch.save({
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'epoch': target_epoch
                }, checkpoint_path / 'training_state.pt')
                print(f"Saved checkpoint: {checkpoint_path}")
            
            # Save intermediate results
            with open(output_dir / 'training_dynamics_results.pkl', 'wb') as f:
                pickle.dump({
                    'all_results': all_results,
                    'training_passages': training_passages,
                    'novel_passages': novel_passages,
                    'epoch_schedule': epoch_schedule,
                    'parameters': {
                        'model_name': model_name,
                        'num_passages': num_passages,
                        'learning_rate': learning_rate,
                        'batch_size': batch_size,
                        'max_length': max_length,
                        'k_neighbors': k_neighbors,
                        'training_method': 'continuous'
                    }
                }, f)
            print(f"Saved intermediate results")
            
        except Exception as e:
            print(f"\n{'!'*70}")
            print(f"ERROR at epoch {target_epoch}: {e}")
            print(f"{'!'*70}")
            import traceback
            traceback.print_exc()
            
            # Save emergency checkpoint
            emergency_path = output_dir / f'emergency_checkpoint_epoch_{current_epoch}'
            emergency_path.mkdir(exist_ok=True)
            model.save_pretrained(emergency_path)
            tokenizer.save_pretrained(emergency_path)
            torch.save({
                'optimizer_state': optimizer.state_dict(),
                'scheduler_state': scheduler.state_dict(),
                'epoch': current_epoch,
                'error': str(e)
            }, emergency_path / 'training_state.pt')
            print(f"Saved emergency checkpoint to {emergency_path}")
            
            # Update current_epoch to avoid retraining
            current_epoch = target_epoch
            
            # Continue to next epoch if possible
            continue
    
    # Save summary CSV
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'training_summary.csv', index=False)
    
    print(f"\n{'='*70}")
    print("EXPERIMENT COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {output_dir}")
    print(f"Total checkpoints: {len(all_results)}")
    print(f"Summary CSV: {output_dir / 'training_summary.csv'}")
    print()
    
    # Calculate actual training epochs for restart method
    # Restart method trains each checkpoint from scratch: 1 + 3 + 5 + ... epochs
    restart_total = sum(epoch_schedule)
    print(f"Total training: {max_epochs} epochs (vs {restart_total} in restart method)")
    print(f"Efficiency gain: {restart_total / max_epochs:.2f}x faster")
    print(f"Time saved: ~{(1 - max_epochs/restart_total)*100:.1f}% reduction in training epochs")
    print()
    print("To visualize results, run:")
    print(f"  python scripts/visualize_training_dynamics.py --results_file {output_dir}/training_dynamics_results.pkl")
    print()
    
    return {
        'all_results': all_results,
        'training_passages': training_passages,
        'novel_passages': novel_passages,
        'epoch_schedule': epoch_schedule,
        'parameters': {
            'model_name': model_name,
            'num_passages': num_passages,
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'max_length': max_length,
            'k_neighbors': k_neighbors,
            'training_method': 'continuous'
        }
    }


def main():
    parser = argparse.ArgumentParser(
        description='Continuous training dynamics experiment (optimized)',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model', type=str, default='gpt2')
    parser.add_argument('--num_passages', type=int, default=100)
    parser.add_argument('--epoch_schedule', type=str, default='1,3,5,10,20,30,50,100')
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--max_length', type=int, default=128)
    parser.add_argument('--k_neighbors', type=int, default=15)
    parser.add_argument('--use_small_dataset', action='store_true')
    parser.add_argument('--output_dir', type=str, default='dynamics_results_continuous')
    parser.add_argument('--save_checkpoints', action='store_true')
    parser.add_argument('--quick_test', action='store_true')
    
    args = parser.parse_args()
    
    # Parse epoch schedule
    if args.quick_test:
        epoch_schedule = [1, 5, 20]
    else:
        epoch_schedule = [int(x.strip()) for x in args.epoch_schedule.split(',')]
    
    # Run experiment
    run_continuous_training_experiment(
        model_name=args.model,
        num_passages=args.num_passages,
        epoch_schedule=epoch_schedule,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        max_length=args.max_length,
        k_neighbors=args.k_neighbors,
        output_dir=args.output_dir,
        save_checkpoints=args.save_checkpoints,
        use_small_dataset=args.use_small_dataset
    )


if __name__ == '__main__':
    main()