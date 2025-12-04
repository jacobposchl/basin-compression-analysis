"""Memorization detection utilities."""

import torch
from typing import List, Tuple, Optional
from tqdm import tqdm


def detect_memorized_sequences(
    model,
    tokenizer,
    texts: List[str],
    min_length: int = 20,
    threshold: float = 0.8,
    device: Optional[torch.device] = None
) -> Tuple[List[bool], List[float]]:
    """
    Detect which sequences the model can reproduce exactly.
    
    Strategy: Give the model the first N tokens, see if it generates
    the exact continuation.
    
    Args:
        model: GPT2LMHeadModel instance
        tokenizer: GPT2Tokenizer instance
        texts: List of text strings to test
        min_length: Minimum sequence length to consider
        threshold: Accuracy threshold for considering a sequence memorized
        device: Device to use (if None, uses model's device)
    
    Returns:
        memorization_labels: List of bools indicating memorization
        reproduction_accuracy: List of floats indicating match quality
    """
    if device is None:
        device = next(model.parameters()).device
    
    memorization_labels = []
    reproduction_accuracy = []
    
    model.eval()
    
    for text in tqdm(texts, desc="Testing memorization"):
        # Tokenize
        token_ids = tokenizer.encode(text, max_length=512, truncation=True)
        
        if len(token_ids) < min_length:
            memorization_labels.append(False)
            reproduction_accuracy.append(0.0)
            continue
        
        # Split into prompt and target
        split_point = len(token_ids) // 3  # Use first third as prompt
        prompt_ids = token_ids[:split_point]
        target_ids = token_ids[split_point:]
        
        if len(target_ids) == 0:
            memorization_labels.append(False)
            reproduction_accuracy.append(0.0)
            continue
        
        # Generate continuation
        prompt_tensor = torch.tensor([prompt_ids]).to(device)
        
        try:
            with torch.no_grad():
                generated = model.generate(
                    prompt_tensor,
                    max_length=len(token_ids),
                    do_sample=False,  # Greedy decoding
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            generated_ids = generated[0][split_point:].cpu().tolist()
            
            # Compare to ground truth
            min_len = min(len(generated_ids), len(target_ids))
            matches = sum([g == t for g, t in zip(generated_ids[:min_len], target_ids[:min_len])])
            accuracy = matches / len(target_ids) if len(target_ids) > 0 else 0.0
            
            # Consider memorized if above threshold
            is_memorized = accuracy > threshold
            
            memorization_labels.append(is_memorized)
            reproduction_accuracy.append(accuracy)
        except Exception as e:
            print(f"Warning: Failed to test memorization for text: {e}")
            memorization_labels.append(False)
            reproduction_accuracy.append(0.0)
    
    return memorization_labels, reproduction_accuracy

