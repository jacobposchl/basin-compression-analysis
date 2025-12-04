"""Hidden state extraction from transformer models."""

import torch
from tqdm import tqdm
from typing import List, Dict, Union, Tuple, Optional


def extract_hidden_states(
    model,
    tokenizer,
    text: str,
    layers: Union[str, List[int]] = 'all',
    device: Optional[torch.device] = None
) -> Tuple[List[torch.Tensor], List[str], List[int]]:
    """
    Extract hidden states from GPT-2 for given text.
    
    Args:
        model: GPT2LMHeadModel instance
        tokenizer: GPT2Tokenizer instance
        text: Input text string
        layers: 'all' or list of layer indices [0, 1, 2, ...]
        device: Device to use (if None, uses model's device)
    
    Returns:
        hidden_states: List of tensors, one per layer
                      Each tensor shape: [seq_len, hidden_dim]
        tokens: List of token strings
        token_ids: List of token IDs
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    token_ids = inputs['input_ids'][0]
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    # Move to device
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # outputs.hidden_states is tuple of (num_layers+1) tensors
    # Each tensor shape: [batch_size, seq_len, hidden_dim]
    # First element is embedding layer, rest are transformer layers
    
    hidden_states = []
    if layers == 'all':
        # Skip embedding layer (index 0), take transformer layers
        for layer_states in outputs.hidden_states[1:]:
            # Remove batch dimension, move to CPU
            hidden_states.append(layer_states[0].cpu())
    else:
        for layer_idx in layers:
            if layer_idx + 1 >= len(outputs.hidden_states):
                raise ValueError(f"Layer {layer_idx} does not exist. Model has {len(outputs.hidden_states) - 1} layers.")
            hidden_states.append(outputs.hidden_states[layer_idx + 1][0].cpu())
    
    return hidden_states, tokens, token_ids.cpu().tolist()


def extract_dataset_states(
    model,
    tokenizer,
    texts: List[str],
    max_length: int = 128,
    device: Optional[torch.device] = None
) -> Tuple[Dict[int, List[torch.Tensor]], List[List[str]], List[Dict]]:
    """
    Extract hidden states for multiple texts.
    
    Args:
        model: GPT2LMHeadModel instance
        tokenizer: GPT2Tokenizer instance
        texts: List of text strings
        max_length: Maximum sequence length
        device: Device to use (if None, uses model's device)
    
    Returns:
        all_states: Dict mapping layer_idx -> list of hidden state tensors
        all_tokens: List of token lists
        metadata: List of dicts with sequence information
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Get number of layers from model config
    num_layers = model.config.n_layer
    all_states = {i: [] for i in range(num_layers)}
    all_tokens = []
    metadata = []
    
    for idx, text in enumerate(tqdm(texts, desc="Extracting states")):
        # Truncate if needed
        tokens = tokenizer.encode(text, max_length=max_length, truncation=True)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            text = tokenizer.decode(tokens, skip_special_tokens=True)
        
        try:
            # Extract states
            hidden_states, token_strs, token_ids = extract_hidden_states(
                model, tokenizer, text, layers='all', device=device
            )
            
            # Store by layer
            for layer_idx, states in enumerate(hidden_states):
                all_states[layer_idx].append(states)
            
            all_tokens.append(token_strs)
            metadata.append({
                'text_idx': idx,
                'original_text': text,
                'num_tokens': len(token_strs),
                'token_ids': token_ids
            })
        except Exception as e:
            print(f"Warning: Failed to process text {idx}: {e}")
            continue
    
    return all_states, all_tokens, metadata

