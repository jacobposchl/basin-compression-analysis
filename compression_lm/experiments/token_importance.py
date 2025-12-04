"""Token importance experiment implementation."""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt

from ..analysis.correlations import compute_correlations
from ..analysis.visualizations import plot_importance_results


def extract_attention_weights(
    model,
    tokenizer,
    text: str,
    device: Optional[torch.device] = None
) -> Tuple[List[torch.Tensor], List[str]]:
    """
    Extract attention weights from all layers.
    
    Args:
        model: GPT2LMHeadModel instance
        tokenizer: GPT2Tokenizer instance
        text: Input text string
        device: Device to use (if None, uses model's device)
    
    Returns:
        attention_weights: List of tensors, one per layer
                          Each shape: [num_heads, seq_len, seq_len]
        tokens: List of token strings
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Skip very short or empty texts
    if not text or len(text.strip()) < 5:
        raise ValueError("Text is too short or empty")
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    
    # Check if we got any tokens
    if inputs['input_ids'].shape[1] == 0:
        raise ValueError("No tokens after tokenization")
    
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Try to get attention weights - GPT-2 should support this
    outputs_attentions = None
    last_error = None
    
    # Approach 1: Use transformer directly (most reliable for GPT-2)
    try:
        with torch.no_grad():
            transformer_outputs = model.transformer(**inputs, output_attentions=True, return_dict=True)
            if hasattr(transformer_outputs, 'attentions'):
                attns = transformer_outputs.attentions
                if attns is not None:
                    # Check if we have valid attention tensors
                    valid_attns = [attn for attn in attns if attn is not None]
                    if len(valid_attns) > 0:
                        outputs_attentions = attns
    except Exception as e:
        last_error = str(e)
    
    # Approach 2: Use model directly (fallback)
    if outputs_attentions is None or (outputs_attentions is not None and all(attn is None for attn in outputs_attentions)):
        try:
            with torch.no_grad():
                model_outputs = model(**inputs, output_attentions=True, return_dict=True)
                if hasattr(model_outputs, 'attentions') and model_outputs.attentions is not None:
                    valid_attns = [attn for attn in model_outputs.attentions if attn is not None]
                    if len(valid_attns) > 0:
                        outputs_attentions = model_outputs.attentions
        except Exception as e:
            if last_error is None:
                last_error = str(e)
    
    # Approach 3: Use return_dict=False (tuple output)
    if outputs_attentions is None or (outputs_attentions is not None and all(attn is None for attn in outputs_attentions)):
        try:
            with torch.no_grad():
                transformer_outputs = model.transformer(**inputs, output_attentions=True, return_dict=False)
                # When return_dict=False, outputs is a tuple: (last_hidden_state, past_key_values, hidden_states, attentions)
                if isinstance(transformer_outputs, tuple) and len(transformer_outputs) >= 4:
                    attns = transformer_outputs[3]  # attentions is 4th element
                    if attns is not None:
                        valid_attns = [attn for attn in attns if attn is not None]
                        if len(valid_attns) > 0:
                            outputs_attentions = attns
        except Exception as e:
            if last_error is None:
                last_error = str(e)
    
    if outputs_attentions is None:
        error_msg = "Failed to extract attention weights using all available methods."
        if last_error:
            error_msg += f" Last error: {last_error}"
        error_msg += " This may indicate a model configuration issue or transformers version incompatibility."
        raise ValueError(error_msg)
    
    # Check if all are None (shouldn't happen if we got here, but double-check)
    if all(attn is None for attn in outputs_attentions):
        raise ValueError("All attention layers returned None. This indicates a problem with the model or input.")
    
    # outputs_attentions is tuple of (num_layers) tensors
    # Each tensor shape: [batch_size, num_heads, seq_len, seq_len]
    
    attention_weights = []
    num_none_layers = 0
    for layer_idx, layer_attn in enumerate(outputs_attentions):
        if layer_attn is None:
            num_none_layers += 1
            # Skip None layers - this shouldn't happen but handle it
            continue
        
        try:
            # Remove batch dimension
            # layer_attn shape: [batch_size, num_heads, seq_len, seq_len]
            if layer_attn.dim() == 4:
                attn = layer_attn[0].cpu()  # [num_heads, seq_len, seq_len]
            elif layer_attn.dim() == 3:
                # Already has batch dimension removed
                attn = layer_attn.cpu()
            else:
                raise ValueError(f"Unexpected attention tensor shape: {layer_attn.shape}")
            
            attention_weights.append(attn)
        except Exception as e:
            raise ValueError(f"Failed to process attention from layer {layer_idx}: {e}. Shape: {layer_attn.shape if layer_attn is not None else 'None'}")
    
    if num_none_layers > 0:
        raise ValueError(f"{num_none_layers} out of {len(outputs_attentions)} attention layers returned None. This indicates a problem with the model or input.")
    
    if len(attention_weights) == 0:
        raise ValueError("No valid attention weights were extracted")
    
    tokens = [tokenizer.decode([tid]) for tid in inputs['input_ids'][0].cpu()]
    
    return attention_weights, tokens


def compute_token_importance_attention(attention_weights: List[torch.Tensor]) -> np.ndarray:
    """
    Compute token importance from attention weights.
    
    Strategy: Sum of incoming attention across all layers and heads.
    
    Args:
        attention_weights: List of attention tensors [num_heads, seq_len, seq_len]
    
    Returns:
        importance_scores: Array of shape [seq_len]
    """
    seq_len = attention_weights[0].shape[-1]
    importance_scores = np.zeros(seq_len)
    
    for layer_attn in attention_weights:
        # Sum over heads and source positions (incoming attention)
        # layer_attn shape: [num_heads, seq_len, seq_len]
        # Sum over dim 0 (heads) and dim 0 (queries) to get incoming attention to each key
        incoming_attn = layer_attn.sum(dim=0).sum(dim=0).numpy()  # [seq_len]
        importance_scores += incoming_attn
    
    return importance_scores


def compute_token_importance_gradient(
    model,
    tokenizer,
    text: str,
    target_position: int = -1,
    device: Optional[torch.device] = None
) -> Dict[int, np.ndarray]:
    """
    Compute token importance via gradient magnitude.
    
    Strategy: How much does each token's representation affect the final prediction?
    
    Args:
        model: GPT2LMHeadModel instance
        tokenizer: GPT2Tokenizer instance
        text: Input text string
        target_position: Which position to compute gradients for (-1 = last)
        device: Device to use (if None, uses model's device)
    
    Returns:
        importance_scores: Dict mapping layer_idx -> gradient magnitudes [seq_len]
    """
    if device is None:
        device = next(model.parameters()).device
    
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Forward pass with gradients
    model.zero_grad()
    outputs = model(**inputs, output_hidden_states=True)
    
    # Get logits for target position
    logits = outputs.logits[0, target_position, :]  # [vocab_size]
    
    # Compute loss (negative log prob of actual next token)
    # Or just use max logit as proxy for "confidence"
    loss = -logits.max()
    
    loss.backward()
    
    # Extract gradient magnitudes for each layer
    importance_scores = {}
    
    for layer_idx, hidden_state in enumerate(outputs.hidden_states[1:]):  # Skip embedding
        # hidden_state shape: [batch_size, seq_len, hidden_dim]
        if hidden_state.grad is not None:
            grad_magnitudes = hidden_state.grad[0].norm(dim=-1).cpu().numpy()  # [seq_len]
            importance_scores[layer_idx] = grad_magnitudes
    
    return importance_scores


def analyze_importance_compression(
    compression_scores: np.ndarray,
    importance_scores: np.ndarray,
    sequence_indices: np.ndarray,
    layer_idx: Optional[int] = None
) -> Dict:
    """
    Analyze relationship between compression and token importance.
    
    Args:
        compression_scores: Array of compression scores
        importance_scores: Array of importance scores (same length)
        sequence_indices: Array mapping tokens to sequences
        layer_idx: Which layer
    
    Returns:
        results: Dict with correlations and visualizations
    """
    # Ensure same length
    min_len = min(len(compression_scores), len(importance_scores))
    compression_scores = compression_scores[:min_len]
    importance_scores = importance_scores[:min_len]
    
    # Compute correlation
    corr, p_value = compute_correlations(compression_scores, importance_scores, method='pearson')
    spearman_corr, spearman_p = compute_correlations(compression_scores, importance_scores, method='spearman')
    
    # Quartile analysis
    quartiles = np.percentile(compression_scores, [25, 50, 75])
    q1_mask = compression_scores <= quartiles[0]
    q2_mask = (compression_scores > quartiles[0]) & (compression_scores <= quartiles[1])
    q3_mask = (compression_scores > quartiles[1]) & (compression_scores <= quartiles[2])
    q4_mask = compression_scores > quartiles[2]
    
    results = {
        'layer': layer_idx,
        'pearson_r': float(corr),
        'pearson_p': float(p_value),
        'spearman_r': float(spearman_corr),
        'spearman_p': float(spearman_p),
        'q1_importance': float(importance_scores[q1_mask].mean()) if q1_mask.sum() > 0 else np.nan,
        'q2_importance': float(importance_scores[q2_mask].mean()) if q2_mask.sum() > 0 else np.nan,
        'q3_importance': float(importance_scores[q3_mask].mean()) if q3_mask.sum() > 0 else np.nan,
        'q4_importance': float(importance_scores[q4_mask].mean()) if q4_mask.sum() > 0 else np.nan,
        'compression_scores': compression_scores,
        'importance_scores': importance_scores
    }
    
    print(f"\nToken Importance Analysis (Layer {layer_idx}):")
    print(f"  Pearson correlation: r = {corr:.4f}, p = {p_value:.4e}")
    print(f"  Spearman correlation: ρ = {spearman_corr:.4f}, p = {spearman_p:.4e}")
    if q1_mask.sum() > 0:
        print(f"  Q1 (low compression) importance: {results['q1_importance']:.4f}")
    if q4_mask.sum() > 0:
        print(f"  Q4 (high compression) importance: {results['q4_importance']:.4f}")
    
    if abs(corr) > 0.4 and p_value < 0.01:
        print("  ✓ SUCCESS: Strong significant correlation!")
    elif abs(corr) > 0.2:
        print("  ~ MODERATE: Weak to moderate correlation")
    else:
        print("  ✗ WEAK: No significant correlation")
    
    return results


def visualize_importance_results(results: Dict) -> plt.Figure:
    """
    Visualize importance-compression relationship.
    
    Args:
        results: Dict from analyze_importance_compression
    
    Returns:
        fig: Matplotlib figure
    """
    return plot_importance_results(results)

