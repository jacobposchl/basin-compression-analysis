"""Model loading utilities for GPT-2 and other transformer models."""

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(model_name='gpt2', device=None):
    """
    Load GPT-2 model and tokenizer.
    
    Args:
        model_name: Model identifier. Options:
            - 'gpt2' (GPT-2 Small, 117M parameters)
            - 'gpt2-medium' (GPT-2 Medium, 345M parameters)
            - 'gpt2-large' (GPT-2 Large, 762M parameters)
            - 'gpt2-xl' (GPT-2 XL, 1.5B parameters)
        device: Device to load model on ('cuda', 'cpu', or None for auto-detect)
    
    Returns:
        model: GPT2LMHeadModel instance
        tokenizer: GPT2Tokenizer instance
        device: torch.device used
    """
    if device is None:
        # Auto-detect: prefer CUDA if available
        if torch.cuda.is_available():
            device = torch.device('cuda')
            print("CUDA is available. Using GPU.")
        else:
            device = torch.device('cpu')
            print("CUDA is not available. Using CPU.")
    else:
        device = torch.device(device)
        if device.type == 'cuda' and not torch.cuda.is_available():
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
    
    print(f"Loading model: {model_name}")
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    # Set pad token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.eval()  # Set to evaluation mode
    model = model.to(device)
    
    # Verify model is on the correct device
    actual_device = next(model.parameters()).device
    if actual_device != device:
        print(f"Warning: Model is on {actual_device} but requested {device}")
    else:
        print(f"Model successfully loaded on {device}")
    
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    print("Model loaded successfully!")
    
    return model, tokenizer, device


def get_model_config(model_name='gpt2'):
    """
    Get model configuration information.
    
    Args:
        model_name: Model identifier
    
    Returns:
        config: Dict with model configuration
    """
    model_configs = {
        'gpt2': {
            'num_layers': 12,
            'hidden_dim': 768,
            'num_heads': 12,
            'num_parameters': 117_000_000,
        },
        'gpt2-medium': {
            'num_layers': 24,
            'hidden_dim': 1024,
            'num_heads': 16,
            'num_parameters': 345_000_000,
        },
        'gpt2-large': {
            'num_layers': 36,
            'hidden_dim': 1280,
            'num_heads': 20,
            'num_parameters': 762_000_000,
        },
        'gpt2-xl': {
            'num_layers': 48,
            'hidden_dim': 1600,
            'num_heads': 25,
            'num_parameters': 1_500_000_000,
        },
    }
    
    if model_name not in model_configs:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(model_configs.keys())}")
    
    return model_configs[model_name]

