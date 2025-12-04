"""Dataset loading utilities."""

from typing import List, Optional
from datasets import load_dataset


def load_wikitext(
    split: str = 'test',
    max_samples: Optional[int] = None,
    min_length: int = 100
) -> List[str]:
    """
    Load WikiText-103 dataset.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        max_samples: Maximum number of samples to load (None for all)
        min_length: Minimum text length in characters
    
    Returns:
        texts: List of text strings
    """
    print(f"Loading WikiText-103 ({split} split)...")
    dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)
    
    # Filter by length
    texts = [text for text in dataset['text'] if len(text.strip()) > min_length]
    
    # Limit samples if requested
    if max_samples is not None:
        texts = texts[:max_samples]
    
    print(f"Loaded {len(texts)} text samples")
    print(f"Average length: {sum(len(t) for t in texts) / len(texts):.1f} characters")
    
    return texts


def load_penn_treebank(
    split: str = 'test',
    max_samples: Optional[int] = None
) -> List[str]:
    """
    Load Penn Treebank dataset.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        max_samples: Maximum number of samples to load (None for all)
    
    Returns:
        texts: List of text strings
    """
    print(f"Loading Penn Treebank ({split} split)...")
    
    try:
        dataset = load_dataset('ptb_text_only', split=split)
        texts = [text for text in dataset['sentence'] if len(text.strip()) > 0]
    except Exception as e:
        print(f"Warning: Could not load Penn Treebank: {e}")
        print("Returning empty list. You may need to install the dataset separately.")
        return []
    
    if max_samples is not None:
        texts = texts[:max_samples]
    
    print(f"Loaded {len(texts)} text samples")
    
    return texts


def load_custom_texts(file_path: str) -> List[str]:
    """
    Load texts from a custom file (one text per line).
    
    Args:
        file_path: Path to text file
    
    Returns:
        texts: List of text strings
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    print(f"Loaded {len(texts)} texts from {file_path}")
    return texts

