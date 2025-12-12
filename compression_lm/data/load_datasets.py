"""Dataset loading utilities."""

from typing import List, Optional
from datasets import load_dataset


def load_wikitext(
    split: str = 'test',
    max_samples: Optional[int] = None,
    min_length: int = 100,
    use_small: bool = False
) -> List[str]:
    """
    Load WikiText-103 dataset.
    
    Args:
        split: Dataset split ('train', 'validation', 'test')
        max_samples: Maximum number of samples to load (None for all)
        min_length: Minimum text length in characters
        use_small: If True, use wikitext-2 (smaller, faster to download) instead of wikitext-103
    
    Returns:
        texts: List of text strings
    """
    if use_small:
        print(f"Loading WikiText-2 ({split} split) - smaller dataset for faster testing...")
        dataset = load_dataset('wikitext', 'wikitext-2-v1', split=split)
    else:
        print(f"Loading WikiText-103 ({split} split)...")
        print("Note: This may take a few minutes to download (~300MB). Use use_small=True for faster testing.")
        dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)
    
    # Filter by length
    texts = [text for text in dataset['text'] if len(text.strip()) > min_length]
    
    # Limit samples if requested
    if max_samples is not None:
        texts = texts[:max_samples]
    
    print(f"Loaded {len(texts)} text samples")
    if len(texts) > 0:
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


def load_fine_tuning_passages(
    num_passages: int = 100,
    min_length: int = 100,
    max_length: Optional[int] = None,
    use_small: bool = False,
    split: str = 'train'
) -> List[str]:
    """
    Load passages from WikiText-103 for fine-tuning.
    
    Samples diverse passages with different topics and lengths to ensure
    good coverage for memorization experiments.
    
    Args:
        num_passages: Number of passages to load
        min_length: Minimum text length in characters
        max_length: Maximum text length in characters (None = no limit)
        use_small: If True, use WikiText-2 instead of WikiText-103
        split: Dataset split to use ('train', 'validation', 'test')
    
    Returns:
        texts: List of passage strings
    """
    print(f"Loading {num_passages} passages for fine-tuning...")
    
    # Load dataset
    if use_small:
        print(f"Loading WikiText-2 ({split} split)...")
        dataset = load_dataset('wikitext', 'wikitext-2-v1', split=split)
    else:
        print(f"Loading WikiText-103 ({split} split)...")
        dataset = load_dataset('wikitext', 'wikitext-103-v1', split=split)
    
    # Filter by length
    texts = []
    for text in dataset['text']:
        text = text.strip()
        if len(text) < min_length:
            continue
        if max_length is not None and len(text) > max_length:
            continue
        # Skip empty or very short texts
        if len(text) < min_length:
            continue
        texts.append(text)
    
    # Sample diverse passages (take every Nth passage to get diversity)
    if len(texts) > num_passages:
        # Use strided sampling to get diverse passages
        step = len(texts) // num_passages
        texts = texts[::step][:num_passages]
    else:
        texts = texts[:num_passages]
    
    print(f"Loaded {len(texts)} passages for fine-tuning")
    if len(texts) > 0:
        avg_length = sum(len(t) for t in texts) / len(texts)
        print(f"Average passage length: {avg_length:.1f} characters")
        print(f"Length range: {min(len(t) for t in texts)} - {max(len(t) for t in texts)} characters")
    
    return texts
