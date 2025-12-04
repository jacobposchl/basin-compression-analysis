"""Text preprocessing utilities."""

from typing import List, Tuple
import nltk
from transformers import GPT2Tokenizer

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("Downloading NLTK punkt tokenizer...")
    nltk.download('punkt', quiet=True)

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    print("Downloading NLTK POS tagger...")
    nltk.download('averaged_perceptron_tagger', quiet=True)

try:
    nltk.data.find('taggers/universal_tagset')
except LookupError:
    print("Downloading NLTK universal tagset...")
    nltk.download('universal_tagset', quiet=True)


def tokenize_texts(
    tokenizer: GPT2Tokenizer,
    texts: List[str],
    max_length: int = 128,
    truncation: bool = True
) -> Tuple[List[List[int]], List[List[str]]]:
    """
    Tokenize a list of texts.
    
    Args:
        tokenizer: GPT2Tokenizer instance
        texts: List of text strings
        max_length: Maximum sequence length
        truncation: Whether to truncate long sequences
    
    Returns:
        token_ids_list: List of token ID lists
        token_strings_list: List of token string lists
    """
    token_ids_list = []
    token_strings_list = []
    
    for text in texts:
        # Tokenize
        token_ids = tokenizer.encode(
            text,
            max_length=max_length,
            truncation=truncation
        )
        
        # Get token strings
        token_strings = [tokenizer.decode([tid]) for tid in token_ids]
        
        token_ids_list.append(token_ids)
        token_strings_list.append(token_strings)
    
    return token_ids_list, token_strings_list


def add_pos_tags(
    texts: List[str],
    tokens_list: List[List[str]]
) -> List[List[str]]:
    """
    Add part-of-speech tags to tokens.
    
    Note: This is an approximate mapping since GPT-2 uses subword tokenization
    while NLTK uses word-level tokenization. The alignment may not be perfect.
    
    Args:
        texts: List of original text strings
        tokens_list: List of token lists (from GPT-2 tokenizer)
    
    Returns:
        pos_tags_list: List of POS tag lists (one tag per token)
    """
    pos_tags_list = []
    
    for text, tokens in zip(texts, tokens_list):
        # Tokenize with NLTK (word-level)
        try:
            words = nltk.word_tokenize(text)
            
            # Get POS tags
            pos_tagged = nltk.pos_tag(words, tagset='universal')
            
            # Map to GPT-2 tokens (approximate - tokens may not align perfectly)
            pos_tags = []
            word_idx = 0
            
            for token in tokens:
                token_clean = token.strip()
                if word_idx < len(pos_tagged) and token_clean:
                    # Check if token matches current word
                    current_word, current_pos = pos_tagged[word_idx]
                    pos_tags.append(current_pos)
                    
                    # Advance to next word if token seems complete
                    # This is a heuristic - subword tokens make this tricky
                    if token_clean.isalpha() and not token_clean.startswith('##'):
                        # Check if we've consumed enough of the word
                        if len(token_clean) >= len(current_word) * 0.5:
                            word_idx += 1
                else:
                    pos_tags.append('X')  # Unknown/other
            
            # Ensure same length as tokens
            while len(pos_tags) < len(tokens):
                pos_tags.append('X')
            pos_tags = pos_tags[:len(tokens)]
            
        except Exception as e:
            print(f"Warning: Failed to tag text: {e}")
            pos_tags = ['X'] * len(tokens)
        
        pos_tags_list.append(pos_tags)
    
    return pos_tags_list

