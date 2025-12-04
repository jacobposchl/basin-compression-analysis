"""Unit tests for hidden state extraction."""

import unittest
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

from compression_lm.models.extract_states import extract_hidden_states, extract_dataset_states


class TestExtractStates(unittest.TestCase):
    """Test hidden state extraction functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model = GPT2LMHeadModel.from_pretrained('gpt2')
        cls.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        cls.model.eval()
        
        # Set pad token if not present
        if cls.tokenizer.pad_token is None:
            cls.tokenizer.pad_token = cls.tokenizer.eos_token
    
    def test_extract_hidden_states_single(self):
        """Test extracting hidden states for a single text."""
        text = "The quick brown fox jumps over the lazy dog."
        
        hidden_states, tokens, token_ids = extract_hidden_states(
            self.model, self.tokenizer, text
        )
        
        # Check output types
        self.assertIsInstance(hidden_states, list)
        self.assertIsInstance(tokens, list)
        self.assertIsInstance(token_ids, list)
        
        # Check that we have states for all layers (GPT-2 small has 12 layers)
        self.assertEqual(len(hidden_states), 12)
        
        # Check tensor shapes
        for states in hidden_states:
            self.assertIsInstance(states, torch.Tensor)
            self.assertEqual(len(states.shape), 2)  # [seq_len, hidden_dim]
            self.assertEqual(states.shape[1], 768)  # GPT-2 small hidden dim
        
        # Check that tokens and states have same length
        self.assertEqual(len(tokens), len(token_ids))
        self.assertEqual(len(tokens), hidden_states[0].shape[0])
    
    def test_extract_dataset_states(self):
        """Test extracting hidden states for multiple texts."""
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A journey of a thousand miles begins with a single step.",
            "To be or not to be, that is the question."
        ]
        
        all_states, all_tokens, metadata = extract_dataset_states(
            self.model, self.tokenizer, texts, max_length=50
        )
        
        # Check output types
        self.assertIsInstance(all_states, dict)
        self.assertIsInstance(all_tokens, list)
        self.assertIsInstance(metadata, list)
        
        # Check that we have states for all layers
        self.assertEqual(len(all_states), 12)
        
        # Check that each layer has states for all sequences
        for layer_idx in range(12):
            self.assertEqual(len(all_states[layer_idx]), len(texts))
        
        # Check metadata
        self.assertEqual(len(metadata), len(texts))
        for meta in metadata:
            self.assertIn('text_idx', meta)
            self.assertIn('original_text', meta)
            self.assertIn('num_tokens', meta)


if __name__ == '__main__':
    unittest.main()

