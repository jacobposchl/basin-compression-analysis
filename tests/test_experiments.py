"""Unit tests for experiment modules."""

import unittest
import numpy as np

from compression_lm.experiments.memorization import analyze_memorization_compression
from compression_lm.experiments.token_importance import analyze_importance_compression
from compression_lm.experiments.linguistic import analyze_pos_compression


class TestExperiments(unittest.TestCase):
    """Test experiment analysis functions."""
    
    def test_analyze_memorization_compression(self):
        """Test memorization-compression analysis."""
        n_tokens = 100
        n_sequences = 10
        
        # Create dummy data
        compression_scores = np.random.randn(n_tokens)
        memorization_labels = [True, False] * (n_sequences // 2)
        sequence_indices = np.repeat(range(n_sequences), n_tokens // n_sequences)
        
        results = analyze_memorization_compression(
            compression_scores,
            memorization_labels,
            sequence_indices,
            layer_idx=0
        )
        
        # Check output structure
        self.assertIn('layer', results)
        self.assertIn('correlation', results)
        self.assertIn('correlation_p', results)
        self.assertIn('t_statistic', results)
        self.assertIn('memorized_mean', results)
        self.assertIn('novel_mean', results)
        self.assertEqual(results['layer'], 0)
    
    def test_analyze_importance_compression(self):
        """Test importance-compression analysis."""
        n_tokens = 100
        
        # Create dummy data
        compression_scores = np.random.randn(n_tokens)
        importance_scores = np.random.randn(n_tokens)
        sequence_indices = np.zeros(n_tokens)  # All from same sequence
        
        results = analyze_importance_compression(
            compression_scores,
            importance_scores,
            sequence_indices,
            layer_idx=0
        )
        
        # Check output structure
        self.assertIn('layer', results)
        self.assertIn('pearson_r', results)
        self.assertIn('pearson_p', results)
        self.assertIn('spearman_r', results)
        self.assertIn('q1_importance', results)
        self.assertIn('q4_importance', results)
        self.assertEqual(results['layer'], 0)
    
    def test_analyze_pos_compression(self):
        """Test POS-compression analysis."""
        n_tokens = 100
        n_sequences = 5
        
        # Create dummy data
        compression_scores = np.random.randn(n_tokens)
        pos_tags = [['NOUN', 'VERB', 'ADJ'] * (n_tokens // 15 + 1) for _ in range(n_sequences)]
        sequence_indices = np.repeat(range(n_sequences), n_tokens // n_sequences)
        
        results = analyze_pos_compression(
            compression_scores,
            pos_tags,
            sequence_indices,
            layer_idx=0
        )
        
        # Check output structure
        self.assertIn('layer', results)
        self.assertIn('pos_stats', results)
        self.assertIn('sorted_pos', results)
        self.assertIn('compression_scores', results)
        self.assertIn('pos_tags', results)
        self.assertEqual(results['layer'], 0)


if __name__ == '__main__':
    unittest.main()

