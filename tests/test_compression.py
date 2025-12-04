"""Unit tests for compression metric computation."""

import unittest
import numpy as np
import torch

from compression_lm.compression.metric import (
    compute_compression_scores_layer,
    compute_all_layers_compression
)
from compression_lm.compression.neighborhoods import find_k_nearest_neighbors
from compression_lm.compression.utils import normalize_states, check_numerical_stability


class TestCompression(unittest.TestCase):
    """Test compression metric computation."""
    
    def test_normalize_states(self):
        """Test state normalization."""
        states = np.random.randn(10, 768)
        normalized = normalize_states(states)
        
        # Check that norms are approximately 1
        norms = np.linalg.norm(normalized, axis=1)
        np.testing.assert_allclose(norms, 1.0, rtol=1e-5)
    
    def test_find_k_nearest_neighbors(self):
        """Test k-NN computation."""
        states = np.random.randn(100, 768)
        k = 10
        
        distances, indices = find_k_nearest_neighbors(states, k=k)
        
        # Check output shapes
        self.assertEqual(distances.shape, (100, k))
        self.assertEqual(indices.shape, (100, k))
        
        # Check that indices are valid
        self.assertTrue(np.all(indices >= 0))
        self.assertTrue(np.all(indices < len(states)))
        
        # Check that distances are non-negative
        self.assertTrue(np.all(distances >= 0))
    
    def test_compute_compression_scores_layer(self):
        """Test compression score computation for one layer."""
        # Create dummy hidden states
        n_sequences = 5
        seq_len = 10
        hidden_dim = 768
        
        hidden_states_L = [torch.randn(seq_len, hidden_dim) for _ in range(n_sequences)]
        hidden_states_Lplus1 = [torch.randn(seq_len, hidden_dim) for _ in range(n_sequences)]
        
        scores, metadata = compute_compression_scores_layer(
            hidden_states_L,
            hidden_states_Lplus1,
            k=5,
            use_faiss=False  # Use sklearn for small test
        )
        
        # Check output
        self.assertIsInstance(scores, np.ndarray)
        self.assertEqual(len(scores), n_sequences * seq_len)
        
        # Check metadata
        self.assertIn('sequence_indices', metadata)
        self.assertIn('mean_score', metadata)
        self.assertIn('std_score', metadata)
        
        # Check numerical stability
        stability = check_numerical_stability(scores)
        self.assertFalse(stability['has_nan'])
        self.assertFalse(stability['has_inf'])
    
    def test_compute_all_layers_compression(self):
        """Test compression computation across all layers."""
        n_sequences = 3
        seq_len = 8
        hidden_dim = 768
        num_layers = 5
        
        # Create dummy states for multiple layers
        all_states = {
            i: [torch.randn(seq_len, hidden_dim) for _ in range(n_sequences)]
            for i in range(num_layers)
        }
        
        layer_compression, layer_metadata = compute_all_layers_compression(
            all_states, k=5, use_faiss=False
        )
        
        # Check that we computed compression for all layer transitions
        self.assertEqual(len(layer_compression), num_layers - 1)
        self.assertEqual(len(layer_metadata), num_layers - 1)
        
        # Check that each layer has scores
        for layer_idx in range(num_layers - 1):
            self.assertIn(layer_idx, layer_compression)
            self.assertIn(layer_idx, layer_metadata)


if __name__ == '__main__':
    unittest.main()

