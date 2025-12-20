"""
Quick test to verify the verification script works.

This creates dummy checkpoints and tests the verification logic.
Run this to make sure everything is set up correctly before using real data.
"""

import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from transformers import GPT2LMHeadModel, GPT2Tokenizer


def create_dummy_checkpoint(checkpoint_dir: Path, model_name: str = 'gpt2'):
    """Create a dummy checkpoint directory with a GPT-2 model."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Load and save a base model
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    
    model.save_pretrained(checkpoint_dir)
    tokenizer.save_pretrained(checkpoint_dir)
    
    print(f"Created dummy checkpoint: {checkpoint_dir}")


def test_verification():
    """Test the verification script with dummy data."""
    print("="*70)
    print("TESTING VERIFICATION SYSTEM")
    print("="*70)
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        results_dir = tmpdir / "test_results"
        results_dir.mkdir()
        
        print(f"\nCreating dummy checkpoints in: {results_dir}")
        
        # Create dummy checkpoints
        for epochs in [1, 10, 100]:
            checkpoint_dir = results_dir / f"checkpoint_{epochs}_epochs"
            create_dummy_checkpoint(checkpoint_dir)
        
        print("\nRunning verification script...")
        print("(This will use the same base model for all checkpoints,")
        print(" so ratios will be ~1.0, but tests the infrastructure)")
        
        # Import and run verification
        from tests.verification.verify_memorization import quick_verification
        
        try:
            df = quick_verification(
                str(results_dir),
                num_samples=5,  # Very small for quick test
                device='cpu'     # CPU for compatibility
            )
            
            print("\n" + "="*70)
            print("✓ TEST PASSED")
            print("="*70)
            print("The verification system is working correctly!")
            print("\nNote: All ratios are ~1.0 because we used the same model.")
            print("With real training, ratios should increase with epochs.")
            
            return True
            
        except Exception as e:
            print("\n" + "="*70)
            print("✗ TEST FAILED")
            print("="*70)
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == '__main__':
    success = test_verification()
    sys.exit(0 if success else 1)
