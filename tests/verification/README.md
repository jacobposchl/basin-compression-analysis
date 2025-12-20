# Verification Tests

This folder contains scripts to verify that your training dynamics experiments are working correctly.

## Phase 1: Verify Memorization

**Purpose**: Confirm the model actually memorized training data (CRITICAL STEP!)

**Script**: `verify_memorization.py`

### Quick Test (Recommended First)

```bash
python tests/verification/verify_memorization.py \
    --results_dir dynamics_results_standard \
    --num_samples 20
```

**Time**: ~5-10 minutes  
**What it does**: Tests 20 passages from train/test sets, computes perplexity

### Full Test (More Reliable)

```bash
python tests/verification/verify_memorization.py \
    --results_dir dynamics_results_standard \
    --num_samples 100
```

**Time**: ~20-30 minutes  
**What it does**: Tests all 100 passages for more accurate results

### In Colab Notebook

The verification is already integrated into `training_dynamics_experiment.ipynb` under the **"PHASE 1: Verify Memorization"** section. Just run those cells after your experiment completes.

## Expected Output

```
======================================================================
MEMORIZATION VERIFICATION
======================================================================
Results directory: dynamics_results_standard
Found 8 checkpoints
Train passages: 20
Test passages: 20
======================================================================

Testing checkpoint: 1 epochs
  Train PPL: 48.23 ± 12.45
  Test PPL:  51.67 ± 13.21
  Ratio:     1.07x
  ✗ NO MEMORIZATION

Testing checkpoint: 10 epochs
  Train PPL: 15.34 ± 8.92
  Test PPL:  42.18 ± 14.67
  Ratio:     2.75x
  ✓ MEMORIZATION DETECTED

Testing checkpoint: 100 epochs
  Train PPL: 3.21 ± 1.45
  Test PPL:  48.93 ± 16.23
  Ratio:     15.24x
  ✓✓ STRONG MEMORIZATION
```

## Interpretation

### ✓ Memorization Confirmed (ratio ≥ 2.0x)
- Your experiment is valid
- Proceed with compression analysis
- Results are meaningful

### ✗ No Memorization (ratio < 2.0x)
- **STOP**: Don't analyze compression yet
- Model hasn't memorized the data
- Your null results are explained

**Fix by:**
1. Training for more epochs (150-200)
2. Using fewer passages (50 instead of 100)
3. Increasing learning rate slightly
4. Using longer sequences (max_length=256)

## Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--results_dir` | Directory with checkpoint folders | (required) |
| `--num_samples` | Number of passages to test | 20 |
| `--device` | cuda or cpu | auto-detect |
| `--output` | Save results to CSV | None |

## Output Files

- **Console output**: Summary tables and verdict
- **CSV file** (if `--output` specified): Detailed results for all checkpoints
- **In notebook**: Generates plots showing train/test perplexity curves

## Next Steps

After confirming memorization:
- **Phase 2**: Verify compression calculation (coming soon)
- **Phase 3**: Verify statistical tests (coming soon)
- **Phase 4**: Cross-validation (coming soon)

## Troubleshooting

**Error: "No checkpoints found"**
- Make sure you ran the training experiment first
- Check that `results_dir` points to correct location
- Checkpoints should be in folders like `checkpoint_1_epochs/`, `checkpoint_10_epochs/`, etc.

**Error: "CUDA out of memory"**
- Reduce `--num_samples` (try 10)
- Use `--device cpu` (slower but works)

**Test perplexity is NaN or infinity**
- Some passages may be too short
- This is normal for a few passages
- Check that most passages have valid scores
