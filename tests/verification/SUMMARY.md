# Verification System - Quick Reference

## What Was Created

### 1. Verification Script
**Location**: `tests/verification/verify_memorization.py`

**Purpose**: Tests if your model actually memorized training data by comparing perplexity

**Usage**:
```bash
# Quick test (20 passages, ~5 min)
python tests/verification/verify_memorization.py \
    --results_dir dynamics_results_standard \
    --num_samples 20

# Full test (100 passages, ~30 min)
python tests/verification/verify_memorization.py \
    --results_dir dynamics_results_standard \
    --num_samples 100
```

### 2. Notebook Integration
**Location**: `notebooks/training_dynamics_experiment.ipynb`

**New Section**: "⭐ PHASE 1: Verify Memorization" (cells added after "Quick Progress Check")

**What it does**:
1. Runs verification automatically
2. Generates perplexity plots
3. Gives clear verdict on whether memorization occurred
4. Provides recommendations if memorization failed

### 3. Documentation
- `tests/verification/README.md` - Full documentation
- `notebooks/COLAB_GUIDE.md` - Updated with verification info

## How to Use (Colab)

After running your training dynamics experiment:

1. **Scroll to "Phase 1: Verify Memorization" section**
2. **Run the verification cell** (uses 20 samples for speed)
3. **Run the visualization cell** to see results
4. **Check the verdict**:
   - ✓ Ratio ≥ 2.0x → Memorization confirmed, continue!
   - ✗ Ratio < 2.0x → No memorization, need to retrain

## Expected Output Example

```
Testing checkpoint: 100 epochs
  Train PPL: 3.21 ± 1.45
  Test PPL:  48.93 ± 16.23
  Ratio:     15.24x
  ✓✓ STRONG MEMORIZATION

DECISION POINT
======================================================================
✓ Memorization confirmed! Proceeding with analysis is valid.
  Final checkpoint has 15.2x perplexity ratio.
```

## Why This Matters

**Your original experiment showed no compression differences** between memorized and novel text. Before investigating compression metrics or statistics, you MUST verify:

1. **Is the model actually memorizing?** (This test)
   - If NO → explains null results, need better training
   - If YES → compression metric or analysis may need work

2. Only proceed to Phase 2+ if memorization is confirmed

## Quick Decision Tree

```
Run experiment
    ↓
Run Phase 1 verification
    ↓
Ratio ≥ 2.0x?
├─ YES → ✓ Continue to compression analysis
└─ NO  → ✗ Stop, retrain with:
         • More epochs (150-200)
         • Fewer passages (50)
         • Higher learning rate
         • Then re-verify
```

## Files Modified/Created

```
tests/verification/
├── __init__.py                    (new)
├── verify_memorization.py         (new - main script)
└── README.md                      (new - documentation)

notebooks/
├── training_dynamics_experiment.ipynb  (modified - added cells)
└── COLAB_GUIDE.md                      (modified - updated note)
```

## Next Steps

After confirming memorization:
- [ ] Phase 2: Verify compression calculation
- [ ] Phase 3: Verify statistical tests
- [ ] Phase 4: Cross-validation

These will be added as separate scripts in `tests/verification/` following the same pattern.
