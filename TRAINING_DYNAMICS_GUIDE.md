# Training Dynamics Experiment - Quick Start Guide

## Overview

This experiment tracks how compression evolves across multiple training epochs to understand the relationship between memorization and geometric representation changes.

## Files Created

### Core Scripts
- **`scripts/run_training_dynamics.py`** - Main experiment runner
- **`scripts/visualize_training_dynamics.py`** - Generate all figures
- **`notebooks/training_dynamics_experiment.ipynb`** - Google Colab interface

### Analysis Modules
- **`compression_lm/analysis/training_dynamics.py`** - Statistical analysis functions
- **`compression_lm/analysis/dynamics_visualizations.py`** - Visualization functions

## Quick Start (Google Colab)

### Option 1: Use the Notebook (Recommended)
1. Upload `notebooks/training_dynamics_experiment.ipynb` to Google Colab
2. Select Runtime → Change runtime type → A100 GPU
3. Run all cells
4. Download results when complete

### Option 2: Command Line
```python
# Clone and setup
!git clone https://github.com/jacobposchl/basin-compression-analysis.git
%cd basin-compression-analysis
!pip install -e .

# Run experiment (standard - 5-6 hours on A100)
!python scripts/run_training_dynamics.py \
    --num_passages 100 \
    --epoch_schedule 1,3,5,10,20,30,50,100 \
    --output_dir dynamics_results

# Generate visualizations
!python scripts/visualize_training_dynamics.py \
    --results_file dynamics_results/training_dynamics_results.pkl
```

## Experiment Modes

### Quick Test (~1 hour)
```bash
python scripts/run_training_dynamics.py \
    --quick_test \
    --num_passages 50 \
    --output_dir dynamics_quick
```

### Standard (~5-6 hours) - RECOMMENDED
```bash
python scripts/run_training_dynamics.py \
    --num_passages 100 \
    --epoch_schedule 1,3,5,10,20,30,50,100 \
    --output_dir dynamics_standard
```

### Deep Analysis (~12-15 hours)
```bash
python scripts/run_training_dynamics.py \
    --num_passages 200 \
    --epoch_schedule 1,3,5,7,10,15,20,30,40,50,75,100,150,200 \
    --output_dir dynamics_deep
```

## Command-Line Arguments

### Required
None - all have sensible defaults

### Important Options
- `--model` - Model to use (default: `gpt2`)
- `--num_passages` - Training passages (default: 100)
- `--epoch_schedule` - Comma-separated epochs (default: "1,3,5,10,20,30,50,100")
- `--output_dir` - Where to save results (default: `dynamics_results`)

### Training Parameters
- `--learning_rate` - Fine-tuning LR (default: 5e-5)
- `--batch_size` - Batch size (default: 4)
- `--max_length` - Max sequence length (default: 128)

### Advanced
- `--save_checkpoints` - Save model weights (~500MB per checkpoint)
- `--resume_from_epoch` - Resume from specific epoch
- `--use_small_dataset` - Use WikiText-2 instead of WikiText-103
- `--quick_test` - Run only epochs [1, 5, 20]

## Output Structure

```
dynamics_results/
├── training_dynamics_results.pkl    # Full results (load with pickle)
├── training_summary.csv             # Quick summary table
├── summary_report.txt               # Text analysis report
├── checkpoints/                     # Model weights (if --save_checkpoints)
│   ├── model_epoch_1/
│   ├── model_epoch_3/
│   └── ...
└── figures/                         # All visualizations
    ├── compression_trajectories.png
    ├── memorization_vs_compression.png
    ├── layer_epoch_heatmap.png
    ├── individual_trajectories.png
    └── memorization_rate.png
```

## Loading and Analyzing Results

```python
import pickle
import pandas as pd
from compression_lm.analysis.training_dynamics import (
    analyze_u_shape_trajectory,
    analyze_memorization_onset,
    generate_summary_report
)

# Load results
with open('dynamics_results/training_dynamics_results.pkl', 'rb') as f:
    data = pickle.load(f)

all_results = data['all_results']  # Dict: epoch -> results
parameters = data['parameters']     # Experiment config
training_passages = data['training_passages']
novel_passages = data['novel_passages']

# Quick summary
summary = pd.read_csv('dynamics_results/training_summary.csv')
print(summary)

# Detailed analysis
for layer_idx in range(12):
    u_analysis = analyze_u_shape_trajectory(all_results, layer_idx)
    print(f"Layer {layer_idx}: {u_analysis['shape']}, R²={u_analysis['r_squared']:.3f}")

# Generate report
report = generate_summary_report(all_results)
print(report)
```

## Key Research Questions

1. **U-Shaped Trajectory?**
   - Does compression decrease → plateau → increase across training?
   - See: `analyze_u_shape_trajectory()` and compression trajectory plots

2. **Memorization Onset**
   - When does verbatim reproduction begin?
   - Can initial compression predict which passages memorize first?
   - See: `analyze_memorization_onset()` and memorization rate plots

3. **Layer Patterns**
   - Do early vs. late layers show different dynamics?
   - See: `analyze_layer_temporal_patterns()` and layer-epoch heatmap

4. **Individual Differences**
   - Why do some passages memorize before others?
   - See: individual trajectory plots

## Interpreting Results

### Success Criteria

**Strong Result:** U-shaped curve found
- Compression decreases early, increases late
- Memorization onset correlates with inflection point
- Different layers show phase transitions at different epochs
- **Publication potential:** High (ICML/NeurIPS)

**Moderate Result:** Monotonic trends
- Compression steadily decreases or increases
- Significant difference between trained/novel persists
- **Publication potential:** Medium (ACL/EMNLP)

**Unexpected Result:** No memorization
- Even 100 epochs doesn't achieve verbatim reproduction
- Suggests task difficulty or model capacity limits
- **Next step:** Reduce to 10 passages, train 200 epochs

### Key Metrics

1. **Memorization Rate**: % passages with ≥60% reproduction accuracy
2. **Compression Difference**: Mean(trained) - Mean(novel)
3. **Correlation**: Point-biserial r between compression and memorization
4. **U-Shape R²**: How well quadratic fits the trajectory
5. **Onset Correlation**: Does initial compression predict memorization timing?

## Troubleshooting

### Out of Memory
- Reduce `--num_passages` to 50
- Reduce `--max_length` to 64
- Remove longest epoch counts (skip 200, 150, 100)

### Too Slow
- Use `--quick_test` mode
- Reduce `--num_passages`
- Use `--use_small_dataset`

### No Memorization
- Increase epochs (add 150, 200 to schedule)
- Increase learning rate to 1e-4
- Reduce num_passages to 50 (easier to memorize less data)

### Resume After Crash
```bash
python scripts/run_training_dynamics.py \
    --resume_from_epoch 20 \  # Skip epochs <= 20
    --output_dir dynamics_results  # Same directory
```

## Tips for Google Colab

1. **Prevent Disconnection**
   - Keep browser tab active
   - Use Colab Pro for longer runtimes
   - Save intermediate results (happens automatically)

2. **Monitor Progress**
   - Check `training_summary.csv` periodically
   - Use the progress check cells in the notebook

3. **Download Results**
   - Results auto-zip at the end
   - Download via `files.download()`
   - Or mount Google Drive and copy there

4. **GPU Selection**
   - A100: Fastest (~5-6 hours for standard)
   - V100: Medium (~8-10 hours)
   - T4: Slower (~12-15 hours)

## Advanced Customization

### Custom Epoch Schedule
```bash
# Test specific hypothesis about 10-30 epoch range
python scripts/run_training_dynamics.py \
    --epoch_schedule 10,12,15,17,20,23,25,27,30
```

### Different Model
```bash
# Use GPT-2 Medium (more capacity)
python scripts/run_training_dynamics.py \
    --model gpt2-medium \
    --batch_size 2  # Reduce batch size for larger model
```

### Save Checkpoints
```bash
# Save model weights for later analysis
python scripts/run_training_dynamics.py \
    --save_checkpoints \
    --output_dir dynamics_with_models
# Warning: Requires ~500MB × num_epochs storage
```

## Citation

```bibtex
@article{basin-compression-2025,
  title={Training Dynamics of Compression in Language Model Memorization},
  author={Your Name},
  year={2025},
  url={https://github.com/jacobposchl/basin-compression-analysis}
}
```

## Support

Issues or questions:
1. Check this guide first
2. Review notebook cells for examples
3. Open GitHub issue with error logs
4. Include: GPU type, num_passages, epoch_schedule, error message
