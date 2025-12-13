# Training Dynamics Experiment - Quick Reference Card

## One-Line Commands

### Google Colab (Recommended)
```python
# Upload notebooks/training_dynamics_experiment.ipynb to Colab
# Select A100 GPU → Run all cells
```

### Command Line - Quick Test
```bash
python scripts/run_training_dynamics.py --quick_test
# ~1 hour, 50 passages, epochs [1,5,20]
```

### Command Line - Standard  
```bash
python scripts/run_training_dynamics.py --num_passages 100 --epoch_schedule 1,3,5,10,20,30,50,100
# ~5-6 hours, full analysis
```

### Visualize Results
```bash
python scripts/visualize_training_dynamics.py --results_file dynamics_results/training_dynamics_results.pkl
```

## Key Files

| File | Purpose |
|------|---------|
| `scripts/run_training_dynamics.py` | Main experiment runner |
| `scripts/visualize_training_dynamics.py` | Generate figures |
| `scripts/example_training_dynamics.py` | Quick demo |
| `notebooks/training_dynamics_experiment.ipynb` | Colab interface |
| `TRAINING_DYNAMICS_GUIDE.md` | Complete documentation |
| `IMPLEMENTATION_SUMMARY.md` | Technical details |

## Command-Line Arguments

### Essential
- `--num_passages 100` - Training passages (default: 100)
- `--epoch_schedule 1,3,5,10,20` - Comma-separated epochs
- `--output_dir results/` - Where to save

### Quick Modes
- `--quick_test` - Fast mode: 50 passages, [1,5,20] epochs
- `--use_small_dataset` - Use WikiText-2 instead of 103

### Advanced
- `--save_checkpoints` - Save model weights
- `--resume_from_epoch 20` - Resume from crash
- `--learning_rate 5e-5` - Fine-tuning LR
- `--batch_size 4` - Training batch size

## Output Files

```
dynamics_results/
├── training_dynamics_results.pkl    ← Load this with pickle
├── training_summary.csv             ← Quick summary table
├── summary_report.txt               ← Text analysis
└── figures/
    ├── compression_trajectories.png
    ├── memorization_vs_compression.png
    ├── layer_epoch_heatmap.png
    ├── individual_trajectories.png
    └── memorization_rate.png
```

## Load and Analyze

```python
import pickle
import pandas as pd

# Load results
with open('dynamics_results/training_dynamics_results.pkl', 'rb') as f:
    data = pickle.load(f)

all_results = data['all_results']  # Dict: epoch -> results

# Quick summary
summary = pd.read_csv('dynamics_results/training_summary.csv')
print(summary)

# Analyze specific layer
from compression_lm.analysis.training_dynamics import analyze_u_shape_trajectory
u_shape = analyze_u_shape_trajectory(all_results, layer_idx=6)
print(f"Shape: {u_shape['shape']}, R²={u_shape['r_squared']:.3f}")
```

## Key Functions

### Analysis
```python
from compression_lm.analysis.training_dynamics import (
    analyze_u_shape_trajectory,      # Detect U-shape
    analyze_memorization_onset,      # When do passages memorize?
    analyze_layer_temporal_patterns, # Which layers change first?
    compute_compression_velocity,    # Rate of change
    generate_summary_report          # Text report
)
```

### Visualization
```python
from compression_lm.analysis.dynamics_visualizations import (
    plot_compression_trajectories,         # All layers over time
    plot_memorization_vs_compression,      # Scatter plots
    plot_layer_epoch_heatmap,             # Heatmap
    plot_individual_passage_trajectories, # Per-passage
    create_all_visualizations             # Generate all
)
```

## Experiment Modes

| Mode | Passages | Epochs | Time (A100) | Use Case |
|------|----------|--------|-------------|----------|
| Quick | 50 | [1,5,20] | ~1 hour | Testing |
| Standard | 100 | [1,3,5,10,20,30,50,100] | ~5-6 hours | Research |
| Deep | 200 | [1,3,5,7,...,200] | ~12-15 hours | Publication |

## Interpretation

### U-Shaped Curve Found ✓
- **Pattern:** Compression decreases → plateaus → increases
- **Meaning:** Memorization = geometric phase transition
- **Impact:** HIGH - Novel finding (ICML/NeurIPS)

### Monotonic Trend
- **Pattern:** Compression steadily decreases/increases
- **Meaning:** LLM memorization ≠ VAE compression
- **Impact:** MEDIUM - Interesting difference (ACL/EMNLP)

### No Memorization
- **Pattern:** Accuracy stays low even at 100 epochs
- **Next Step:** Reduce to 10 passages, train 200 epochs
- **Or:** Increase learning rate to 1e-4

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | `--num_passages 50` or `--max_length 64` |
| Too slow | `--quick_test` or `--use_small_dataset` |
| Crashed | `--resume_from_epoch 20` |
| No memorization | More epochs, higher LR, fewer passages |

## Research Questions

1. **U-shape?** → `analyze_u_shape_trajectory()`
2. **When memorize?** → `analyze_memorization_onset()`
3. **Which layers first?** → `analyze_layer_temporal_patterns()`
4. **Prediction?** → Check correlation at epoch 1

## GPU Requirements

- **Minimum:** T4 (12-15 hours for standard)
- **Recommended:** V100 (8-10 hours) or A100 (5-6 hours)
- **Memory:** ~16 GB VRAM for 100 passages

## Example Workflow

```bash
# 1. Run experiment
python scripts/run_training_dynamics.py --num_passages 100

# 2. Check progress (while running)
cat dynamics_results/training_summary.csv

# 3. Visualize (after complete)
python scripts/visualize_training_dynamics.py \
    --results_file dynamics_results/training_dynamics_results.pkl

# 4. View figures
ls dynamics_results/figures/

# 5. Read report
cat dynamics_results/summary_report.txt
```

## Citation

```bibtex
@software{basin_compression_dynamics,
  title={Training Dynamics of Compression in Language Model Memorization},
  author={Poschl, Jacob},
  year={2025},
  url={https://github.com/jacobposchl/basin-compression-analysis}
}
```

---

**Full Documentation:** [TRAINING_DYNAMICS_GUIDE.md](TRAINING_DYNAMICS_GUIDE.md)  
**Implementation Details:** [IMPLEMENTATION_SUMMARY.md](IMPLEMENTATION_SUMMARY.md)  
**Colab Notebook:** `notebooks/training_dynamics_experiment.ipynb`
