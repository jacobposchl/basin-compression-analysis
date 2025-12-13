# Training Dynamics Implementation - Complete Summary

## What Was Implemented

A comprehensive multi-epoch training dynamics experiment system that tracks how compression evolves as a language model transitions from initial exposure through deep memorization.

## Files Created

### 1. Main Scripts (4 files)

#### `scripts/run_training_dynamics.py` (Main Experiment)
- **Purpose**: Core experiment runner that trains models at multiple epoch counts
- **Key Functions**:
  - `compute_detailed_reproduction_metrics()` - Enhanced memorization detection with per-passage accuracies, edit distances, prefix accuracies
  - `train_and_analyze_checkpoint()` - Train, analyze, and save results for one epoch count
  - `run_training_dynamics_experiment()` - Orchestrate full multi-epoch experiment
- **Features**:
  - Automatic checkpoint saving and resumption
  - Memory-efficient (clears models between epochs)
  - Progress tracking with intermediate saves
  - CSV summary for quick inspection
  - Configurable epoch schedules

#### `scripts/visualize_training_dynamics.py` (Visualization)
- **Purpose**: Generate all figures from saved results
- **Output**: 5 comprehensive figures + text report

#### `scripts/example_training_dynamics.py` (Quick Test)
- **Purpose**: Minimal example (20 passages, 3 epochs, ~20 minutes)
- **Use Case**: Testing, development, demonstration

#### `notebooks/training_dynamics_experiment.ipynb` (Colab Interface)
- **Purpose**: Interactive Google Colab notebook
- **Features**:
  - GPU detection and setup
  - Three experiment modes (quick/standard/deep)
  - Real-time progress monitoring
  - Inline visualizations
  - Results download
  - Interpretation guide

### 2. Analysis Modules (2 files)

#### `compression_lm/analysis/training_dynamics.py`
Statistical analysis functions:

- **`analyze_u_shape_trajectory()`**
  - Fits quadratic curve to compression over epochs
  - Detects U-shaped, inverted-U, or linear patterns
  - Returns vertex (inflection point), R², trend direction

- **`analyze_memorization_onset()`**
  - Identifies epoch when each passage first memorizes
  - Correlates initial compression with memorization timing
  - Tests: Do low-compression passages memorize earlier?

- **`analyze_layer_temporal_patterns()`**
  - Determines which layers show effects first
  - Tests: Do early or late layers memorize first?
  - Tracks significance emergence per layer

- **`compute_compression_velocity()`**
  - Calculates first derivative (velocity) and second derivative (acceleration)
  - Identifies critical transition points

- **`generate_summary_report()`**
  - Comprehensive text report of all findings

#### `compression_lm/analysis/dynamics_visualizations.py`
Visualization functions:

- **`plot_compression_trajectories()`** - Figure 1: 3×4 grid showing all 12 layers
- **`plot_memorization_vs_compression()`** - Figure 2: Scatter at key epochs
- **`plot_layer_epoch_heatmap()`** - Figure 3: Heatmap of compression differences
- **`plot_individual_passage_trajectories()`** - Figure 4: Dual-axis trajectories
- **`plot_memorization_rate_over_time()`** - Additional: Memorization progress
- **`plot_compression_velocity()`** - Additional: Velocity and acceleration
- **`create_all_visualizations()`** - Generate all figures at once

### 3. Documentation (2 files)

#### `TRAINING_DYNAMICS_GUIDE.md`
Complete user guide covering:
- Quick start (Colab and command-line)
- Experiment modes (quick/standard/deep)
- All command-line arguments
- Output structure
- Result loading and analysis
- Interpretation guidelines
- Troubleshooting
- Tips for Google Colab

#### This summary document

## How to Use

### For Google Colab (Recommended)

1. **Upload the notebook**:
   - Upload `notebooks/training_dynamics_experiment.ipynb` to Colab
   - Select A100 GPU runtime

2. **Choose experiment mode**:
   ```python
   # In the notebook configuration cell
   EXPERIMENT_MODE = "standard"  # or "quick" or "deep"
   ```

3. **Run all cells**:
   - Experiment runs automatically
   - Progress shown in real-time
   - Results visualized inline

4. **Download results**:
   - Auto-zipped at the end
   - Or mount Google Drive

### For Command Line

```bash
# Standard experiment (5-6 hours on A100)
python scripts/run_training_dynamics.py \
    --num_passages 100 \
    --epoch_schedule 1,3,5,10,20,30,50,100 \
    --output_dir dynamics_results

# Generate visualizations
python scripts/visualize_training_dynamics.py \
    --results_file dynamics_results/training_dynamics_results.pkl

# Quick test (20 minutes)
python scripts/example_training_dynamics.py
```

### For Programmatic Use

```python
from scripts.run_training_dynamics import run_training_dynamics_experiment

results = run_training_dynamics_experiment(
    model_name='gpt2',
    num_passages=100,
    epoch_schedule=[1, 3, 5, 10, 20, 30, 50, 100],
    output_dir='my_results'
)
```

## Output Structure

```
dynamics_results/
├── training_dynamics_results.pkl    # Full data (pickle)
├── training_summary.csv             # Quick summary
├── summary_report.txt               # Text analysis
├── figures/
│   ├── compression_trajectories.png      # 12 layers over time
│   ├── memorization_vs_compression.png   # Scatter plots
│   ├── layer_epoch_heatmap.png          # Heatmap
│   ├── individual_trajectories.png       # 10 passages
│   └── memorization_rate.png            # Progress over time
└── checkpoints/ (optional, if --save_checkpoints)
    ├── model_epoch_1/
    ├── model_epoch_3/
    └── ...
```

## Data Structures

### Results Dictionary
```python
{
    'all_results': {
        1: {  # epoch count
            'epoch': 1,
            'timestamp': '2025-12-12T...',
            'reproduction_metrics': [  # per passage
                {
                    'passage_idx': 0,
                    'overall_accuracy': 0.23,
                    'exact_matches': 45,
                    'total_tokens': 195,
                    'is_memorized': False,
                    'prefix_accuracy': {'first_5': 0.8, 'first_10': 0.6, ...}
                },
                ...
            ],
            'layer_analyses': {  # per layer
                0: {
                    'layer': 0,
                    'correlation': -0.45,
                    'correlation_p': 1.2e-10,
                    'memorized_mean': -5.47,
                    'novel_mean': -5.28,
                    't_statistic': -7.13,
                    't_test_p': 1.8e-11,
                    'sequence_compression': np.array([...]),  # per sequence
                    'sequence_memorization': np.array([...])  # boolean labels
                },
                ...
            },
            'best_layer': 4,
            'best_correlation': -0.513,
            'num_memorized': 0,
            'mean_accuracy': 0.234,
            'median_accuracy': 0.198
        },
        3: { ... },  # next epoch
        ...
    },
    'training_passages': [...],
    'novel_passages': [...],
    'epoch_schedule': [1, 3, 5, 10, ...],
    'parameters': { ... }
}
```

## Research Questions Addressed

### 1. U-Shaped Trajectory?
**Analysis**: `analyze_u_shape_trajectory()`
- Fits quadratic: compression = a·epoch² + b·epoch + c
- If a > 0: U-shaped (decrease → increase)
- If a < 0: Inverted U (increase → decrease)
- Returns R², vertex location, trend

**Visualization**: Compression trajectories plot (Figure 1)

### 2. Memorization Onset
**Analysis**: `analyze_memorization_onset()`
- Tracks when each passage first reaches 60% accuracy
- Correlates onset with initial compression
- Tests: Do distinctive patterns memorize earlier/later?

**Visualization**: Individual trajectories (Figure 4), scatter plots (Figure 2)

### 3. Layer Patterns
**Analysis**: `analyze_layer_temporal_patterns()`
- Identifies which layers show significant differences first
- Tests: early_first, late_first, or simultaneous

**Visualization**: Layer-epoch heatmap (Figure 3)

### 4. Prediction
**Analysis**: Correlation between initial compression and final memorization
- Can compression at epoch 1 predict which passages will memorize?

**Visualization**: Memorization vs compression scatter (Figure 2)

## Key Innovations

### 1. Enhanced Reproduction Metrics
Beyond binary memorized/not:
- Overall accuracy (% exact tokens)
- Prefix accuracy (first 5, 10, 20, 50 tokens)
- Edit distance
- Enables gradual memorization tracking

### 2. Multi-Layer Analysis
- All 12 layers analyzed independently
- Identifies "best layer" per checkpoint
- Tracks layer-wise evolution

### 3. Velocity Analysis
- First derivative: Rate of compression change
- Second derivative: Acceleration
- Identifies phase transitions

### 4. Automatic Checkpointing
- Saves intermediate results
- Resume from crashes with `--resume_from_epoch`
- Memory-efficient (clears models between epochs)

### 5. Comprehensive Visualization
- 5 publication-ready figures
- Interactive Colab interface
- Customizable for papers

## Computational Requirements

| Mode | Passages | Epochs | Time (A100) | Storage |
|------|----------|--------|-------------|---------|
| Quick | 50 | [1,5,20] | ~1 hour | ~500 MB |
| Standard | 100 | [1,3,5,10,20,30,50,100] | ~5-6 hours | ~2 GB |
| Deep | 200 | [1,3,5,7,10,15,...,200] | ~12-15 hours | ~5 GB |

Storage without model checkpoints. Add ~500 MB per epoch if `--save_checkpoints`.

## Scientific Impact

### Potential Findings

1. **U-shaped curve discovered**
   - Novel mechanistic insight
   - Shows memorization as geometric phase transition
   - Publication: ICML/NeurIPS (high impact)

2. **Monotonic trends only**
   - LLM memorization differs from VAE compression
   - Maintains distinctions even when memorized
   - Publication: ACL/EMNLP (solid contribution)

3. **Layer specialization**
   - Early layers memorize syntax, late layers semantics
   - Hierarchical organization revealed
   - Publication: ACL/EMNLP (interesting finding)

4. **No memorization achieved**
   - Task difficulty or capacity limits
   - Suggests compression detects "training exposure" not "memorization"
   - Still publishable as negative result

## Next Steps

### If U-shape found:
1. Test with different models (GPT-2 medium/large)
2. Vary content types (code, poetry, equations)
3. Theoretical analysis of phase transition
4. Connect to loss landscape geometry

### If monotonic only:
1. Compare to VAE behavior (does VAE show U-shape?)
2. Analyze why LLMs differ
3. Connect to generation vs. reconstruction tasks
4. Test semantic vs. syntactic memorization

### If no memorization:
1. Reduce to 10 passages, train 200+ epochs
2. Use higher learning rate (1e-4, 2e-4)
3. Try smaller model (distilgpt2) or larger (gpt2-medium)
4. Analyze what prevents memorization

## Integration with Existing Code

This implementation **reuses** existing modules:
- `load_model()` from `models/model_loader.py`
- `fine_tune_model()` from `models/fine_tune.py`
- `extract_hidden_states_batch()` from `models/extract_states.py`
- `compute_compression_all_layers()` from `compression/metric.py`
- `analyze_memorization_layer()` from `experiments/memorization.py`

**Only extends** with:
- `compute_detailed_reproduction_metrics()` (in main script)
- New analysis functions (training_dynamics.py)
- New visualization functions (dynamics_visualizations.py)

## Testing

```bash
# Quick test (~20 minutes)
python scripts/example_training_dynamics.py

# Verify outputs
ls example_dynamics/
# Should see: training_dynamics_results.pkl, training_summary.csv

# Load and check
python -c "
import pickle
with open('example_dynamics/training_dynamics_results.pkl', 'rb') as f:
    data = pickle.load(f)
print('Epochs:', sorted(data['all_results'].keys()))
print('Num layers:', len(data['all_results'][1]['layer_analyses']))
"
```

## Troubleshooting

See `TRAINING_DYNAMICS_GUIDE.md` for:
- Out of memory solutions
- Performance optimization
- Resume after crashes
- GPU selection tips
- Colab-specific issues

## Citation

```bibtex
@software{basin_compression_training_dynamics,
  title={Training Dynamics of Compression in Language Model Memorization},
  author={Poschl, Jacob},
  year={2025},
  url={https://github.com/jacobposchl/basin-compression-analysis}
}
```
