# Running Compression Basins Analysis in Google Colab

This guide explains how to run the compression basins experiments in Google Colab.

> **⭐ NEW**: The training dynamics notebook now includes built-in verification tests to ensure your experiments are valid! See the "Phase 1: Verify Memorization" section after running your experiment.

## Quick Start

### 1. Open Colab and Enable GPU

1. Go to [Google Colab](https://colab.research.google.com/)
2. Create a new notebook
3. Enable GPU: **Runtime > Change runtime type > GPU (T4 or better)**

### 2. Clone and Setup

```python
# Install dependencies
!pip install -q torch transformers numpy scipy scikit-learn faiss-cpu nltk spacy matplotlib seaborn pandas tqdm datasets statsmodels

# Download NLTK data
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('universal_tagset', quiet=True)

# Clone your repository (replace with your GitHub URL)
!git clone https://github.com/YOUR_USERNAME/basin-compression-analysis.git
%cd basin-compression-analysis

# Install the package
!pip install -e .
```

### 3. Verify Setup

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# Test import
from compression_lm.models.model_loader import load_model
print("Setup successful!")
```

## Running Experiments

### Option 1: Using Command Line Scripts

```python
# Memorization experiment (use --use_small_dataset for faster testing)
!python scripts/run_memorization.py --max_sequences 100 --k_neighbors 15 --use_small_dataset

# Token importance experiment  
!python scripts/run_token_importance.py --max_sequences 50 --k_neighbors 15 --use_small_dataset

# Linguistic structure experiment
!python scripts/run_linguistic.py --max_sequences 100 --k_neighbors 15 --use_small_dataset

# Full pipeline (all experiments)
!python scripts/run_full_pipeline.py --max_sequences 100 --k_neighbors 15 --use_small_dataset
```

**Note**: The `--use_small_dataset` flag uses WikiText-2 instead of WikiText-103, which is much smaller (~4MB vs ~300MB) and downloads in seconds instead of minutes. Remove this flag for the full dataset.

### Option 2: Using Python API

```python
from compression_lm.models.model_loader import load_model
from compression_lm.data.load_datasets import load_wikitext
from compression_lm.experiments.memorization import run_memorization_experiment

# Load model (will automatically use GPU if available)
model, tokenizer, device = load_model('gpt2')

# Load data
texts = load_wikitext(split='test', max_samples=100)

# Run experiment
results = run_memorization_experiment(
    model=model,
    tokenizer=tokenizer,
    texts=texts,
    k_neighbors=15,
    max_sequences=100,
    max_length=128
)
```

## Viewing Results

```python
# Display generated plots
from IPython.display import Image, display
import glob

for img_file in glob.glob("*.png"):
    print(f"\n{img_file}:")
    display(Image(img_file))

# Load saved results
import pickle
with open('memorization_results.pkl', 'rb') as f:
    results = pickle.load(f)
```

## Colab-Specific Tips

### Memory Management

- Colab free tier has limited RAM (~12GB)
- Start with small experiments (`--max_sequences 50-100`)
- Use `max_length=128` to limit sequence length
- Monitor memory usage: `!nvidia-smi` (if GPU enabled)

### Speed Optimization

- **Always enable GPU** for faster processing
- Use smaller `k_neighbors` (10-15) for faster k-NN computation
- Process in batches if running out of memory
- Consider using `faiss-gpu` instead of `faiss-cpu` if available

### Saving Results

```python
# Save to Google Drive (mount first)
from google.colab import drive
drive.mount('/content/drive')

# Copy results to Drive
!cp *.pkl *.png /content/drive/MyDrive/compression_basins_results/
```

### Recommended Settings for Colab

```python
# Small experiment (quick test)
max_sequences = 50
k_neighbors = 10
max_length = 64

# Medium experiment (balanced)
max_sequences = 100
k_neighbors = 15
max_length = 128

# Large experiment (may need more RAM)
max_sequences = 500
k_neighbors = 15
max_length = 128
```

## Troubleshooting

### GPU Not Available
- Go to **Runtime > Change runtime type > GPU**
- Wait for GPU to be allocated (may take a minute)
- Verify with: `torch.cuda.is_available()`

### Out of Memory
- Reduce `max_sequences` (try 50 instead of 100)
- Reduce `max_length` (try 64 instead of 128)
- Restart runtime: **Runtime > Restart runtime**

### Slow Performance
- Ensure GPU is enabled
- Check GPU utilization: `!nvidia-smi`
- Reduce dataset size for initial testing

### Import Errors
- Make sure you ran `!pip install -e .` after cloning
- Restart runtime after installing packages
- Check that you're in the correct directory: `!pwd`

## Example Complete Workflow

```python
# Cell 1: Setup
!pip install -q torch transformers numpy scipy scikit-learn faiss-cpu nltk matplotlib seaborn pandas tqdm datasets statsmodels
import nltk
nltk.download('punkt', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)
nltk.download('universal_tagset', quiet=True)

# Cell 2: Clone repo
!git clone https://github.com/YOUR_USERNAME/basin-compression-analysis.git
%cd basin-compression-analysis
!pip install -e .

# Cell 3: Verify GPU
import torch
assert torch.cuda.is_available(), "Enable GPU in Runtime settings!"
print(f"Using GPU: {torch.cuda.get_device_name(0)}")

# Cell 4: Run experiment
!python scripts/run_memorization.py --max_sequences 100 --k_neighbors 15

# Cell 5: View results
from IPython.display import Image, display
display(Image('memorization_experiment_summary.png'))
```

## Next Steps

1. Start with a small experiment to verify everything works
2. Gradually increase dataset size as needed
3. Save results to Google Drive for persistence
4. Use the generated visualizations for analysis

For more details, see the main `README.md` and `experiment-proposal.md`.

