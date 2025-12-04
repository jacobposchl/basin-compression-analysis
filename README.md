# Compression Basins in Language Models

A Python package for investigating compression basins in transformer language models, particularly GPT-2. This implementation adapts the compression basin framework from VAEs to analyze how language models organize information across layers.

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)

### Setup

1. Clone this repository:
```bash
git clone <repository-url>
cd basin-compression-analysis
```

2. Install the package and dependencies:
```bash
pip install -r requirements.txt
pip install -e .
```

3. Download required NLTK data:
```python
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')
```

4. Download spaCy model (if using POS tagging):
```bash
python -m spacy download en_core_web_sm
```

## Quick Start

### Running Individual Experiments

```bash
# Memorization experiment
python scripts/run_memorization.py --max_sequences 100

# Token importance experiment
python scripts/run_token_importance.py --max_sequences 100

# Linguistic structure experiment
python scripts/run_linguistic.py --max_sequences 100
```

### Running Full Pipeline

```bash
python scripts/run_full_pipeline.py --max_sequences 500
```

### Using as a Library

```python
from compression_lm.models.model_loader import load_model
from compression_lm.models.extract_states import extract_hidden_states
from compression_lm.compression.metric import compute_compression_scores_layer

# Load model
model, tokenizer = load_model('gpt2')

# Extract hidden states
hidden_states, tokens, token_ids = extract_hidden_states(
    model, tokenizer, "The quick brown fox jumps over the lazy dog."
)

# Compute compression scores
# ... (see examples in scripts/)
```

## Project Structure

```
compression_lm/
├── data/           # Dataset loading and preprocessing
├── models/         # Model loading and state extraction
├── compression/    # Core compression metric computation
├── analysis/       # Statistical analysis and visualization
└── experiments/    # Main experiment implementations
```

## Research Questions

This package investigates:

1. **Memorization Detection**: Do memorized sequences exhibit different compression patterns than novel sequences?
2. **Token Importance**: Does compression predict which tokens are important for predictions?
3. **Linguistic Structure**: Do compression patterns align with linguistic categories (POS tags)?
4. **Layer-wise Organization**: How do compression patterns evolve across transformer layers?

## Documentation

For detailed implementation guide and methodology, see `experiment-proposal.md`.

## License

[Add your license here]

## Citation

[Add citation information when published]

