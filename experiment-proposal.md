# Compression Basins in Large Language Models: Complete Implementation Guide

## Executive Summary and Research Objectives

This document provides a complete implementation guide for investigating whether compression basins in transformer language models reveal interpretable structure about memorization, linguistic processing, and token importance. The core hypothesis is that different regions of a language model's hidden state space compress information at different rates, and these compression patterns correlate with meaningful properties like whether sequences are memorized versus generalized, which tokens are important for predictions, and how linguistic structure is represented across layers.

The experiment adapts the compression basin framework originally developed for VAEs to the sequential, discrete token setting of transformer language models. Success in this experiment would provide a new geometric lens for understanding how language models organize knowledge, with immediate applications to memorization detection, model interpretability, and understanding emergent capabilities.

## Experimental Design Overview

### Core Research Questions

1. **Memorization Detection**: Do memorized sequences exhibit different compression patterns than novel generated sequences? Can we use compression scores to identify which parts of the model's output come from retrieval versus reasoning?

2. **Token Importance**: Does compression predict which tokens are important for downstream predictions? Do high-attention tokens or high-gradient tokens correspond to low-compression regions where the model preserves fine distinctions?

3. **Linguistic Structure**: Do compression patterns align with linguistic categories? Do function words compress more than content words? Do nouns compress differently than verbs?

4. **Layer-wise Organization**: How do compression patterns evolve across transformer layers? Do early layers compress syntactic regularities while late layers compress semantic similarities, or vice versa?

### Why These Questions Matter

Memorization in language models is a critical concern for privacy, copyright, and model reliability. Existing approaches for detecting memorization rely on comparing model outputs to training data or measuring perplexity differences. A geometric approach through compression basins would provide complementary information about the internal representation structure that produces memorization, potentially revealing cases where models memorize patterns not easily detected by string matching.

Token importance and linguistic structure questions connect to the broader interpretability agenda. If compression reveals which tokens matter or how linguistic categories are organized, this provides actionable insight for model analysis, debugging, and improvement. Layer-wise analysis addresses fundamental questions about how transformers organize computation hierarchically.

### High-Level Experimental Pipeline

```
1. Model Selection & Data Preparation
   ↓
2. Extract Hidden States Across All Layers
   ↓
3. Compute Compression Scores for Each Token
   ↓
4. Correlate Compression with Target Properties
   (memorization, attention, POS tags, etc.)
   ↓
5. Analyze Layer-wise Patterns
   ↓
6. Statistical Testing & Visualization
   ↓
7. Ablations & Robustness Checks
```

## Model and Dataset Selection

### Primary Model: GPT-2 Small

**Why GPT-2:**
- Fully open source with extensive documentation
- Well-studied architecture that researchers understand
- Fast enough to iterate on consumer GPUs
- Multiple size variants (small, medium, large) for scaling analysis
- Causal language model structure makes sequence processing clean

**Why Small variant first:**
- 12 layers, 768 hidden dimensions, 12 attention heads
- ~117M parameters - fits comfortably in Colab GPU memory
- Fast inference allows processing thousands of sequences
- Can scale to medium/large if results are promising

**Model Loading:**
```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

model_name = "gpt2"  # This is GPT-2 Small
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model.eval()  # Put in evaluation mode
model = model.cuda()  # Move to GPU
```

### Dataset Selection Strategy

You need multiple datasets for different hypotheses:

**For Memorization Testing:**

1. **WikiText-103 Memorization Set**
   - Take sequences known to appear in training data
   - GPT-2 was trained on WebText which overlaps with Wikipedia
   - Extract passages that GPT-2 can reproduce exactly (exact string match)
   - Compare compression of these memorized sequences vs. paraphrased versions

2. **Controlled Memorization Experiment**
   - Fine-tune GPT-2 on specific passages (e.g., 50 Shakespeare quotes)
   - Test compression on: (a) fine-tuned passages, (b) other Shakespeare, (c) unrelated text
   - This gives you ground truth for what's memorized

**For Linguistic Structure Testing:**

1. **Penn Treebank**
   - Includes POS tags and syntactic annotations
   - Standard benchmark for linguistic analysis
   - ~1M words with detailed annotations
   
2. **Stanford Sentiment Treebank (if needed)**
   - Sentence-level sentiment labels
   - Can test if compression varies with sentiment

**For Token Importance Testing:**
- Use the same datasets but focus on extracting attention patterns and gradient-based importance scores

### Data Preparation Requirements

**Text preprocessing:**
- Tokenize using GPT-2 tokenizer (BPE tokenization)
- Sequences should be 50-256 tokens for manageability
- Truncate or split longer documents
- Maintain metadata: original text, POS tags, memorization labels

**Expected data structure:**
```python
data_samples = [
    {
        'text': "The quick brown fox jumps over the lazy dog.",
        'tokens': [464, 2068, 7586, 21831, ...],  # Token IDs
        'pos_tags': ['DET', 'ADJ', 'ADJ', 'NOUN', ...],
        'is_memorized': False,
        'source': 'wikitext'
    },
    # ... more samples
]
```

## Core Implementation: Compression Metric for Transformers

### Conceptual Adaptation from VAEs

In the VAE experiments, compression was measured by:
1. Finding k nearest neighbors in pixel space
2. Encoding all neighbors to latent space
3. Measuring variance of latent codes

For transformers, we adapt this to:
1. Finding k nearest neighbors in hidden state space at layer L
2. Passing those neighbors through layer L+1
3. Measuring variance of resulting hidden states

**Key difference**: We're measuring compression happening *between* layers rather than from input to latent space. This captures how each transformer layer collapses distinctions that existed in the previous layer.

### Detailed Algorithm

**For each token at each layer:**

```
Input: Hidden state h_i^L for token i at layer L
Output: Compression score C_i^L

1. Extract all hidden states at layer L: {h_1^L, h_2^L, ..., h_N^L}

2. Find k nearest neighbors of h_i^L in this set:
   neighbors = k_nearest_neighbors(h_i^L, {h_j^L}, k=15)
   
3. Pass each neighbor through the next layer to get h_j^(L+1):
   - This requires running the layer L→L+1 transformation
   - For GPT-2: attention + MLP block
   
4. Measure variance of resulting hidden states:
   var = sum_across_dimensions(variance({h_neighbor^(L+1)}))
   
5. Compression score (log scale for stability):
   C_i^L = -log(var + epsilon)
   
Higher C_i^L = more compression (neighbors collapse together)
Lower C_i^L = less compression (distinctions preserved)
```

### Neighborhood Definition Options

**Option 1: Same-layer neighbors (recommended for start)**
- Find neighbors among all tokens at the same layer
- Most direct adaptation from VAE approach
- Captures: "which tokens have similar representations at layer L?"

**Option 2: Cross-sequence neighbors**
- Find neighbors across different input sequences
- Requires processing multiple sentences simultaneously
- Captures: "which tokens from different contexts map similarly?"

**Option 3: Same-token-type neighbors**
- Find neighbors among instances of the same token type in different contexts
- Example: all instances of "the" across different sentences
- Captures: "how much does context matter for this token?"

**Start with Option 1**, then explore Options 2-3 if needed.

### Implementation Considerations

**Computational Efficiency:**
- Hidden states are 768-dimensional for GPT-2 Small
- With 1000 sequences of 100 tokens each = 100,000 hidden states per layer
- k-NN search in 100k points × 768 dims is expensive
- Solutions:
  - Use approximate k-NN (FAISS library)
  - Process in batches
  - Sample subset of tokens for initial exploration

**Memory Management:**
- Storing all hidden states for all layers is memory-intensive
- 12 layers × 100k tokens × 768 dims × 4 bytes = ~3.5 GB
- Strategy: Process one layer at a time, compute compression, discard states

**Numerical Stability:**
- Use epsilon = 1e-10 when taking logarithms
- Check for NaN/Inf values in compression scores
- Normalize hidden states before computing distances if needed

## Step-by-Step Implementation Guide

### Phase 1: Infrastructure Setup (Days 1-2)

**File Structure:**
```
compression_lm/
├── data/
│   ├── load_datasets.py      # Data loading utilities
│   ├── preprocess.py          # Tokenization, POS tagging
│   └── memorization.py        # Memorization detection
├── models/
│   ├── model_loader.py        # Load GPT-2 variants
│   └── extract_states.py      # Hidden state extraction
├── compression/
│   ├── metric.py              # Core compression computation
│   ├── neighborhoods.py       # k-NN implementation
│   └── utils.py               # Helper functions
├── analysis/
│   ├── correlations.py        # Statistical tests
│   ├── visualizations.py      # Plotting functions
│   └── layer_analysis.py      # Layer-wise patterns
├── experiments/
│   ├── memorization.py        # Memorization experiments
│   ├── token_importance.py    # Attention/gradient analysis
│   └── linguistic.py          # POS/syntactic analysis
├── notebooks/
│   └── main_experiment.ipynb  # Colab notebook
└── README.md
```

**Key Dependencies:**
```python
# requirements.txt
torch>=2.0.0
transformers>=4.30.0
numpy>=1.24.0
scipy>=1.10.0
scikit-learn>=1.3.0
faiss-cpu>=1.7.4  # or faiss-gpu for GPU acceleration
nltk>=3.8.0
spacy>=3.5.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
tqdm>=4.65.0
```

**Initial Setup Script:**
```python
# setup.py
import nltk
import spacy

# Download required NLTK data
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

# Download spaCy model for POS tagging
# Run: python -m spacy download en_core_web_sm

print("Setup complete!")
```

### Phase 2: Hidden State Extraction (Days 2-3)

**Core extraction function:**

```python
# models/extract_states.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

def extract_hidden_states(model, tokenizer, text, layers='all'):
    """
    Extract hidden states from GPT-2 for given text.
    
    Args:
        model: GPT2LMHeadModel
        tokenizer: GPT2Tokenizer
        text: Input text string
        layers: 'all' or list of layer indices [0, 1, 2, ...]
    
    Returns:
        hidden_states: List of tensors, one per layer
                      Each tensor shape: [seq_len, hidden_dim]
        tokens: List of token strings
        token_ids: List of token IDs
    """
    # Tokenize
    inputs = tokenizer(text, return_tensors='pt')
    token_ids = inputs['input_ids'][0]
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    # Move to GPU
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    # outputs.hidden_states is tuple of (num_layers+1) tensors
    # Each tensor shape: [batch_size, seq_len, hidden_dim]
    # First element is embedding layer, rest are transformer layers
    
    hidden_states = []
    if layers == 'all':
        # Skip embedding layer (index 0), take transformer layers
        for layer_states in outputs.hidden_states[1:]:
            # Remove batch dimension, move to CPU
            hidden_states.append(layer_states[0].cpu())
    else:
        for layer_idx in layers:
            hidden_states.append(outputs.hidden_states[layer_idx + 1][0].cpu())
    
    return hidden_states, tokens, token_ids.cpu().tolist()

def extract_dataset_states(model, tokenizer, texts, max_length=128):
    """
    Extract hidden states for multiple texts.
    
    Args:
        model: GPT2LMHeadModel
        tokenizer: GPT2Tokenizer
        texts: List of text strings
        max_length: Maximum sequence length
    
    Returns:
        all_states: Dict mapping layer_idx -> list of hidden state tensors
        all_tokens: List of token lists
        metadata: List of dicts with sequence information
    """
    all_states = {i: [] for i in range(12)}  # GPT-2 Small has 12 layers
    all_tokens = []
    metadata = []
    
    for idx, text in enumerate(tqdm(texts, desc="Extracting states")):
        # Truncate if needed
        tokens = tokenizer.encode(text)
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
            text = tokenizer.decode(tokens)
        
        # Extract states
        hidden_states, token_strs, token_ids = extract_hidden_states(
            model, tokenizer, text
        )
        
        # Store by layer
        for layer_idx, states in enumerate(hidden_states):
            all_states[layer_idx].append(states)
        
        all_tokens.append(token_strs)
        metadata.append({
            'text_idx': idx,
            'original_text': text,
            'num_tokens': len(token_strs)
        })
    
    return all_states, all_tokens, metadata
```

**Testing the extraction:**
```python
# Test script
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()
model = model.cuda()

test_text = "The quick brown fox jumps over the lazy dog."
hidden_states, tokens, token_ids = extract_hidden_states(model, tokenizer, test_text)

print(f"Number of layers: {len(hidden_states)}")
print(f"Number of tokens: {len(tokens)}")
print(f"Hidden state shape per layer: {hidden_states[0].shape}")
print(f"Tokens: {tokens}")

# Expected output:
# Number of layers: 12
# Number of tokens: 10 (approximately, depends on tokenization)
# Hidden state shape per layer: torch.Size([10, 768])
# Tokens: ['The', ' quick', ' brown', ' fox', ...]
```

### Phase 3: Compression Metric Implementation (Days 3-5)

**Core compression computation:**

```python
# compression/metric.py
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import faiss

def compute_compression_scores_layer(
    hidden_states_L,      # Hidden states at layer L: list of tensors
    hidden_states_Lplus1, # Hidden states at layer L+1: list of tensors
    k=15,                 # Number of neighbors
    use_faiss=True,       # Use FAISS for speed
    normalize=False       # Normalize hidden states before distance computation
):
    """
    Compute compression scores for all tokens at layer L.
    
    Measures how much layer L+1 compresses neighborhoods from layer L.
    
    Args:
        hidden_states_L: List of tensors, each [seq_len, hidden_dim]
        hidden_states_Lplus1: List of tensors, each [seq_len, hidden_dim]
        k: Number of nearest neighbors
        use_faiss: Use FAISS for faster k-NN
        normalize: Normalize vectors before computing distances
    
    Returns:
        compression_scores: Array of compression scores, one per token
        metadata: Dict with additional information
    """
    # Flatten all hidden states into single arrays
    all_states_L = []
    all_states_Lplus1 = []
    sequence_indices = []  # Track which sequence each token came from
    position_indices = []  # Track position within sequence
    
    for seq_idx, (states_L, states_Lplus1) in enumerate(
        zip(hidden_states_L, hidden_states_Lplus1)
    ):
        seq_len = states_L.shape[0]
        all_states_L.append(states_L.numpy())
        all_states_Lplus1.append(states_Lplus1.numpy())
        sequence_indices.extend([seq_idx] * seq_len)
        position_indices.extend(list(range(seq_len)))
    
    # Concatenate into single arrays
    all_states_L = np.vstack(all_states_L)  # [total_tokens, hidden_dim]
    all_states_Lplus1 = np.vstack(all_states_Lplus1)
    
    print(f"Total tokens: {all_states_L.shape[0]}")
    print(f"Hidden dimension: {all_states_L.shape[1]}")
    
    # Normalize if requested
    if normalize:
        all_states_L = all_states_L / (np.linalg.norm(all_states_L, axis=1, keepdims=True) + 1e-10)
    
    # Find k nearest neighbors in layer L
    if use_faiss and all_states_L.shape[0] > 10000:
        # Use FAISS for large datasets
        index = faiss.IndexFlatL2(all_states_L.shape[1])
        index.add(all_states_L.astype('float32'))
        distances, neighbor_indices = index.search(all_states_L.astype('float32'), k)
    else:
        # Use sklearn for smaller datasets
        nbrs = NearestNeighbors(n_neighbors=k, algorithm='auto', metric='euclidean')
        nbrs.fit(all_states_L)
        distances, neighbor_indices = nbrs.kneighbors(all_states_L)
    
    # Compute compression scores
    compression_scores = []
    
    for i in range(len(all_states_L)):
        # Get indices of k nearest neighbors at layer L
        neighbor_idx = neighbor_indices[i]  # [k]
        
        # Get their representations at layer L+1
        neighbor_states_Lplus1 = all_states_Lplus1[neighbor_idx]  # [k, hidden_dim]
        
        # Compute variance across neighbors in layer L+1
        variance = np.var(neighbor_states_Lplus1, axis=0).sum()  # Scalar
        
        # Compression score (negative log variance)
        compression_score = -np.log(variance + 1e-10)
        compression_scores.append(compression_score)
    
    compression_scores = np.array(compression_scores)
    
    # Package metadata
    metadata = {
        'sequence_indices': sequence_indices,
        'position_indices': position_indices,
        'mean_score': compression_scores.mean(),
        'std_score': compression_scores.std(),
        'min_score': compression_scores.min(),
        'max_score': compression_scores.max()
    }
    
    return compression_scores, metadata

def compute_all_layers_compression(all_states, k=15, use_faiss=True):
    """
    Compute compression scores for all layers.
    
    Args:
        all_states: Dict mapping layer_idx -> list of hidden state tensors
        k: Number of neighbors
        use_faiss: Use FAISS for speed
    
    Returns:
        layer_compression: Dict mapping layer_idx -> compression scores array
        layer_metadata: Dict mapping layer_idx -> metadata dict
    """
    num_layers = len(all_states)
    layer_compression = {}
    layer_metadata = {}
    
    # Compute compression for each layer transition
    for layer_idx in range(num_layers - 1):  # Stop before last layer
        print(f"\nComputing compression for layer {layer_idx} -> {layer_idx + 1}")
        
        scores, meta = compute_compression_scores_layer(
            all_states[layer_idx],
            all_states[layer_idx + 1],
            k=k,
            use_faiss=use_faiss
        )
        
        layer_compression[layer_idx] = scores
        layer_metadata[layer_idx] = meta
        
        print(f"  Mean compression: {meta['mean_score']:.4f}")
        print(f"  Std compression: {meta['std_score']:.4f}")
    
    return layer_compression, layer_metadata
```

**Testing compression computation:**

```python
# Test on small sample
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A journey of a thousand miles begins with a single step.",
    "To be or not to be, that is the question."
]

# Extract states
all_states, all_tokens, metadata = extract_dataset_states(
    model, tokenizer, test_texts, max_length=50
)

# Compute compression
layer_compression, layer_metadata = compute_all_layers_compression(
    all_states, k=10, use_faiss=False
)

# Visualize compression for first layer
import matplotlib.pyplot as plt

scores = layer_compression[0]
plt.figure(figsize=(10, 4))
plt.hist(scores, bins=50, alpha=0.7, edgecolor='black')
plt.xlabel('Compression Score')
plt.ylabel('Count')
plt.title('Distribution of Compression Scores (Layer 0→1)')
plt.axvline(scores.mean(), color='red', linestyle='--', label=f'Mean: {scores.mean():.2f}')
plt.legend()
plt.show()

print(f"Compression scores shape: {scores.shape}")
print(f"Example scores: {scores[:10]}")
```

### Phase 4: Memorization Experiments (Days 5-8)

**Memorization detection strategy:**

```python
# experiments/memorization.py
import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr

def detect_memorized_sequences(model, tokenizer, texts, min_length=20):
    """
    Detect which sequences the model can reproduce exactly.
    
    Strategy: Give the model the first N tokens, see if it generates
    the exact continuation.
    
    Args:
        model: GPT2LMHeadModel
        tokenizer: GPT2Tokenizer
        texts: List of text strings to test
        min_length: Minimum sequence length to consider
    
    Returns:
        memorization_labels: List of bools indicating memorization
        reproduction_accuracy: List of floats indicating match quality
    """
    memorization_labels = []
    reproduction_accuracy = []
    
    model.eval()
    
    for text in tqdm(texts, desc="Testing memorization"):
        # Tokenize
        token_ids = tokenizer.encode(text)
        
        if len(token_ids) < min_length:
            memorization_labels.append(False)
            reproduction_accuracy.append(0.0)
            continue
        
        # Split into prompt and target
        split_point = len(token_ids) // 3  # Use first third as prompt
        prompt_ids = token_ids[:split_point]
        target_ids = token_ids[split_point:]
        
        # Generate continuation
        prompt_tensor = torch.tensor([prompt_ids]).cuda()
        
        with torch.no_grad():
            generated = model.generate(
                prompt_tensor,
                max_length=len(token_ids),
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_ids = generated[0][split_point:].cpu().tolist()
        
        # Compare to ground truth
        min_len = min(len(generated_ids), len(target_ids))
        matches = sum([g == t for g, t in zip(generated_ids[:min_len], target_ids[:min_len])])
        accuracy = matches / len(target_ids) if len(target_ids) > 0 else 0.0
        
        # Consider memorized if >80% exact match
        is_memorized = accuracy > 0.8
        
        memorization_labels.append(is_memorized)
        reproduction_accuracy.append(accuracy)
    
    return memorization_labels, reproduction_accuracy

def analyze_memorization_compression(
    compression_scores,
    memorization_labels,
    sequence_indices,  # From compression metadata
    layer_idx=None
):
    """
    Analyze relationship between compression and memorization.
    
    Args:
        compression_scores: Array of compression scores for all tokens
        memorization_labels: List of bools, one per sequence
        sequence_indices: Array mapping each token to its sequence
        layer_idx: Which layer these scores came from
    
    Returns:
        results: Dict with statistical tests and visualizations
    """
    # Convert to arrays
    compression_scores = np.array(compression_scores)
    memorization_labels = np.array(memorization_labels)
    sequence_indices = np.array(sequence_indices)
    
    # Compute mean compression per sequence
    unique_sequences = np.unique(sequence_indices)
    sequence_compression = []
    sequence_memorization = []
    
    for seq_idx in unique_sequences:
        mask = sequence_indices == seq_idx
        seq_compression = compression_scores[mask].mean()
        seq_memorization = memorization_labels[seq_idx]
        
        sequence_compression.append(seq_compression)
        sequence_memorization.append(seq_memorization)
    
    sequence_compression = np.array(sequence_compression)
    sequence_memorization = np.array(sequence_memorization)
    
    # Statistical tests
    memorized_compression = sequence_compression[sequence_memorization]
    novel_compression = sequence_compression[~sequence_memorization]
    
    # T-test
    from scipy.stats import ttest_ind
    t_stat, p_value = ttest_ind(memorized_compression, novel_compression)
    
    # Point-biserial correlation (for binary memorization label)
    from scipy.stats import pointbiserialr
    corr, corr_p = pointbiserialr(sequence_memorization, sequence_compression)
    
    # Results dictionary
    results = {
        'layer': layer_idx,
        'correlation': corr,
        'correlation_p': corr_p,
        't_statistic': t_stat,
        't_test_p': p_value,
        'memorized_mean': memorized_compression.mean(),
        'memorized_std': memorized_compression.std(),
        'novel_mean': novel_compression.mean(),
        'novel_std': novel_compression.std(),
        'n_memorized': memorized_compression.shape[0],
        'n_novel': novel_compression.shape[0],
        'sequence_compression': sequence_compression,
        'sequence_memorization': sequence_memorization
    }
    
    # Print summary
    print(f"\nMemorization Analysis (Layer {layer_idx}):")
    print(f"  Correlation (point-biserial): r = {corr:.4f}, p = {corr_p:.4e}")
    print(f"  Memorized sequences: mean = {memorized_compression.mean():.4f}, std = {memorized_compression.std():.4f}, n = {len(memorized_compression)}")
    print(f"  Novel sequences: mean = {novel_compression.mean():.4f}, std = {novel_compression.std():.4f}, n = {len(novel_compression)}")
    print(f"  Difference: {memorized_compression.mean() - novel_compression.mean():.4f}")
    print(f"  T-test: t = {t_stat:.4f}, p = {p_value:.4e}")
    
    if abs(corr) > 0.4 and corr_p < 0.01:
        print("  ✓ SUCCESS: Strong significant correlation!")
    elif abs(corr) > 0.2 and corr_p < 0.05:
        print("  ~ MODERATE: Weak to moderate correlation")
    else:
        print("  ✗ WEAK: No significant correlation")
    
    return results

def visualize_memorization_results(results):
    """
    Create visualizations for memorization analysis.
    
    Args:
        results: Dict from analyze_memorization_compression
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Plot 1: Scatter plot
    ax = axes[0]
    memorized = results['sequence_memorization']
    compression = results['sequence_compression']
    
    colors = ['red' if m else 'blue' for m in memorized]
    ax.scatter(compression, np.random.randn(len(compression)) * 0.1,
              c=colors, alpha=0.5, s=30)
    ax.axvline(results['memorized_mean'], color='red', linestyle='--', 
              label=f"Memorized: {results['memorized_mean']:.3f}")
    ax.axvline(results['novel_mean'], color='blue', linestyle='--',
              label=f"Novel: {results['novel_mean']:.3f}")
    ax.set_xlabel('Compression Score')
    ax.set_ylabel('Jitter (for visualization)')
    ax.set_title(f"Compression by Memorization Status\nLayer {results['layer']}")
    ax.legend()
    
    # Plot 2: Distribution comparison
    ax = axes[1]
    memorized_scores = compression[memorized]
    novel_scores = compression[~memorized]
    
    ax.hist(memorized_scores, bins=20, alpha=0.6, label='Memorized', color='red', edgecolor='black')
    ax.hist(novel_scores, bins=20, alpha=0.6, label='Novel', color='blue', edgecolor='black')
    ax.set_xlabel('Compression Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution Comparison')
    ax.legend()
    
    # Plot 3: Statistics summary
    ax = axes[2]
    ax.axis('off')
    
    stats_text = f"""
    Layer {results['layer']} Statistics:
    
    Correlation: r = {results['correlation']:.4f}
    P-value: {results['correlation_p']:.2e}
    
    Memorized (n={results['n_memorized']}):
      Mean: {results['memorized_mean']:.4f}
      Std: {results['memorized_std']:.4f}
    
    Novel (n={results['n_novel']}):
      Mean: {results['novel_mean']:.4f}
      Std: {results['novel_std']:.4f}
    
    Difference: {results['memorized_mean'] - results['novel_mean']:.4f}
    T-test p: {results['t_test_p']:.2e}
    """
    
    ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.tight_layout()
    return fig
```

**Full memorization experiment pipeline:**

```python
# Main memorization experiment
def run_memorization_experiment(
    model,
    tokenizer,
    texts,
    k_neighbors=15,
    max_sequences=500
):
    """
    Complete pipeline for memorization experiment.
    
    Args:
        model: GPT2LMHeadModel
        tokenizer: GPT2Tokenizer
        texts: List of text strings
        k_neighbors: Number of neighbors for compression
        max_sequences: Maximum number of sequences to process
    
    Returns:
        full_results: Dict with all results across layers
    """
    # Limit to max_sequences
    if len(texts) > max_sequences:
        texts = texts[:max_sequences]
    
    print(f"Running memorization experiment on {len(texts)} sequences...")
    
    # Step 1: Detect memorized sequences
    print("\n1. Detecting memorized sequences...")
    memorization_labels, reproduction_accuracy = detect_memorized_sequences(
        model, tokenizer, texts
    )
    
    n_memorized = sum(memorization_labels)
    print(f"   Found {n_memorized}/{len(texts)} memorized sequences ({100*n_memorized/len(texts):.1f}%)")
    
    # Step 2: Extract hidden states
    print("\n2. Extracting hidden states...")
    all_states, all_tokens, metadata = extract_dataset_states(
        model, tokenizer, texts, max_length=128
    )
    
    # Step 3: Compute compression scores
    print("\n3. Computing compression scores...")
    layer_compression, layer_metadata = compute_all_layers_compression(
        all_states, k=k_neighbors, use_faiss=True
    )
    
    # Step 4: Analyze each layer
    print("\n4. Analyzing memorization-compression relationship...")
    full_results = {}
    
    for layer_idx in range(len(layer_compression)):
        results = analyze_memorization_compression(
            layer_compression[layer_idx],
            memorization_labels,
            layer_metadata[layer_idx]['sequence_indices'],
            layer_idx=layer_idx
        )
        
        full_results[layer_idx] = results
    
    # Step 5: Create summary visualizations
    print("\n5. Creating visualizations...")
    
    # Plot layer-wise correlation
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Correlation across layers
    ax = axes[0, 0]
    layers = list(full_results.keys())
    correlations = [full_results[l]['correlation'] for l in layers]
    ax.plot(layers, correlations, 'o-', linewidth=2, markersize=8)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax.axhline(y=0.4, color='green', linestyle='--', alpha=0.5, label='Strong threshold')
    ax.axhline(y=-0.4, color='green', linestyle='--', alpha=0.5)
    ax.set_xlabel('Layer')
    ax.set_ylabel('Correlation (r)')
    ax.set_title('Memorization-Compression Correlation Across Layers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Mean compression across layers
    ax = axes[0, 1]
    memorized_means = [full_results[l]['memorized_mean'] for l in layers]
    novel_means = [full_results[l]['novel_mean'] for l in layers]
    ax.plot(layers, memorized_means, 'o-', linewidth=2, label='Memorized', color='red')
    ax.plot(layers, novel_means, 'o-', linewidth=2, label='Novel', color='blue')
    ax.set_xlabel('Layer')
    ax.set_ylabel('Mean Compression Score')
    ax.set_title('Mean Compression by Memorization Status')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Best layer detailed view
    best_layer = max(full_results.keys(), key=lambda l: abs(full_results[l]['correlation']))
    
    ax = axes[1, 0]
    memorized = full_results[best_layer]['sequence_memorization']
    compression = full_results[best_layer]['sequence_compression']
    colors = ['red' if m else 'blue' for m in memorized]
    ax.scatter(range(len(compression)), compression, c=colors, alpha=0.6, s=30)
    ax.set_xlabel('Sequence Index')
    ax.set_ylabel('Compression Score')
    ax.set_title(f'Best Layer: {best_layer} (r={full_results[best_layer]["correlation"]:.3f})')
    
    # Statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = f"""
    MEMORIZATION EXPERIMENT SUMMARY
    
    Total sequences: {len(texts)}
    Memorized: {n_memorized} ({100*n_memorized/len(texts):.1f}%)
    Novel: {len(texts) - n_memorized} ({100*(len(texts)-n_memorized)/len(texts):.1f}%)
    
    Best Layer: {best_layer}
    Best Correlation: {full_results[best_layer]['correlation']:.4f}
    P-value: {full_results[best_layer]['correlation_p']:.2e}
    
    Difference in compression:
    {full_results[best_layer]['memorized_mean'] - full_results[best_layer]['novel_mean']:.4f}
    
    Overall finding:
    {"✓ Strong effect found!" if abs(full_results[best_layer]['correlation']) > 0.4 else "~ Moderate effect" if abs(full_results[best_layer]['correlation']) > 0.2 else "✗ Weak/no effect"}
    """
    
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
           verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig('memorization_experiment_summary.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return full_results
```

### Phase 5: Token Importance Experiments (Days 8-10)

**Token importance via attention:**

```python
# experiments/token_importance.py

def extract_attention_weights(model, tokenizer, text):
    """
    Extract attention weights from all layers.
    
    Args:
        model: GPT2LMHeadModel
        tokenizer: GPT2Tokenizer
        text: Input text string
    
    Returns:
        attention_weights: List of tensors, one per layer
                          Each shape: [num_heads, seq_len, seq_len]
        tokens: List of token strings
    """
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)
    
    # outputs.attentions is tuple of (num_layers) tensors
    # Each tensor shape: [batch_size, num_heads, seq_len, seq_len]
    
    attention_weights = []
    for layer_attn in outputs.attentions:
        attention_weights.append(layer_attn[0].cpu())  # Remove batch dim
    
    tokens = [tokenizer.decode([tid]) for tid in inputs['input_ids'][0]]
    
    return attention_weights, tokens

def compute_token_importance_attention(attention_weights):
    """
    Compute token importance from attention weights.
    
    Strategy: Sum of incoming attention across all layers and heads.
    
    Args:
        attention_weights: List of attention tensors [num_heads, seq_len, seq_len]
    
    Returns:
        importance_scores: Array of shape [seq_len]
    """
    seq_len = attention_weights[0].shape[-1]
    importance_scores = np.zeros(seq_len)
    
    for layer_attn in attention_weights:
        # Sum over heads and source positions (incoming attention)
        # layer_attn shape: [num_heads, seq_len, seq_len]
        # Sum over dim 1 (queries) to get incoming attention to each key
        incoming_attn = layer_attn.sum(dim=0).sum(dim=0).numpy()  # [seq_len]
        importance_scores += incoming_attn
    
    return importance_scores

def compute_token_importance_gradient(model, tokenizer, text, target_position=-1):
    """
    Compute token importance via gradient magnitude.
    
    Strategy: How much does each token's representation affect the final prediction?
    
    Args:
        model: GPT2LMHeadModel
        tokenizer: GPT2Tokenizer
        text: Input text string
        target_position: Which position to compute gradients for (-1 = last)
    
    Returns:
        importance_scores: Dict mapping layer_idx -> gradient magnitudes [seq_len]
    """
    inputs = tokenizer(text, return_tensors='pt')
    inputs = {k: v.cuda() for k, v in inputs.items()}
    
    # Forward pass with gradients
    model.zero_grad()
    outputs = model(**inputs, output_hidden_states=True)
    
    # Get logits for target position
    logits = outputs.logits[0, target_position, :]  # [vocab_size]
    
    # Compute loss (negative log prob of actual next token)
    # Or just use max logit as proxy for "confidence"
    loss = -logits.max()
    
    loss.backward()
    
    # Extract gradient magnitudes for each layer
    importance_scores = {}
    
    for layer_idx, hidden_state in enumerate(outputs.hidden_states[1:]):  # Skip embedding
        # hidden_state shape: [batch_size, seq_len, hidden_dim]
        if hidden_state.grad is not None:
            grad_magnitudes = hidden_state.grad[0].norm(dim=-1).cpu().numpy()  # [seq_len]
            importance_scores[layer_idx] = grad_magnitudes
    
    return importance_scores

def analyze_importance_compression(
    compression_scores,
    importance_scores,
    sequence_indices,
    layer_idx=None
):
    """
    Analyze relationship between compression and token importance.
    
    Args:
        compression_scores: Array of compression scores
        importance_scores: Array of importance scores (same length)
        sequence_indices: Array mapping tokens to sequences
        layer_idx: Which layer
    
    Returns:
        results: Dict with correlations and visualizations
    """
    # Ensure same length
    min_len = min(len(compression_scores), len(importance_scores))
    compression_scores = compression_scores[:min_len]
    importance_scores = importance_scores[:min_len]
    
    # Compute correlation
    corr, p_value = pearsonr(compression_scores, importance_scores)
    spearman_corr, spearman_p = spearmanr(compression_scores, importance_scores)
    
    # Quartile analysis
    quartiles = np.percentile(compression_scores, [25, 50, 75])
    q1_mask = compression_scores <= quartiles[0]
    q2_mask = (compression_scores > quartiles[0]) & (compression_scores <= quartiles[1])
    q3_mask = (compression_scores > quartiles[1]) & (compression_scores <= quartiles[2])
    q4_mask = compression_scores > quartiles[2]
    
    results = {
        'layer': layer_idx,
        'pearson_r': corr,
        'pearson_p': p_value,
        'spearman_r': spearman_corr,
        'spearman_p': spearman_p,
        'q1_importance': importance_scores[q1_mask].mean(),
        'q2_importance': importance_scores[q2_mask].mean(),
        'q3_importance': importance_scores[q3_mask].mean(),
        'q4_importance': importance_scores[q4_mask].mean(),
        'compression_scores': compression_scores,
        'importance_scores': importance_scores
    }
    
    print(f"\nToken Importance Analysis (Layer {layer_idx}):")
    print(f"  Pearson correlation: r = {corr:.4f}, p = {p_value:.4e}")
    print(f"  Spearman correlation: ρ = {spearman_corr:.4f}, p = {spearman_p:.4e}")
    print(f"  Q1 (low compression) importance: {results['q1_importance']:.4f}")
    print(f"  Q4 (high compression) importance: {results['q4_importance']:.4f}")
    
    if abs(corr) > 0.4 and p_value < 0.01:
        print("  ✓ SUCCESS: Strong significant correlation!")
    elif abs(corr) > 0.2:
        print("  ~ MODERATE: Weak to moderate correlation")
    else:
        print("  ✗ WEAK: No significant correlation")
    
    return results

def visualize_importance_results(results):
    """
    Visualize importance-compression relationship.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    compression = results['compression_scores']
    importance = results['importance_scores']
    
    # Scatter plot
    ax = axes[0]
    ax.scatter(compression, importance, alpha=0.3, s=20)
    ax.set_xlabel('Compression Score')
    ax.set_ylabel('Importance Score')
    ax.set_title(f"Compression vs Importance\nLayer {results['layer']} (r={results['pearson_r']:.3f})")
    
    # Add regression line
    z = np.polyfit(compression, importance, 1)
    p = np.poly1d(z)
    x_line = np.linspace(compression.min(), compression.max(), 100)
    ax.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
    
    # Quartile comparison
    ax = axes[1]
    quartile_means = [results['q1_importance'], results['q2_importance'],
                     results['q3_importance'], results['q4_importance']]
    ax.bar(['Q1\n(Low)', 'Q2', 'Q3', 'Q4\n(High)'], quartile_means, 
          color=['blue', 'lightblue', 'orange', 'red'], edgecolor='black')
    ax.set_xlabel('Compression Quartile')
    ax.set_ylabel('Mean Importance')
    ax.set_title('Importance by Compression Quartile')
    
    # Joint distribution
    ax = axes[2]
    h = ax.hist2d(compression, importance, bins=30, cmap='viridis')
    plt.colorbar(h[3], ax=ax, label='Count')
    ax.set_xlabel('Compression Score')
    ax.set_ylabel('Importance Score')
    ax.set_title('Joint Distribution')
    
    plt.tight_layout()
    return fig
```

### Phase 6: Linguistic Structure Analysis (Days 10-12)

**POS tagging and analysis:**

```python
# experiments/linguistic.py
import nltk
from collections import defaultdict

def add_pos_tags(texts, tokens_list):
    """
    Add part-of-speech tags to tokens.
    
    Args:
        texts: List of original text strings
        tokens_list: List of token lists (from tokenizer)
    
    Returns:
        pos_tags_list: List of POS tag lists
    """
    pos_tags_list = []
    
    for text, tokens in zip(texts, tokens_list):
        # Tokenize with NLTK (word-level)
        words = nltk.word_tokenize(text)
        
        # Get POS tags
        pos_tagged = nltk.pos_tag(words, tagset='universal')
        
        # Map to GPT-2 tokens (approximate - tokens may not align perfectly)
        # For now, use simple heuristic
        pos_tags = []
        word_idx = 0
        
        for token in tokens:
            token_clean = token.strip()
            if word_idx < len(pos_tagged) and token_clean:
                # Check if token matches current word
                current_word, current_pos = pos_tagged[word_idx]
                pos_tags.append(current_pos)
                
                # Advance to next word if token seems complete
                if token_clean.isalpha():
                    word_idx += 1
            else:
                pos_tags.append('OTHER')
        
        pos_tags_list.append(pos_tags)
    
    return pos_tags_list

def analyze_pos_compression(
    compression_scores,
    pos_tags,
    sequence_indices,
    layer_idx=None
):
    """
    Analyze compression patterns across POS categories.
    
    Args:
        compression_scores: Array of compression scores
        pos_tags: List of POS tag lists
        sequence_indices: Array mapping tokens to sequences
        layer_idx: Which layer
    
    Returns:
        results: Dict with POS-based statistics
    """
    # Flatten POS tags to match compression scores
    all_pos_tags = []
    for seq_idx, tags in enumerate(pos_tags):
        all_pos_tags.extend(tags)
    
    # Ensure same length
    min_len = min(len(compression_scores), len(all_pos_tags))
    compression_scores = compression_scores[:min_len]
    all_pos_tags = all_pos_tags[:min_len]
    
    # Group by POS category
    pos_compression = defaultdict(list)
    for score, pos in zip(compression_scores, all_pos_tags):
        pos_compression[pos].append(score)
    
    # Compute statistics per POS
    pos_stats = {}
    for pos, scores in pos_compression.items():
        if len(scores) >= 10:  # Only include categories with sufficient samples
            pos_stats[pos] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'count': len(scores)
            }
    
    # Sort by mean compression
    sorted_pos = sorted(pos_stats.items(), key=lambda x: x[1]['mean'], reverse=True)
    
    results = {
        'layer': layer_idx,
        'pos_stats': pos_stats,
        'sorted_pos': sorted_pos,
        'compression_scores': compression_scores,
        'pos_tags': all_pos_tags
    }
    
    print(f"\nPOS Compression Analysis (Layer {layer_idx}):")
    print(f"{'POS Tag':<15} {'Mean':<10} {'Std':<10} {'Count':<10}")
    print("-" * 45)
    for pos, stats in sorted_pos[:10]:  # Top 10
        print(f"{pos:<15} {stats['mean']:<10.4f} {stats['std']:<10.4f} {stats['count']:<10}")
    
    # ANOVA test: Do POS categories differ significantly?
    from scipy.stats import f_oneway
    pos_groups = [scores for pos, scores in pos_compression.items() if len(scores) >= 10]
    if len(pos_groups) >= 3:
        f_stat, p_value = f_oneway(*pos_groups)
        print(f"\nANOVA test: F = {f_stat:.4f}, p = {p_value:.4e}")
        results['anova_f'] = f_stat
        results['anova_p'] = p_value
    
    return results

def visualize_pos_results(results):
    """
    Visualize POS-compression patterns.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Bar chart of mean compression by POS
    ax = axes[0]
    sorted_pos = results['sorted_pos'][:15]  # Top 15
    pos_names = [pos for pos, _ in sorted_pos]
    pos_means = [stats['mean'] for _, stats in sorted_pos]
    pos_stds = [stats['std'] for _, stats in sorted_pos]
    
    ax.barh(pos_names, pos_means, xerr=pos_stds, capsize=5, 
           color='steelblue', edgecolor='black')
    ax.set_xlabel('Mean Compression Score')
    ax.set_ylabel('Part of Speech')
    ax.set_title(f'Compression by POS Category (Layer {results["layer"]})')
    ax.axvline(np.mean(results['compression_scores']), color='red', 
              linestyle='--', alpha=0.7, label='Overall mean')
    ax.legend()
    
    # Distribution comparison for selected POS
    ax = axes[1]
    pos_stats = results['pos_stats']
    
    # Select most frequent categories
    frequent_pos = sorted(pos_stats.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
    
    compression = results['compression_scores']
    pos_tags = results['pos_tags']
    
    for pos, _ in frequent_pos:
        mask = np.array([tag == pos for tag in pos_tags])
        scores = compression[mask]
        ax.hist(scores, bins=30, alpha=0.5, label=pos, edgecolor='black')
    
    ax.set_xlabel('Compression Score')
    ax.set_ylabel('Count')
    ax.set_title('Distribution by POS (Top 5 Categories)')
    ax.legend()
    
    plt.tight_layout()
    return fig
```

### Phase 7: Integration and Main Experiment Script (Days 12-14)

**Google Colab notebook structure:**

```python
# notebooks/main_experiment.ipynb

"""
COMPRESSION BASINS IN LANGUAGE MODELS
Complete experimental pipeline for Google Colab
"""

# ===========================
# SETUP AND INSTALLATION
# ===========================

# Install dependencies
!pip install -q transformers torch numpy scipy scikit-learn matplotlib seaborn tqdm nltk

# Download NLTK data
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')

# Clone your repository
!git clone https://github.com/yourusername/compression_lm.git
%cd compression_lm

# Import all modules
from models.model_loader import *
from models.extract_states import *
from compression.metric import *
from experiments.memorization import *
from experiments.token_importance import *
from experiments.linguistic import *
from analysis.visualizations import *

import torch
import numpy as np
import matplotlib.pyplot as plt

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# ===========================
# LOAD MODEL
# ===========================

print("\nLoading GPT-2 model...")
model = GPT2LMHeadModel.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model.eval()
model = model.to(device)
print("Model loaded successfully!")

# ===========================
# PREPARE DATA
# ===========================

print("\nPreparing datasets...")

# Option 1: Load from Hugging Face datasets
from datasets import load_dataset

# Load WikiText-103
dataset = load_dataset('wikitext', 'wikitext-103-v1', split='test')
texts = [text for text in dataset['text'] if len(text.strip()) > 100]
texts = texts[:1000]  # Limit for speed

print(f"Loaded {len(texts)} text samples")
print(f"Average length: {np.mean([len(t.split()) for t in texts]):.1f} words")

# ===========================
# EXPERIMENT 1: MEMORIZATION
# ===========================

print("\n" + "="*70)
print("EXPERIMENT 1: MEMORIZATION")
print("="*70)

memorization_results = run_memorization_experiment(
    model=model,
    tokenizer=tokenizer,
    texts=texts,
    k_neighbors=15,
    max_sequences=500
)

# Save results
import pickle
with open('memorization_results.pkl', 'wb') as f:
    pickle.dump(memorization_results, f)

print("\nMemorization experiment complete!")

# ===========================
# EXPERIMENT 2: TOKEN IMPORTANCE
# ===========================

print("\n" + "="*70)
print("EXPERIMENT 2: TOKEN IMPORTANCE")
print("="*70)

# Extract attention patterns for subset
importance_texts = texts[:100]

print("\nExtracting attention patterns...")
attention_importance_scores = []

for text in tqdm(importance_texts, desc="Processing"):
    attention_weights, tokens = extract_attention_weights(model, tokenizer, text)
    importance = compute_token_importance_attention(attention_weights)
    attention_importance_scores.extend(importance)

print(f"Extracted importance scores for {len(attention_importance_scores)} tokens")

# Extract hidden states and compute compression
print("\nExtracting hidden states...")
all_states, all_tokens, metadata = extract_dataset_states(
    model, tokenizer, importance_texts, max_length=128
)

print("\nComputing compression...")
layer_compression, layer_metadata = compute_all_layers_compression(
    all_states, k=15, use_faiss=True
)

# Analyze correlation
print("\nAnalyzing importance-compression correlation...")
importance_results = {}

for layer_idx in range(len(layer_compression)):
    results = analyze_importance_compression(
        layer_compression[layer_idx],
        np.array(attention_importance_scores[:len(layer_compression[layer_idx])]),
        layer_metadata[layer_idx]['sequence_indices'],
        layer_idx=layer_idx
    )
    importance_results[layer_idx] = results
    
    # Visualize best layer
    if abs(results['pearson_r']) > 0.3:
        fig = visualize_importance_results(results)
        plt.savefig(f'importance_layer_{layer_idx}.png', dpi=150)
        plt.show()

# Save results
with open('importance_results.pkl', 'wb') as f:
    pickle.dump(importance_results, f)

print("\nToken importance experiment complete!")

# ===========================
# EXPERIMENT 3: LINGUISTIC STRUCTURE
# ===========================

print("\n" + "="*70)
print("EXPERIMENT 3: LINGUISTIC STRUCTURE")
print("="*70)

# Add POS tags
linguistic_texts = texts[:200]

print("\nAdding POS tags...")
all_states, all_tokens, metadata = extract_dataset_states(
    model, tokenizer, linguistic_texts, max_length=128
)

pos_tags_list = add_pos_tags(linguistic_texts, all_tokens)

print("\nComputing compression...")
layer_compression, layer_metadata = compute_all_layers_compression(
    all_states, k=15, use_faiss=True
)

# Analyze POS patterns
print("\nAnalyzing POS-compression patterns...")
linguistic_results = {}

for layer_idx in range(len(layer_compression)):
    results = analyze_pos_compression(
        layer_compression[layer_idx],
        pos_tags_list,
        layer_metadata[layer_idx]['sequence_indices'],
        layer_idx=layer_idx
    )
    linguistic_results[layer_idx] = results
    
    # Visualize interesting layers
    if layer_idx % 3 == 0:  # Every 3rd layer
        fig = visualize_pos_results(results)
        plt.savefig(f'pos_layer_{layer_idx}.png', dpi=150)
        plt.show()

# Save results
with open('linguistic_results.pkl', 'wb') as f:
    pickle.dump(linguistic_results, f)

print("\nLinguistic structure experiment complete!")

# ===========================
# COMPREHENSIVE ANALYSIS
# ===========================

print("\n" + "="*70)
print("COMPREHENSIVE CROSS-EXPERIMENT ANALYSIS")
print("="*70)

# Layer-wise summary
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# Memorization correlations
ax = axes[0]
layers = list(memorization_results.keys())
mem_corrs = [memorization_results[l]['correlation'] for l in layers]
ax.plot(layers, mem_corrs, 'o-', linewidth=2, markersize=8, label='Memorization')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(y=0.4, color='green', linestyle='--', alpha=0.5)
ax.axhline(y=-0.4, color='green', linestyle='--', alpha=0.5)
ax.set_ylabel('Correlation')
ax.set_title('Memorization-Compression Correlation Across Layers')
ax.legend()
ax.grid(True, alpha=0.3)

# Token importance correlations
ax = axes[1]
importance_corrs = [importance_results[l]['pearson_r'] for l in layers if l in importance_results]
ax.plot(layers[:len(importance_corrs)], importance_corrs, 'o-', linewidth=2, markersize=8, 
       label='Token Importance', color='orange')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax.axhline(y=0.4, color='green', linestyle='--', alpha=0.5)
ax.axhline(y=-0.4, color='green', linestyle='--', alpha=0.5)
ax.set_ylabel('Correlation')
ax.set_title('Importance-Compression Correlation Across Layers')
ax.legend()
ax.grid(True, alpha=0.3)

# POS variance (ANOVA F-statistic)
ax = axes[2]
anova_fs = [linguistic_results[l].get('anova_f', 0) for l in layers if l in linguistic_results]
ax.plot(layers[:len(anova_fs)], anova_fs, 'o-', linewidth=2, markersize=8,
       label='POS Variance', color='purple')
ax.set_xlabel('Layer')
ax.set_ylabel('ANOVA F-statistic')
ax.set_title('Linguistic Structure (POS) Variance Across Layers')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('comprehensive_analysis.png', dpi=150, bbox_inches='tight')
plt.show()

# Print summary table
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

print("\nMemorization:")
best_mem_layer = max(memorization_results.keys(), 
                    key=lambda l: abs(memorization_results[l]['correlation']))
print(f"  Best layer: {best_mem_layer}")
print(f"  Correlation: {memorization_results[best_mem_layer]['correlation']:.4f}")
print(f"  P-value: {memorization_results[best_mem_layer]['correlation_p']:.2e}")

print("\nToken Importance:")
best_imp_layer = max(importance_results.keys(),
                    key=lambda l: abs(importance_results[l]['pearson_r']))
print(f"  Best layer: {best_imp_layer}")
print(f"  Correlation: {importance_results[best_imp_layer]['pearson_r']:.4f}")
print(f"  P-value: {importance_results[best_imp_layer]['pearson_p']:.2e}")

print("\nLinguistic Structure:")
print(f"  Significant POS differences found in {sum(1 for l in linguistic_results if linguistic_results[l].get('anova_p', 1) < 0.01)} / {len(linguistic_results)} layers")

# Overall assessment
print("\n" + "="*70)
print("OVERALL ASSESSMENT")
print("="*70)

strong_findings = []
if abs(memorization_results[best_mem_layer]['correlation']) > 0.4:
    strong_findings.append("Memorization")
if abs(importance_results[best_imp_layer]['pearson_r']) > 0.4:
    strong_findings.append("Token Importance")

if len(strong_findings) > 0:
    print(f"✓ STRONG FINDINGS: {', '.join(strong_findings)}")
    print("  Recommendation: Write full paper targeting top venue (NeurIPS/ICML/ICLR)")
elif len(strong_findings) == 0 and (abs(memorization_results[best_mem_layer]['correlation']) > 0.2 or
                                     abs(importance_results[best_imp_layer]['pearson_r']) > 0.2):
    print("~ MODERATE FINDINGS: Some correlations but below strong threshold")
    print("  Recommendation: Target ACL/EMNLP or solid methods paper (TMLR/AISTATS)")
else:
    print("✗ WEAK FINDINGS: No strong correlations found")
    print("  Recommendation: Pivot to different hypothesis or different model")

print("\n" + "="*70)
print("EXPERIMENTS COMPLETE!")
print("="*70)
```

## Success Criteria and Evaluation

### Quantitative Thresholds

**Strong Success (Target: Top Venue)**
- Correlation |r| > 0.4 with p < 0.01 for at least one hypothesis
- Effect replicates across at least 2 model sizes (GPT-2 small and medium)
- Pattern is interpretable and aligns with theoretical expectations

**Moderate Success (Target: Good Venue)**
- Correlation |r| > 0.2 with p < 0.05 for multiple hypotheses
- Clear layer-wise structure or linguistic patterns
- Results suggest interesting phenomena worth deeper investigation

**Exploratory Success (Target: Methods Paper)**
- Reliable measurement of compression in LLMs demonstrated
- Interesting qualitative patterns even without strong correlations
- Clear characterization of when/where compression occurs

### Qualitative Indicators

**Paper-worthy findings include:**
- Compression patterns that differ across layers in interpretable ways
- Clear separation between memorized and novel content
- POS categories showing systematic compression differences
- Connections to existing interpretability tools (attention, probing)

**Red flags that suggest pivoting:**
- Compression scores are too noisy to show patterns
- No correlation survives multiple comparison correction
- Patterns don't replicate across different text samples
- Results are entirely explained by simple confounds (frequency, position)

## Timeline and Milestones

**Week 1:**
- Days 1-2: Setup infrastructure, test on toy examples
- Days 3-5: Implement compression metric, verify correctness
- Days 5-7: Run initial memorization experiments

**Week 2:**
- Days 8-10: Token importance experiments
- Days 10-12: Linguistic structure experiments
- Days 12-14: Comprehensive analysis and visualization

**Week 3:**
- Days 15-17: Write up results section
- Days 18-19: Create all final figures
- Days 20-21: Draft introduction and related work

**Week 4:**
- Days 22-25: Complete draft
- Days 26-28: Internal review and revision

**Total: 4 weeks** to complete draft if working efficiently

## Common Pitfalls and Solutions

**Problem: Compression scores are all very similar (low variance)**
- Solution: Adjust k (number of neighbors), try different layers, normalize hidden states

**Problem: Results don't replicate across text samples**
- Solution: Increase sample size, check for batch effects, verify preprocessing

**Problem: Correlations are weak (<0.2)**
- Solution: Don't force it - pivot to qualitative analysis or try different hypotheses

**Problem: Computational resources insufficient**
- Solution: Use smaller model (GPT-2 small), reduce sample size, use approximate k-NN

**Problem: Can't detect memorization reliably**
- Solution: Use controlled fine-tuning approach or existing memorization benchmarks

## Final Deliverables

At the end of the experiment, you should have:

1. **Code repository** with all implementations
2. **Experimental results** saved as pickle files
3. **Visualizations** for each experiment (10-15 figures)
4. **Statistical summaries** documenting all correlations and tests
5. **Written analysis** interpreting the findings
6. **Decision on publication venue** based on strength of results

This forms the foundation for writing the full paper targeting your chosen venue.