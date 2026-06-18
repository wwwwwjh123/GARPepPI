# GARPepPI

**Transformer-based Protein-Peptide Interaction Prediction**

Predicts protein-peptide interaction probability using ProtT5 embeddings, contact maps, graph attention (TAGConv), and TextCNN.

---

## Directory Structure

```
GARPepPI/
├── 1_database/               # Raw data for each dataset
│   ├── Human/
│   ├── Propedia-PPB/
│   ├── SARS-CoV-2-human/
│   └── Yeast/
├── 2_preprocessing/          # Data preprocessing scripts
│   ├── data_process_txt.py
│   ├── generate_embeddings.py
│   └── generate_contact_map.py
├── 3_model/                  # Model code
│   ├── args.py               # Hyperparameters (modify paths here)
│   ├── data_loader.py        # Contact map → DGL graph
│   ├── GARPepPI.py           # Model definition
│   ├── main.py               # Entry point
│   ├── seq2tensor.py         # Embedding dictionary
│   ├── train_and_validation.py
│   └── utils.py
└── 4_results/                # Output (created at runtime)
```

---

## Setup

### 1. Download ProtT5 Embedding Model

Download from HuggingFace:

```
https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc/tree/main
```

Download all files (`config.json`, `pytorch_model.bin`, `special_tokens_map.json`, `spiece.model`, `tokenizer_config.json`) into:

```
2_preprocessing/prot_t5_xl_half_uniref50-enc/
```

### 2. Install Dependencies

```bash
pip install torch dgl numpy pandas scipy openpyxl selfies xlwt scikit-learn
```

> DGL installation varies by CUDA version. Refer to https://www.dgl.ai/pages/start.html

---

## Data Preparation

Each dataset in `1_database/` contains two files:

| File | Format | Content |
|---|---|---|
| `{name}.dictionary.tsv` | `ID\tsequence` | Protein/peptide sequences |
| `{name}.actions.{N}-{N}.tsv` | `ID1\tID2\tlabel` | Interaction pairs |

### Step 1 — Generate Embeddings

```python
# generate_embeddings.py
python generate_embeddings.py
```
- Requires: `1_database/{name}.dictionary.tsv`
- Output: `embeddings/{name}_embeddings.npz`
- Loads ProtT5 from `2_preprocessing/prot_t5_xl_half_uniref50-enc/`

### Step 2 — Generate Contact Maps

```python
# generate_contact_map.py
python generate_contact_map.py
```
- Requires: `1_database/{name}.dictionary.tsv`
- Output: `contact_map/{name}_contact_map/*.npz`

---

## Training

### 1. Configure Paths

Edit `3_model/args.py`:

```python
# === Output paths ===
rst_file        = '/path/to/results/result.tsv'
pkl_path        = '/path/to/model_pkl/model_1.0'
test_5fold_path = '/path/to/results/train_5fold_cross_validation/'

# === Data paths ===
actions_file = '/path/to/{name}.actions.{N}-{N}.tsv'
cmaproot      = '/path/to/contact_map/{name}_contact_map/'
embed_data    = np.load("/path/to/{name}_embeddings.npz", allow_pickle=True)
```

### 2. Run Training

```bash
cd 3_model
python main.py
```

Training performs 5-fold stratified cross-validation. Best model per fold (highest validation accuracy) is saved to `model_pkl/`. Test results per fold are saved as `.xls` files.

---

## Model Architecture

**GARPepPI** combines two modalities:

1. **Graph branch**: Contact map → DGL graph → TAGConv (k=2) → MaxPooling
2. **Sequence branch**: ProtT5 embedding → 3-layer 1D CNN (TextCNN)

Features are fused via a learnable weighted sum + gated interaction + residual connection, then passed through a 3-layer MLP for binary classification.

---

## Citation

If this work is useful, please cite:

```
TODO: add your paper citation here
```
