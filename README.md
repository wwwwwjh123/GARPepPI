# GARPepPI

**A dual-branch multimodal collaborative learning framework for PepPI prediction**

---

<p align="center">
  <img src="assets/GARPepPI.svg" alt="GARPepPI Architecture" width="700"/>
  <br>
  <em>Figure 1. Overall architecture of GARPepPI.</em>
</p>

---

## Installation

### Requirements

| Dependency | Version |
|---|---|
| Python | 3.7+ |
| PyTorch | ≥ 1.5.1 |
| DGL | ≥ 0.6.1 |
| CUDA | 10.1+ |
| scikit-learn | latest |

### Install Commands

```bash
# PyTorch (CUDA 10.1)
conda install pytorch==1.5.1 torchvision==0.6.1 cudatoolkit=10.1 -c pytorch

# DGL for CUDA 10.1
pip install dgl_cu101 -f https://www.dgl.ai/pages/start.html

# Other dependencies
pip install numpy pandas scipy openpyxl xlwt selfies scikit-learn transformers
```

---

## Data Preparation

### Required Input Files

Prepare the following three types of files:

1. **`actions_file`** — interaction pairs in TSV format:
   ```
   protein_id\tpeptide_id\tlabel
   ```
   `label`: `1` = interacting, `0` = non-interacting.

2. **PDB files** — one `.pdb` file per protein and peptide, containing the 3D structure.

3. **FASTA file** — all protein and peptide sequences in one file:
   ```
   >protein_id_1
   MKTVRQERLKSIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTK...
   >protein_id_2
   AXXX
   >peptide_id_1
   SIQAEEWYFGKITRRESERLLLNAENPRGTFLVRESETTKGAYCLSVSDFDNAKG...
   ```

### Step 1 — Generate ProtT5 Embeddings

Download the ProtT5 model from HuggingFace:

```
https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc/tree/main
```

Download all files (`config.json`, `pytorch_model.bin`, `special_tokens_map.json`, `spiece.model`, `tokenizer_config.json`) into `2_preprocessing/prot_t5_xl_half_uniref50-enc/`.

Edit the FASTA path in `2_preprocessing/generate_embeddings.py` and run:

```bash
python 2_preprocessing/generate_embeddings.py
```

Output: `embeddings/{name}_embeddings.npz`

### Step 2 — Generate Contact Maps

Edit the PDB file directory path in `2_preprocessing/generate_contact_map.py` and run:

```bash
python 2_preprocessing/generate_contact_map.py
```

Output: `contact_map/{name}_contact_map/*.npz`, one `.npz` file per protein/peptide.

---

## Training

### 1. Configure Paths

Edit `3_model/args.py` with your actual paths:

```python
# === Output ===
rst_file        = '/path/to/results/result.tsv'
pkl_path        = '/path/to/model_pkl/model_1.0'
test_5fold_path = '/path/to/results/train_5fold/'

# === Data ===
actions_file = '/path/to/{name}.actions.{N}-{N}.tsv'
cmaproot     = '/path/to/contact_map/{name}_contact_map/'
embed_data   = np.load("/path/to/{name}_embeddings.npz", allow_pickle=True)
```

### 2. Run 5-Fold Cross-Validation

```bash
cd 3_model
python main.py
```

The best model for each fold (highest validation accuracy) is saved to `model_pkl/`. Per-fold test predictions are exported as `.xls` files.

---

## Input & Output

### Output Format

Per-fold test predictions are exported as `.xls` files with the following columns:

| Field | Description |
|---|---|
| index | Sample index |
| receptor | Protein ID |
| peptide | Peptide ID |
| label | Ground truth |
| predict_score | Interaction probability (0–1) |
| predict_label | Prediction (`1` if score ≥ 0.5) |

---
