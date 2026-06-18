# -*- coding: utf-8 -*-
"""
Generate ProtT5-XL-UniRef50 embeddings for protein/peptide sequences.

Usage:
    python generate_embeddings.py --fasta path/to/input.fasta
"""
import torch
from transformers import T5EncoderModel, T5Tokenizer
import re
import numpy as np
from tqdm import tqdm
import argparse
import io
from Bio import SeqIO
import gc
import zipfile
import os

parser = argparse.ArgumentParser(description='Generate ProtT5 embeddings for protein/peptide sequences')
parser.add_argument('--fasta', help='Path to input FASTA file')
args = parser.parse_args()

id_file = "path/to/protein_ID.txt"
fasta_file = "path/to/input.fasta"
embedding_file = "path/to/embeddings.npz"
model_dir = "path/to/prot_t5_xl_half_uniref50-enc/"

# Load valid protein IDs
valid_ids = set()
for line in open(id_file):
    for pid in line.strip().split():
        if pid:
            valid_ids.add(pid)
print(f"Loaded {len(valid_ids)} valid protein IDs")

# Parse FASTA
input_str = open(fasta_file).read()
if not input_str.strip().startswith(">") and input_str.strip():
    input_str = "> unnamed_protein\n" + input_str

buf = io.StringIO(input_str)
sequences = []
sequence_ids = []
for record in SeqIO.parse(buf, "fasta"):
    if record.id in valid_ids:
        sequences.append(str(record.seq))
        sequence_ids.append(record.id)

fasta_len = len(sequences)
print(f"Sequences loaded (after filtering): {fasta_len}")

# Replace non-standard amino acids and format for ProtT5
processed = [re.sub(r"[UZOB]", "X", seq) for seq in sequences]
processed = [' '.join(list(seq)) for seq in processed]

def chunks(lst, n):
    """Split list into chunks of size n."""
    return [lst[i:i + n] for i in range(0, len(lst), n)]

proc_seq_chunks = chunks(processed, 100)

# Load model
print("Loading ProtT5 model...")
tokenizer = T5Tokenizer.from_pretrained(model_dir, do_lower_case=False)
model = T5EncoderModel.from_pretrained(model_dir)
gc.collect()

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.device_count()} device(s), using {device}")

model = model.to(device).eval()

seq_idx = 0
with zipfile.ZipFile(embedding_file, mode="a",
                     compression=zipfile.ZIP_DEFLATED,
                     allowZip64=True) as zf:
    for chunk_idx, chunk in enumerate(tqdm(proc_seq_chunks, desc="Batches"), 1):
        with torch.no_grad():
            for proc_seq in chunk:
                seq_id = sequence_ids[seq_idx]
                npy_name = f"{seq_id}.npy"
                original_len = len(proc_seq.replace(' ', ''))

                # Check if already processed
                skip = False
                if npy_name in zf.namelist():
                    try:
                        with zf.open(npy_name, 'r') as buf_in:
                            existing = np.load(buf_in, allow_pickle=False)
                        if existing.shape[0] == original_len:
                            print(f"Skipped (exists, length match): {seq_id}")
                            seq_idx += 1
                            continue
                        else:
                            print(f"Regenerate (length mismatch): {seq_id}")
                    except Exception as e:
                        print(f"Error reading existing embedding for {seq_id}: {e}")

                # Tokenize
                ids = tokenizer.batch_encode_plus(
                    [proc_seq], add_special_tokens=True, padding="longest"
                )
                input_ids = torch.tensor(ids['input_ids']).to(device)
                attention_mask = torch.tensor(ids['attention_mask']).to(device)

                # Generate embeddings
                emb = model(input_ids=input_ids, attention_mask=attention_mask)
                emb = emb.last_hidden_state.cpu().numpy()

                for seq_num in range(len(emb)):
                    seq_len = (attention_mask[seq_num] == 1).sum()
                    seq_emd = emb[seq_num][:seq_len - 1]
                    emb_len = seq_emd.shape[0]
                    if emb_len != original_len:
                        print(f"Warning: embedding length {emb_len} != original {original_len} for {seq_id}")
                    with zf.open(f"{seq_id}.npy", 'w', force_zip64=True) as buf_out:
                        np.lib.npyio.format.write_array(
                            buf_out, np.asanyarray(seq_emd), allow_pickle=False
                        )

                seq_idx += 1

        print(f"Batch {chunk_idx}/{len(proc_seq_chunks)} completed")

print(f"\nDone. Processed {fasta_len} sequences. Embeddings saved to: {embedding_file}")
gc.collect()
