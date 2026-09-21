import pandas as pd
import os

input_file = "path/to/dictionary.xlsx"        # .xlsx (ID, sequence) or .tsv (ID only)
output_file1 = "path/to/protein_ID.txt"
output_file2 = "path/to/protein_sequences.txt"
output_file3 = "path/to/protein_chain_id.txt"

if not os.path.exists(input_file):
    print(f"Error: file not found: {input_file}")
    exit(1)

ext = os.path.splitext(input_file)[1].lower()
if ext == '.xlsx':
    df = pd.read_excel(input_file, header=None)
    if len(df.columns) < 2:
        print("Error: Excel file must contain at least two columns")
        exit(1)
    column1 = df.iloc[:, 0].astype(str)
    column2 = df.iloc[:, 1].astype(str)
elif ext == '.tsv':
    column1 = []
    column2 = None
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                column1.extend(line.split())
    column1 = pd.Series(column1)
else:
    print(f"Error: unsupported file format '{ext}'. Only .xlsx and .tsv are supported.")
    exit(1)

with open(output_file1, 'w') as f:
    f.write(' '.join(column1))
print(f"Saved column 1 to {output_file1}")

if column2 is not None:
    with open(output_file2, 'w') as f:
        f.write(' '.join(column2))
    print(f"Saved column 2 to {output_file2}")

print(f"Number of proteins: {len(column1)}")
if column2 is not None:
    print(f"Number of peptides: {len(column2)}")

with open(output_file1, 'r') as f:
    protein_ids = f.read().strip().split()
    output_count1 = len(protein_ids)

if column2 is not None:
    with open(output_file2, 'r') as f:
        protein_sequences = f.read().strip().split()
        output_count2 = len(protein_sequences)

    if output_count1 == len(column1):
        print(f"Verification passed: protein ID count {output_count1} matches original")
    else:
        print(f"Warning: protein ID count {output_count1} does not match original ({len(column1)})")

    if output_count2 == len(column2):
        print(f"Verification passed: protein sequence count {output_count2} matches original")
    else:
        print(f"Warning: protein sequence count {output_count2} does not match original ({len(column2)})")

protein_chain_ids = []
for protein_id in column1:
    chain_id = protein_id[-1] if protein_id else ""
    protein_chain_ids.append(chain_id)

with open(output_file3, 'w') as f:
    for chain_id in protein_chain_ids:
        f.write(f"{chain_id} ")
print(f"Saved chain IDs to {output_file3}")

with open(output_file3, 'r') as f:
    chain_ids = f.read().strip().split()
    output_count3 = len(chain_ids)

if output_count3 == len(column1):
    print(f"Verification passed: chain ID count {output_count3} matches original")
else:
    print(f"Warning: chain ID count {output_count3} does not match original ({len(column1)})")
