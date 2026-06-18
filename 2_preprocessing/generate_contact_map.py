# -*- coding: utf-8 -*-
"""
Generate protein/peptide contact maps from PDB files.

Each line in the three input files corresponds to the same protein/peptide:
  - protein_ID.txt: protein/peptide ID
  - protein_sequences.txt: amino acid sequence
  - protein_chain_id.txt: chain ID in the complex
"""
import Bio.PDB
import numpy as np
import os
from tqdm import tqdm

id_file = "path/to/protein_ID.txt"
seq_file = "path/to/protein_sequences.txt"
chain_file = "path/to/protein_chain_id.txt"
pdb_dir = "path/to/pdb/"
contact_map_dir = "path/to/contact_map/"

# Three-letter to one-letter amino acid code mapping
aa_codes = {
    'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
    'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'LYS': 'K',
    'ILE': 'I', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
    'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER': 'S',
    'THR': 'T', 'VAL': 'V', 'TYR': 'Y', 'TRP': 'W',
}

DISTANCE_THRESHOLD = 8.0  # Angstroms


def get_center_atom(residue):
    """Return the name of the preferred center atom for distance computation."""
    for atom in ('CA', 'N', 'C', 'O', 'CB', 'CD', 'CG'):
        if residue.has_id(atom):
            return atom
    return 'CA'


def calc_residue_dist(res1, res2):
    """Compute Euclidean distance between the center atoms of two residues."""
    a1 = get_center_atom(res1)
    a2 = get_center_atom(res2)
    diff = res1[a1].coord - res2[a2].coord
    return np.sqrt(np.sum(diff * diff))


def calc_dist_matrix(chain1, chain2):
    """
    Compute residue distance matrix between two chains.
    Only standard amino acids (hetfield=' ' and in aa_codes) are counted.
    Diagonal is set to 100 to avoid self-contact.
    """
    residues1 = [r for r in chain1
                 if r.get_id()[0] == ' ' and r.get_resname() in aa_codes]
    residues2 = [r for r in chain2
                 if r.get_id()[0] == ' ' and r.get_resname() in aa_codes]
    n1, n2 = len(residues1), len(residues2)
    dist_mat = np.zeros((n1, n2), np.float_)

    for i, r1 in enumerate(residues1):
        for j, r2 in enumerate(residues2):
            dist_mat[i, j] = calc_residue_dist(r1, r2)

    k = min(n1, n2)
    for i in range(k):
        dist_mat[i, i] = 100.0
    return dist_mat


def calc_contact_map(pdb_id, chain_id):
    """
    Generate binary contact map for a given PDB entry and chain.
    A residue pair is considered in contact if their center-atom distance < 8.0 A.
    Returns None if the file or chain is not found.
    """
    pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        print(f"Warning: PDB file not found: {pdb_path}")
        return None

    try:
        structure = Bio.PDB.PDBParser().get_structure(pdb_id, pdb_path)
        model = structure[0]
        available_chains = list(model.child_dict.keys())

        if chain_id not in available_chains:
            matched = None
            for cid in available_chains:
                if str(cid).lower() == chain_id.lower():
                    matched = cid
                    break
            if matched is None:
                print(f"Error: chain '{chain_id}' not found in PDB {pdb_id}. Available: {available_chains}")
                return None
            print(f"Note: chain '{chain_id}' not found, using '{matched}' instead")
            chain_id = matched

        dist_mat = calc_dist_matrix(model[chain_id], model[chain_id])
        contact_map = (dist_mat < DISTANCE_THRESHOLD).astype(np.int_)
        return contact_map
    except Exception as e:
        print(f"Error processing PDB {pdb_id} chain {chain_id}: {e}")
        return None


# Load input files
ids = []
for line in open(id_file):
    ids.extend(line.strip().split())
seqs = []
for line in open(seq_file):
    seqs.extend(line.strip().split())
chains = []
for line in open(chain_file):
    chains.extend(line.strip().split())

n = len(ids)
print(f"Loaded {n} proteins/peptides")

if len(seqs) != n or len(chains) != n:
    print(f"Warning: file length mismatch - IDs:{n} seqs:{len(seqs)} chains:{len(chains)}")
    n = min(n, len(seqs), len(chains))
    print(f"Processing only first {n} entries")

os.makedirs(contact_map_dir, exist_ok=True)

total_count = n
success_count = 0
failure_count = 0
skip_existing_count = 0

print(f"Starting contact map generation for {n} entries...")
for idx in tqdm(range(n), desc="Generating contact maps"):
    contact_file = os.path.join(contact_map_dir, f"{ids[idx]}.npz")

    if os.path.exists(contact_file):
        skip_existing_count += 1
        continue

    contact_map = calc_contact_map(ids[idx], str(chains[idx]))
    if contact_map is None:
        failure_count += 1
        continue

    try:
        np.savez(contact_file, seq=seqs[idx], contact=contact_map)
        success_count += 1
    except Exception as e:
        print(f"Error saving {contact_file}: {e}")
        failure_count += 1

print("\n" + "=" * 60)
print("Summary")
print("=" * 60)
print(f"Total:       {total_count}")
print(f"Success:     {success_count}")
print(f"Failed:      {failure_count}")
print(f"Skipped:     {skip_existing_count}")
print("=" * 60)
