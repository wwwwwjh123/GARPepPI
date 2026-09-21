import numpy as np

# === Version & training ===
version = '1.0'
batchsize = 64
trainArgs = {
    'lr': 0.0002,
    'epochs': 50,
    'doTest': True,
    'doSave': True,
    'grad_clip': 1.0,
    'early_stopping': False,
    'patience': 10,
    'min_delta': 0.0001,
    'monitor': 'auc',
}

# === Output paths ===
rst_file        = 'path/to/results/result.tsv'
pkl_path        = 'path/to/model_pkl/model_{}'.format(version)
test_5fold_path = 'path/to/results/train_5fold_cross_validation/'

# === Model architecture ===
modelArgs = {
    'emb_dim': 1024,      # must match embedding dimension
    'output_dim': 128,
    'dense_hid': 64,
    'task_type': 0,       # 0: classification, 1: regression
    'n_classes': 1,
}

# === Data paths (modify these for your dataset) ===
# Replace with actual paths to your data

actions_file = "path/to/actions.tsv"
cmaproot      = "path/to/contact_map/"
embed_data    = np.load("path/to/embeddings.npz", allow_pickle=True)
