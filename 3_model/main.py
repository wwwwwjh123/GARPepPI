# -*- coding: utf-8 -*-
"""Entry point: check output directories and start training."""
import os
import shutil
import torch
import warnings
warnings.filterwarnings("ignore")
torch.set_default_tensor_type('torch.cuda.FloatTensor')

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from args import *
from train_and_validation import *


def _prepare_dir(dir_path, desc):
    if not dir_path:
        dir_path = os.getcwd()
    dir_path = os.path.abspath(dir_path)
    if os.path.exists(dir_path):
        choice = input(f"[{desc}] already exists: {dir_path}, delete and recreate? [y/N]: ").strip().lower()
        if choice in ("y", "yes"):
            shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)
            print(f"Recreated {desc}: {dir_path}")
        else:
            print(f"Kept existing {desc}: {dir_path}")
    else:
        os.makedirs(dir_path, exist_ok=True)
        print(f"Created {desc}: {dir_path}")
    return dir_path


def check_and_create_dirs():
    print("=" * 60)
    print("Checking output directories...")
    print("=" * 60)
    _prepare_dir(os.path.dirname(os.path.abspath(rst_file)), "result file directory")
    _prepare_dir(os.path.dirname(os.path.abspath(pkl_path)), "model checkpoint directory")
    _prepare_dir(test_5fold_path, "5-fold test result directory")
    for fold_num in range(1, 6):
        os.makedirs(os.path.join(test_5fold_path, f'fold{fold_num}'), exist_ok=True)
    print("=" * 60)
    print("Ready to train")
    print("=" * 60)


if __name__ == "__main__":
    check_and_create_dirs()
    train(trainArgs)
