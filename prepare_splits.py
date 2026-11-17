import os
import numpy as np
from glob import glob
import random

BASE = "/Users/rumeysa/Desktop/2.5D-Unet"
CT_DIR   = f"{BASE}/NLST_2D_BALANCED/CT"
MASK_DIR = f"{BASE}/NLST_2D_BALANCED/Mask"
SPLIT    = f"{BASE}/splits"

os.makedirs(SPLIT, exist_ok=True)

tumor_patients = []
normal_patients = []

patients = sorted(os.listdir(CT_DIR))

for pid in patients:
    mdir = os.path.join(MASK_DIR, pid)
    if not os.path.isdir(mdir):
        continue

    masks = glob(f"{mdir}/*.npy")
    has_tumor = False

    for m in masks:
        if np.load(m).sum() > 0:
            has_tumor = True
            break

    if has_tumor:
        tumor_patients.append(pid)
    else:
        normal_patients.append(pid)

print("Tumor patients:", len(tumor_patients))
print("Normal patients:", len(normal_patients))

random.seed(42)
random.shuffle(tumor_patients)
random.shuffle(normal_patients)


n_val_tumor = max(1, int(0.2 * len(tumor_patients)))
n_val_normal = int(0.2 * len(normal_patients))

val_patients = (
    tumor_patients[:n_val_tumor] +
    normal_patients[:n_val_normal]
)

train_patients = (
    tumor_patients[n_val_tumor:] +
    normal_patients[n_val_normal:]
)

with open(f"{SPLIT}/train.txt", "w") as f:
    f.write("\n".join(train_patients))

with open(f"{SPLIT}/val.txt", "w") as f:
    f.write("\n".join(val_patients))

print("Train patients:", len(train_patients))
print("Val patients:", len(val_patients))
