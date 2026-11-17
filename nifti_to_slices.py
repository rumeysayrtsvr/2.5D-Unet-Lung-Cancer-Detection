import os
import nibabel as nib
import numpy as np
from tqdm import tqdm

RAW_ROOT = "NLSTseg_raw"
OUT_ROOT = "NLST_2D"

HU_MIN = -500
HU_MAX = 1400

ct_out_root = os.path.join(OUT_ROOT, "CT")
mask_out_root = os.path.join(OUT_ROOT, "Mask")

os.makedirs(ct_out_root, exist_ok=True)
os.makedirs(mask_out_root, exist_ok=True)

patient_ids = sorted(os.listdir(RAW_ROOT))

for pid in tqdm(patient_ids):
    patient_dir = os.path.join(RAW_ROOT, pid)
    if not os.path.isdir(patient_dir):
        continue

    ct_path = os.path.join(patient_dir, f"{pid}_CT.nii.gz")
    mask_path = os.path.join(patient_dir, f"{pid}_tumor.nii.gz")

    if not os.path.exists(ct_path) or not os.path.exists(mask_path):
        print(f"⚠️ Eksik dosya: {pid}")
        continue

    ct = nib.load(ct_path).get_fdata()
    mask = nib.load(mask_path).get_fdata()


    min_slices = min(ct.shape[2], mask.shape[2])
    ct = ct[:, :, :min_slices]
    mask = mask[:, :, :min_slices]

    ct_out_dir = os.path.join(ct_out_root, pid)
    mask_out_dir = os.path.join(mask_out_root, pid)

    os.makedirs(ct_out_dir, exist_ok=True)
    os.makedirs(mask_out_dir, exist_ok=True)

    for i in range(min_slices):
        ct_slice = ct[:, :, i]
        mask_slice = mask[:, :, i]

        ct_slice = np.clip(ct_slice, HU_MIN, HU_MAX)
        ct_slice = (ct_slice - HU_MIN) / (HU_MAX - HU_MIN)
        ct_slice = ct_slice.astype(np.float32)

        mask_slice = (mask_slice > 0).astype(np.uint8)

        np.save(os.path.join(ct_out_dir, f"slice_{i:03d}.npy"), ct_slice)
        np.save(os.path.join(mask_out_dir, f"slice_{i:03d}.npy"), mask_slice)

print("✅ Shape mismatch problemi otomatik çözüldü.")
