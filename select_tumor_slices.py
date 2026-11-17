import os
import numpy as np
import shutil
from tqdm import tqdm

SRC_ROOT = "/Users/rumeysa/Desktop/2.5D-Unet/NLST_2D"        
DST_ROOT = "/Users/rumeysa/Desktop/2.5D-Unet/NLST_2D_TUMOR_ONLY"

CT_SRC  = os.path.join(SRC_ROOT, "CT")
MSK_SRC = os.path.join(SRC_ROOT, "Mask")

CT_DST  = os.path.join(DST_ROOT, "CT")
MSK_DST = os.path.join(DST_ROOT, "Mask")

os.makedirs(CT_DST, exist_ok=True)
os.makedirs(MSK_DST, exist_ok=True)


kept = 0
removed = 0

patients = sorted(os.listdir(CT_SRC))

for pid in tqdm(patients, desc="Filtering patients"):
    ct_p  = os.path.join(CT_SRC, pid)
    msk_p = os.path.join(MSK_SRC, pid)

    if not os.path.isdir(ct_p):
        continue

    out_ct_p  = os.path.join(CT_DST, pid)
    out_msk_p = os.path.join(MSK_DST, pid)

    os.makedirs(out_ct_p, exist_ok=True)
    os.makedirs(out_msk_p, exist_ok=True)

    slices = sorted([f for f in os.listdir(ct_p) if f.endswith(".npy")])

    for sl in slices:
        mask_path = os.path.join(msk_p, sl)


        if not os.path.exists(mask_path):
            removed += 1
            continue

        mask = np.load(mask_path)


        if np.sum(mask) > 0:
            shutil.copy(
                os.path.join(ct_p, sl),
                os.path.join(out_ct_p, sl)
            )
            shutil.copy(
                mask_path,
                os.path.join(out_msk_p, sl)
            )
            kept += 1
        else:
            removed += 1

print("===================================")
print(f"✅ Tutulan slice  : {kept}")
print(f"❌ Atılan slice  : {removed}")
print("===================================")
