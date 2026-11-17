import os, random

BASE = "/Users/rumeysa/Desktop/2.5D-Unet/NLST_2D_TUMOR_ONLY/CT"
OUT  = "/Users/rumeysa/Desktop/2.5D-Unet/splits"

os.makedirs(OUT, exist_ok=True)

patients = sorted(os.listdir(BASE))
random.shuffle(patients)

n = len(patients)
train = patients[:int(0.7*n)]
val   = patients[int(0.7*n):int(0.85*n)]
test  = patients[int(0.85*n):]

for name, lst in zip(["train","val","test"], [train,val,test]):
    with open(os.path.join(OUT, f"{name}.txt"), "w") as f:
        f.write("\n".join(lst))

print("✅ splits created")
