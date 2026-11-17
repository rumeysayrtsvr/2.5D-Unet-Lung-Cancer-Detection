import matplotlib
matplotlib.use("Agg")


import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, session, redirect, url_for
import secrets

import torchvision.transforms as T

from model import load_model
from preprocess import make_25d, largest_cc, overlay
from classifier import load_classifier, CLASSES


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

SEG_MODEL_P = os.path.join(BASE_DIR, "models", "best_denseunet_25d.pth")
CLS_MODEL_P = os.path.join(BASE_DIR, "models", "best_resnet50_lung.pth")

OUT_DIR    = os.path.join(BASE_DIR, "static", "outputs")
UPLOAD_DIR = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

seg_model = load_model(SEG_MODEL_P, DEVICE)
cls_model = load_classifier(CLS_MODEL_P, DEVICE)

cls_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


app = Flask(__name__)
app.secret_key = secrets.token_hex(16)


@app.route("/", methods=["GET", "POST"])
def index():

  
    if request.method == "GET" and not session.get("has_result"):
        session.clear()

    result_image = session.get("result_image")
    histo_label  = session.get("histo_label")
    histo_conf   = session.get("histo_conf")
    histo_image  = session.get("histo_image")
    error = None


    if request.method == "POST":

   
        if "run_ct_upload" in request.form:

            files = request.files.getlist("ct_slices")

            if len(files) != 3:
                error = "Lütfen tam olarak 3 adet CT slice (.npy) seçin."
                return render_template("index.html", error=error)

           
            paths = []
            for f in files:
                fname = f"{secrets.token_hex(4)}_{f.filename}"
                path = os.path.join(UPLOAD_DIR, fname)
                f.save(path)
                paths.append(path)

            paths = sorted(paths)


            ct_prev = np.load(paths[0])
            ct_mid  = np.load(paths[1])
            ct_next = np.load(paths[2])


            ct_prev = (ct_prev - ct_prev.min()) / (np.ptp(ct_prev) + 1e-8)
            ct_mid  = (ct_mid  - ct_mid.min())  / (np.ptp(ct_mid)  + 1e-8)
            ct_next = (ct_next - ct_next.min()) / (np.ptp(ct_next) + 1e-8)


            x = make_25d(ct_prev, ct_mid, ct_next).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                prob = torch.sigmoid(seg_model(x))[0, 0].cpu().numpy()

            pred = largest_cc((prob > 0.32).astype(np.uint8))


            ct_show = (ct_mid * 255).astype(np.uint8)

            ct_show = np.rot90(ct_show, k=3)
            pred    = np.rot90(pred, k=3)

            ct_show = np.fliplr(ct_show)
            pred    = np.fliplr(pred)

 
            fig, ax = plt.subplots(1, 2, figsize=(10, 4))

            ax[0].imshow(ct_show, cmap="gray")
            ax[0].set_title("CT Slice (Middle)")

            ax[1].imshow(overlay(ct_show, pred, (0, 255, 0)))
            ax[1].set_title("Prediction")

            for a in ax:
                a.axis("off")

            plt.tight_layout()

            out_name = f"seg_{secrets.token_hex(4)}.png"
            out_path = os.path.join(OUT_DIR, out_name)
            plt.savefig(out_path, dpi=200)
            plt.close()

            session["result_image"] = f"outputs/{out_name}"
            session["has_result"] = True

            return redirect(url_for("index"))


        if "run_histo" in request.form:

            file = request.files.get("histo_image")

            if file:
                fname = f"{secrets.token_hex(4)}_{file.filename}"
                save_path = os.path.join(UPLOAD_DIR, fname)
                file.save(save_path)

                img = cv2.imread(save_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                x = cls_transform(img).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    logits = cls_model(x)
                    probs = torch.softmax(logits, dim=1)[0]
                    cid = probs.argmax().item()

                session["histo_label"] = CLASSES[cid]
                session["histo_conf"]  = float(probs[cid] * 100)
                session["histo_image"] = f"uploads/{fname}"
                session["has_result"]  = True

                return redirect(url_for("index"))

 
    return render_template(
        "index.html",
        result_image=result_image,
        histo_label=histo_label,
        histo_conf=histo_conf,
        histo_image=histo_image,
        error=error
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5001, debug=True, use_reloader=False)
