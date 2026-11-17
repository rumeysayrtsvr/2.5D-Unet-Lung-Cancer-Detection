import cv2
import numpy as np
import torch


def hu_norm(x):
    x = np.clip(x, -1000, 400)
    return (x + 1000) / 1400.0

def make_25d(a, b, c):
    x = np.stack([hu_norm(a), hu_norm(b), hu_norm(c)], axis=0)
    return torch.from_numpy(x).float()



def largest_cc(mask):
    num, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    if num <= 1:
        return mask
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    out = np.zeros_like(mask)
    out[lab == largest] = 1
    return out

def overlay(ct, mask, color):
    img = cv2.cvtColor(ct, cv2.COLOR_GRAY2RGB)
    lay = img.copy()
    lay[mask == 1] = color
    out = cv2.addWeighted(lay, 0.45, img, 0.55, 0)

    if mask.sum() > 0:
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, color, 2)

    return out
