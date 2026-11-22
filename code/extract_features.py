"""Frozen-backbone feature extraction (CPU).

usage: python extract_features.py <meta.csv> <out_prefix>
meta.csv needs a 'path' column. Every image is letterboxed to a square (no
cropping, so the fish is never cut off), resized to 224 px, and passed through
  - DINOv2 ViT-S/14 (self-supervised, timm: vit_small_patch14_dinov2.lvd142m)
  - ResNet-50 (supervised ImageNet, timm: resnet50.a1_in1k)
Saves <out_prefix>_<backbone>.npy (float32, rows aligned with meta.csv).
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from PIL import Image, ImageOps

sys.stdout.reconfigure(encoding="utf-8")
torch.set_num_threads(8)
BACKBONES = {"dinov2s": "vit_small_patch14_dinov2.lvd142m", "resnet50": "resnet50.a1_in1k"}
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def load(path, size=224):
    img = Image.open(path).convert("RGB")
    img.draft("RGB", (size * 4, size * 4))  # fast JPEG downscale for 4000x3000 photos
    w, h = img.size
    s = max(w, h)
    sq = Image.new("RGB", (s, s), (0, 0, 0))
    sq.paste(img, ((s - w) // 2, (s - h) // 2))
    sq = sq.resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(sq).copy()).permute(2, 0, 1).float() / 255
    return (x - MEAN) / STD


def main(meta_csv, out_prefix):
    paths = pd.read_csv(meta_csv)["path"].tolist()
    for key, name in BACKBONES.items():
        out = Path(f"{out_prefix}_{key}.npy")
        if out.exists():
            print("skip", out)
            continue
        kw = {"img_size": 224} if "dinov2" in name else {}
        model = timm.create_model(name, pretrained=True, num_classes=0, **kw).eval()
        feats = []
        with torch.inference_mode():
            for i in range(0, len(paths), 32):
                x = torch.stack([load(p) for p in paths[i:i + 32]])
                feats.append(model(x).numpy().astype(np.float32))
                if i % 320 == 0:
                    print(key, i, "/", len(paths), flush=True)
        np.save(out, np.concatenate(feats))
        print("saved", out, np.load(out).shape)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
