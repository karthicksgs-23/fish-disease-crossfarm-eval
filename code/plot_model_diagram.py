"""Block diagram of DINOv2 ViT-S/14 + LoRA with the scoring heads.
Boxes are drawn with exact edges (padding subtracted), and every arrow starts and ends on a
box edge; branches use right-angle connectors."""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from PIL import Image, ImageOps

ROOT = _ROOT + ""
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8})
LINE = "#3a3a3a"
PAD = 0.008


class Box:
    def __init__(self, ax, x, y, w, h, text, fc, ec="#2f5d8a", fs=7.4, ls="-", lw=1.0):
        self.l, self.r, self.b, self.t = x - w / 2, x + w / 2, y - h / 2, y + h / 2
        self.x, self.y = x, y
        ax.add_patch(FancyBboxPatch((self.l + PAD, self.b + PAD), w - 2 * PAD, h - 2 * PAD,
                                    boxstyle=f"round,pad={PAD},rounding_size=0.012",
                                    fc=fc, ec=ec, lw=lw, ls=ls, zorder=2))
        if text:
            ax.text(x, y, text, ha="center", va="center", fontsize=fs, linespacing=1.3, zorder=3)


def arrow(ax, p, q):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=9, lw=1.0, color=LINE,
                                 shrinkA=0, shrinkB=0, zorder=4))


def line(ax, *pts):
    xs, ys = zip(*pts)
    ax.plot(xs, ys, color=LINE, lw=1.0, solid_capstyle="butt", zorder=4)


def main():
    md = pd.read_csv(f"{ROOT}/results/matsyadx_meta.csv")
    W, H = 7.6, 3.4
    fig = plt.figure(figsize=(W, H)); ax = fig.add_axes([0, 0, 1, 1])
    XMAX = 1.14; ax.set_xlim(0, XMAX); ax.set_ylim(0, 1); ax.axis("off")
    yx = (H / 1.0) / (W / XMAX)            # data-unit aspect, to keep the photo square
    Y = 0.55                                # main axis

    # input photo (square, letterboxed)
    im = ImageOps.exif_transpose(Image.open(md[md.health_condition == "Bacterial Red Diseases"].path.iloc[5])).convert("RGB")
    im.thumbnail((300, 300)); im = ImageOps.pad(im, (300, 300), color=(255, 255, 255))
    iw = 0.105; ih = iw / yx
    ax.imshow(np.asarray(im), extent=(0.01, 0.01 + iw, Y - ih / 2, Y + ih / 2), zorder=1, aspect="auto")
    ax.set_xlim(0, XMAX); ax.set_ylim(0, 1)
    ax.add_patch(plt.Rectangle((0.01, Y - ih / 2), iw, ih, fill=False, ec="#888888", lw=0.6, zorder=2))
    ax.text(0.01 + iw / 2, Y - ih / 2 - 0.03, "Input photo\n224 × 224", ha="center", va="top", fontsize=7.2)

    patch = Box(ax, 0.205, Y, 0.10, 0.30, "Patch\nembedding\n16 × 16\npatches\n+ [CLS]", "#e8f1fb")
    arrow(ax, (0.01 + iw, Y), (patch.l, Y))

    # transformer stack
    enc = Box(ax, 0.42, 0.50, 0.27, 0.80, "", "#f6f6f6", ec="#808080", ls="--", lw=0.9)
    ax.text(enc.x, enc.t - 0.045, "12 × transformer blocks", ha="center", va="center", fontsize=7.6, fontweight="bold")
    att = Box(ax, 0.42, 0.72, 0.23, 0.12, "Multi-head self-attention\nqkv: W x + B A x (LoRA)", "#f1eaf8")
    prj = Box(ax, 0.42, 0.55, 0.23, 0.10, "Output projection\nW x + B A x (LoRA)", "#f1eaf8")
    mlp = Box(ax, 0.42, 0.39, 0.23, 0.10, "LayerNorm + MLP\n(frozen)", "#ffffff")
    note = Box(ax, 0.42, 0.20, 0.23, 0.11, "W frozen; A, B trained\nr = 8, α = 16 (0.22 M of 22 M)", "#fff4e0", ec="#b07a1a", fs=6.9)
    arrow(ax, (patch.r, Y), (enc.l, Y))
    arrow(ax, (att.x, att.b), (prj.x, prj.t))
    arrow(ax, (prj.x, prj.b), (mlp.x, mlp.t))

    cls = Box(ax, 0.645, Y, 0.10, 0.20, "[CLS]\nfeature f\n(384-d)", "#e8f1fb")
    arrow(ax, (enc.r, Y), (cls.l, Y))

    # branch from the feature to the two heads (right-angle connector)
    head = Box(ax, 0.84, 0.78, 0.15, 0.15, "Linear head\nlogits z,\nsoftmax p", "#eaf6ea")
    bank = Box(ax, 0.84, 0.30, 0.15, 0.17, "Stored training\nfeatures, class\nmeans, covariance", "#eaf6ea", fs=7.0)
    jx = 0.725
    line(ax, (cls.r, Y), (jx, Y)); line(ax, (jx, bank.y), (jx, head.y))
    arrow(ax, (jx, head.y), (head.l, head.y)); arrow(ax, (jx, bank.y), (bank.l, bank.y))

    # outputs
    pred = Box(ax, 1.045, 0.89, 0.17, 0.10, "Class prediction", "#fdecec")
    msp = Box(ax, 1.045, 0.66, 0.17, 0.17, "MSP, energy\n(unknown);\nconformal set", "#fdecec", fs=7.0)
    knn = Box(ax, 1.045, 0.30, 0.17, 0.14, "kNN, Mahalanobis\n(unknown)", "#fdecec", fs=7.0)
    jx2 = 0.94
    line(ax, (head.r, head.y), (jx2, head.y)); line(ax, (jx2, msp.y), (jx2, pred.y))
    arrow(ax, (jx2, pred.y), (pred.l, pred.y)); arrow(ax, (jx2, msp.y), (msp.l, msp.y))
    arrow(ax, (bank.r, bank.y), (knn.l, knn.y))

    fig.savefig(f"{ROOT}/figures/fig0_model.png", dpi=300, bbox_inches="tight", pad_inches=0.03)


if __name__ == "__main__":
    main()
