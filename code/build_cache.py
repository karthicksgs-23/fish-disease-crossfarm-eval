"""Cache every image of both datasets as letterboxed 256x256 uint8 (train-time crops to 224)."""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import sys
import numpy as np
import pandas as pd
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

sys.stdout.reconfigure(encoding="utf-8")
R = _ROOT + "/results"
S = 256


def load(path):
    img = Image.open(path).convert("RGB")
    img.draft("RGB", (S * 4, S * 4))
    w, h = img.size
    m = max(w, h)
    sq = Image.new("RGB", (m, m))
    sq.paste(img, ((m - w) // 2, (m - h) // 2))
    return np.asarray(sq.resize((S, S), Image.BICUBIC))


for name in ["southasia", "matsyadx"]:
    paths = pd.read_csv(f"{R}/{name}_meta.csv").path.tolist()
    with ThreadPoolExecutor(8) as ex:
        arr = np.stack(list(ex.map(load, paths)))
    np.save(f"{R}/cache_{name}_{S}.npy", arr)
    print(name, arr.shape)
