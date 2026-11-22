"""Extract MatsyaDx-BD v2 archives and build results/matsyadx_meta.csv.

Group id = class + species + specimen (one physical fish); burst photos of the
same fish must never be split across train and test.
"""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import sys
from pathlib import Path

import pandas as pd
import py7zr

sys.stdout.reconfigure(encoding="utf-8")
ROOT = Path(_ROOT + "")
D = ROOT / "data" / "matsyadx"
RAW = D / "raw"
RAW.mkdir(exist_ok=True)

for a in sorted(D.glob("*.7z")):
    flag = RAW / (a.stem + ".done")
    if flag.exists():
        continue
    print("extracting", a.name, flush=True)
    with py7zr.SevenZipFile(a, "r") as z:
        z.extractall(RAW)
    flag.touch()

meta = pd.read_csv(D / "metadata.csv")
# locate each file on disk (archive top folder may or may not repeat the class name)
on_disk = {p.name: p for p in RAW.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
meta["path"] = meta.image_id.map(lambda n: str(on_disk.get(n, "")))
missing = (meta.path == "").sum()
print("images in metadata:", len(meta), "| found on disk:", len(meta) - missing)
meta = meta[meta.path != ""].reset_index(drop=True)
meta["group"] = meta.health_condition + "|" + meta.fish_category + "|" + meta.specimen_id.astype(str)
print(meta.groupby("health_condition").agg(images=("image_id", "size"), fish=("group", "nunique")))
meta.to_csv(ROOT / "results" / "matsyadx_meta.csv", index=False)
