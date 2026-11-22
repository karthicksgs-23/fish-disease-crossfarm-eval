# Fish-disease image classification: leakage audit, cross-dataset, open-set and conformal evaluation

Python code for evaluating freshwater fish-disease image classifiers on two public datasets:

- duplicate / near-duplicate audit (MD5, perceptual hash, DINOv2 feature similarity) and group-aware splits;
- frozen-backbone baselines (DINOv2 ViT-S/14, ResNet-50) with logistic regression;
- LoRA fine-tuning of DINOv2 ViT-S/14 on CPU;
- cross-dataset transfer and simple shift remedies (feature centering, CORAL, labelled target fish);
- unknown-class detection (MSP, energy, kNN, Mahalanobis; AUROC, FPR95);
- split conformal prediction at photo level and fish level, including covariate-shift weighting;
- paired Wilcoxon signed-rank tests over folds and seeds.

## Repository layout

```
code/        all scripts (run from the repository root)
```

The scripts write their outputs to `results/` and `figures/` (create both before the first run).

| Script | Purpose |
|---|---|
| `audit_southasia.py` | export South Asia images, MD5 + pHash duplicate audit, `results/southasia_meta.csv` |
| `map_kaggle_split.py` | match the Hugging Face mirror to the original Kaggle files; check the Kaggle train/test split for copies |
| `dl_matsyadx.sh`, `prep_matsyadx.py` | download and unpack MatsyaDx-BD v2, `results/matsyadx_meta.csv` (fish ids) |
| `extract_features.py` | frozen DINOv2-S and ResNet-50 features (`results/feat_*.npy`) |
| `regroup_strict.py` | stricter groups: pHash groups merged with DINOv2 cosine ≥ 0.85 links |
| `build_cache.py` | letterboxed 256 × 256 image cache for fine-tuning |
| `exp_within_southasia.py` | official split vs copy-free test vs grouped cross-validation |
| `exp_cross_openset.py` | random vs fish-grouped split, cross-dataset, open-set scores, conformal |
| `exp_shift_fixes.py` | centering, CORAL, weighted conformal, combined unknown score |
| `exp_few_target.py` | adding k labelled fish from the target dataset |
| `exp_fish_conformal.py` | photo-level vs fish-level conformal calibration |
| `exp_openset_frozen_folds.py` | frozen-feature open-set baseline on the LoRA folds |
| `train_lora.py`, `run_queue.sh`, `queue.txt` | LoRA fine-tuning runs (resumable queue) |
| `check_residual_leak.py` | nearest-training-image similarity for grouped test folds |
| `aggregate_runs.py` | mean ± SD over runs and paired Wilcoxon tests (`results/v2_aggregate.json`) |
| `plot_results.py`, `plot_examples.py`, `plot_model_diagram.py` | figures |

## Setup

Python 3.11, CPU only.

```bash
pip install -r requirements.txt
```

## Data

The datasets are not included; download them yourself and respect their licences.

- **South Asia set** (CC0 1.0): Kaggle "Freshwater Fish Disease Aquaculture in South Asia",
  <https://www.kaggle.com/datasets/subirbiswas19/freshwater-fish-disease-aquaculture-in-south-asia>; the scripts read its Hugging Face mirror,
  <https://huggingface.co/datasets/panda992/fish_disease_datasets>. Place `train.parquet` and `test.parquet` in `data/southasia/`.
- **MatsyaDx-BD v2** (CC BY 4.0): <https://doi.org/10.17632/sxkynv9t7n.2>. `code/dl_matsyadx.sh` downloads the four archives into `data/matsyadx/`.

Run every script from the repository root.

## Running

```bash
mkdir -p results figures
python code/audit_southasia.py
bash   code/dl_matsyadx.sh && python code/prep_matsyadx.py
python code/extract_features.py results/southasia_meta.csv results/feat_southasia
python code/extract_features.py results/matsyadx_meta.csv  results/feat_matsyadx
python code/regroup_strict.py
python code/build_cache.py
python code/map_kaggle_split.py

# frozen-feature experiments (SA_GROUP=group_strict uses the stricter groups)
SA_GROUP=group_strict python code/exp_within_southasia.py
SA_GROUP=group_strict python code/exp_cross_openset.py
SA_GROUP=group_strict python code/exp_shift_fixes.py
SA_GROUP=group_strict python code/exp_few_target.py
python code/exp_fish_conformal.py

# LoRA runs (about 11-103 min each on an 8-core CPU), then the frozen baseline on the same folds
bash code/run_queue.sh
python code/exp_openset_frozen_folds.py
python code/check_residual_leak.py

python code/aggregate_runs.py
python code/plot_results.py
python code/plot_examples.py
python code/plot_model_diagram.py
```

Single LoRA run, e.g. fold 2 of the grouped South Asia split:

```bash
python code/train_lora.py sa_dedup --fold 2 --seed 0 --groups group_strict --epochs 8 --bs 16
```

## License

MIT, see [LICENSE](LICENSE). The datasets keep their own licences.
