"""Result figures from result files (strict South Asia grouping, multi-run LoRA). No hand-typed numbers
except the LoRA official-split accuracy on the 124 strict copy-free test images, recomputed below from the npz."""
_ROOT = __import__('os').path.dirname(__import__('os').path.dirname(__import__('os').path.abspath(__file__)))
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

R = Path(_ROOT + "/results")
OUT = Path(_ROOT + "/figures"); OUT.mkdir(parents=True, exist_ok=True)
J = lambda f: json.load(open(R / f))
e1, e25, e7, e8, ag = (J("exp1_within_southasia_strict.json"), J("exp2to5_results_strict.json"),
                       J("exp7_few_target_fish_strict.json"), J("exp8_fish_conformal.json"), J("v2_aggregate.json"))
off = J("lora_sa_official.json")["SA_test_all"]["acc_known"]
# LoRA official-split model scored on the strict copy-free subset
z = np.load(R / "lora_sa_official.npz", allow_pickle=True); sa = pd.read_csv(R / "southasia_meta.csv")
te = np.where(sa.split == "test")[0]; tr = np.where(sa.split == "train")[0]
y = z["SA_test_all_y"].astype(str); pred = np.array(sorted(set(y)))[z["SA_test_all_logits"].argmax(1)]
trg = set(sa.group_strict.values[tr]); clean = np.array([g not in trg for g in sa.group_strict.values[te]])
off_clean = float((pred[clean] == y[clean]).mean())
json.dump({"n_clean_strict": int(clean.sum()), "lora_official_acc_on_clean_strict": round(off_clean, 4)},
          open(R / "lora_official_on_strict_clean.json", "w"), indent=1)
L = ag["lora"]
plt.rcParams.update({"font.family": "Times New Roman", "font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
C = {"frozen": "#7A8CA6", "lora": "#1F4E79", "third": "#C9A227"}
w = 0.36

# Fig 1 -- effect of leakage-aware splitting
fig, ax = plt.subplots(1, 2, figsize=(7.4, 2.9))
x = np.arange(3)
fz = [e1["dinov2s"]["A_official_split"]["acc"], e1["dinov2s"]["B_official_minus_leaky"]["acc"], e1["dinov2s"]["C_group_kfold_5x5"]["acc"]]
fzs = [0, 0, e1["dinov2s"]["C_group_kfold_5x5"]["acc_sd"]]
lo = [off, off_clean, L["SA_strict_group_split"]["acc"]["mean"]]; los = [0, 0, L["SA_strict_group_split"]["acc"]["sd"]]
for i, (v, s, k) in enumerate([(fz, fzs, "frozen"), (lo, los, "lora")]):
    b = ax[0].bar(x + (i - .5) * w, np.array(v) * 100, w, yerr=np.array(s) * 100, capsize=3, color=C[k],
                  label="Frozen DINOv2-S + LR" if k == "frozen" else "LoRA DINOv2-S")
    ax[0].bar_label(b, fmt="%.1f", fontsize=7.5, padding=2)
ax[0].set_xticks(x, ["Mirror\nsplit", "Copy-free\ntest subset", "Group\nsplit"]); ax[0].set_ylabel("Accuracy (%)")
ax[0].set_ylim(60, 106); ax[0].set_title("(a) South Asia set", fontsize=10)
fig.legend(*ax[0].get_legend_handles_labels(), fontsize=8, frameon=False, loc="upper center", ncol=2)
md = e25["dinov2s"]["exp2_matsyadx_within"]; x2 = np.arange(2)
b = ax[1].bar(x2 - w / 2, [md["random_split_macroF1"][0], md["specimen_split_macroF1"][0]], w,
              yerr=[md["random_split_macroF1"][1], md["specimen_split_macroF1"][1]], capsize=3, color=C["frozen"])
ax[1].bar_label(b, fmt="%.2f", fontsize=7.5, padding=6)
b = ax[1].bar([1 + w / 2], [L["MD_split_by_fish"]["macroF1"]["mean"]], w, yerr=[L["MD_split_by_fish"]["macroF1"]["sd"]],
              capsize=3, color=C["lora"])
ax[1].bar_label(b, fmt="%.2f", fontsize=7.5, padding=6)
ax[1].set_xticks(x2, ["Random image split", "Split by fish"]); ax[1].set_ylabel("Macro-F1"); ax[1].set_ylim(0.5, 1.07)
ax[1].set_title("(b) MatsyaDx-BD", fontsize=10)
fig.tight_layout(rect=(0, 0, 1, 0.91)); fig.savefig(OUT / "fig1_leakage_v2.png", dpi=300); plt.close(fig)

# Fig 2 -- in-domain vs cross-dataset
fig, ax = plt.subplots(figsize=(5.4, 2.9))
cr = e25["dinov2s"]["exp3_cross_dataset"]; x = np.arange(4)
fzv = [cr["SA_in_domain_macroF1"], cr["MD_in_domain_macroF1"], cr["SA_to_MD"]["macroF1"], cr["MD_to_SA"]["macroF1"]]
b = ax.bar(x - w / 2, fzv, w, color=C["frozen"], label="Frozen DINOv2-S + LR"); ax.bar_label(b, fmt="%.2f", fontsize=8)
lv = [L["SA_to_MD"]["macroF1"], L["MD_to_SA"]["macroF1"]]
b = ax.bar(x[2:] + w / 2, [d["mean"] for d in lv], w, yerr=[d["sd"] for d in lv], capsize=3, color=C["lora"], label="LoRA DINOv2-S (3 seeds)")
ax.bar_label(b, fmt="%.2f", fontsize=8, padding=5)
ax.set_xticks(x, ["SA in-domain", "MD in-domain", "SA \u2192 MD", "MD \u2192 SA"]); ax.set_ylabel("Macro-F1 (gill, red, healthy)")
ax.set_ylim(0, 1.1); ax.legend(fontsize=7.5, frameon=False, loc="upper right")
fig.tight_layout(); fig.savefig(OUT / "fig2_cross_dataset_v2.png", dpi=300); plt.close(fig)

# Fig 3 -- labelled target fish
fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.6))
d = e7["dinov2s"]; ks = [0, 1, 2, 3, 4]
m = [d[f"k={k}_fish_per_class"]["macroF1_source+target"][0] for k in ks]
sd = [d[f"k={k}_fish_per_class"]["macroF1_source+target"][1] for k in ks]
t = [np.nan] + [d[f"k={k}_fish_per_class"]["macroF1_target_only"][0] for k in ks[1:]]
ax[0].errorbar(ks, m, yerr=sd, marker="o", color=C["lora"], capsize=3, label="Source + target fish")
ax[0].plot(ks, t, marker="s", ls="--", color=C["frozen"], label="Target fish only")
ax[0].set_xlabel("Labelled MatsyaDx fish per class"); ax[0].set_ylabel("Macro-F1"); ax[0].set_xticks(ks)
ax[0].legend(fontsize=7.5, frameon=False); ax[0].set_title("(a) Classification", fontsize=10)
cov = [e25["dinov2s"]["exp5_conformal_90"]["MD_shifted"]["coverage"]] + [d[f"k={k}_fish_per_class"]["conformal90_calib_on_k_other_fish"]["coverage"] for k in ks[1:]]
ax[1].plot(ks, np.array(cov) * 100, marker="o", color=C["lora"]); ax[1].axhline(90, ls=":", color="k", lw=1)
ax[1].text(3.9, 91, "target 90 %", ha="right", fontsize=8)
ax[1].set_xlabel("Labelled calibration fish per class (0 = source calibration)"); ax[1].set_ylabel("Empirical coverage (%)"); ax[1].set_xticks(ks)
ax[1].set_ylim(45, 100); ax[1].set_title("(b) Conformal coverage on MatsyaDx-BD", fontsize=10)
fig.tight_layout(); fig.savefig(OUT / "fig3_target_fish_v2.png", dpi=300); plt.close(fig)

# Fig 4 -- photo-level vs fish-level conformal calibration (MatsyaDx-BD in-domain, 100 repeats)
A = e8["partA_in_domain_MD_4class_100reps"]; meth = ["photo", "fish_max", "fish_avg"]
fig, ax = plt.subplots(figsize=(5.4, 2.9)); x = np.arange(3)
for i, (key, lab, col) in enumerate([("photo_coverage", "Photo-level coverage", C["frozen"]), ("fish_coverage", "Fish-level coverage", C["lora"])]):
    b = ax.bar(x + (i - .5) * w, [A[m][key][0] * 100 for m in meth], w, yerr=[A[m][key][1] * 100 for m in meth], capsize=3, color=col, label=lab)
    ax.bar_label(b, fmt="%.1f", fontsize=7.5, padding=6)
ax.axhline(90, ls=":", color="k", lw=1, label="Target 90 %")
ax.set_xticks(x, [f"Photo scores\n(set size {A['photo']['mean_set_size']:.2f})", f"Worst photo per fish\n(set size {A['fish_max']['mean_set_size']:.2f})",
                  f"Mean probability per fish\n(set size {A['fish_avg']['mean_set_size']:.2f})"], fontsize=8)
ax.set_ylabel("Empirical coverage (%)"); ax.set_ylim(40, 108)
fig.legend(*ax.get_legend_handles_labels(), fontsize=8, frameon=False, loc="upper center", ncol=3)
fig.tight_layout(rect=(0, 0, 1, 0.9)); fig.savefig(OUT / "fig4_fish_conformal_v2.png", dpi=300); plt.close(fig)
print("figures written; LoRA official on strict copy-free:", round(off_clean, 4), int(clean.sum()))
