#!/usr/bin/env python3
"""Build candidate work dirs for co-visibility context selection.

For each scene x method: a new dir with a split.json identical to the source work dir
except split["context"] replaced by the candidate 16 names; images/sparse/module_a/
image_cache are symlinked to the source (inputs only, nothing modified in the source).
Writes a manifest JSON consumed by viewsel_run.sh.
"""
import json, os, shutil

REPO = "/workspace/organizer-projects/FFSplat-Hackathon"
B = os.path.join(REPO, "outputs/viewsel-20261030")
SCENES = [
    dict(scene="mipnerf360-bicycle", case="mipnerf360/bicycle", budget=95.0,
         work="outputs/cand/mipnerf360-bicycle/work",
         control_ctx_psnr=21.40678823605526, control_total=25.310, control_q=9.76),
    dict(scene="mipnerf360-counter", case="mipnerf360/counter", budget=92.57,
         work="outputs/cand/mipnerf360-counter/work",
         control_ctx_psnr=26.638701895514906, control_total=55.172, control_q=40.28),
    dict(scene="dl3dv-001dccbc",
         case="dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f",
         budget=94.29, work="outputs/multi/dl3dv-001dccbc/work",
         control_ctx_psnr=30.947608092563776, control_total=78.942, control_q=64.46),
]
methods = ["uniform", "covis-cover", "covis-maxmin", "covis-fps"]
manifest = {"scenes": [], "candidates": []}
for sc in SCENES:
    sel = json.load(open(os.path.join(B, "selection", sc["scene"] + ".json")))
    src = os.path.join(REPO, sc["work"])
    for meth in methods:
        assert meth in sel["methods"], (sc["scene"], meth)
        ctx_names = sel["methods"][meth]["names"]
        assert len(ctx_names) == 16 and len(set(ctx_names)) == 16
        wd = os.path.join(B, "work", sc["scene"], meth)
        os.makedirs(wd, exist_ok=True)
        split = json.load(open(os.path.join(src, "split.json")))
        # context may be stored as names (v2 prepared dirs); keep the same schema
        split["context"] = ctx_names
        # keep images/train/test untouched -- selection only rewrites context
        json.dump(split, open(os.path.join(wd, "split.json"), "w"), indent=2)
        for name in ("images", "sparse", "module_a.json", "image_cache"):
            dst = os.path.join(wd, name)
            if os.path.islink(dst):
                os.unlink(dst)
            if os.path.exists(os.path.join(src, name)):
                os.symlink(os.path.join(src, name), dst)
        manifest["candidates"].append(dict(
            scene=sc["scene"], method=meth, case=sc["case"], budget=sc["budget"],
            work=os.path.relpath(wd, REPO), source_work=sc["work"],
            control_ctx_psnr=sc["control_ctx_psnr"],
            control_total=sc["control_total"], control_q=sc["control_q"],
            context_names=ctx_names,
            covis_stats={k: sel["methods"][meth][k] for k in
                         ("mean_pair_jaccard", "median_pair_jaccard", "min_pair_jaccard",
                          "union_visible_frac", "mean_visible_points")}))
    manifest["scenes"].append(sc)
os.makedirs(os.path.join(B, "work"), exist_ok=True)
json.dump(manifest, open(os.path.join(B, "candidates.json"), "w"), indent=2)
print("candidates:", len(manifest["candidates"]))
for c in manifest["candidates"]:
    print(" ", c["scene"], c["method"], c["work"])
