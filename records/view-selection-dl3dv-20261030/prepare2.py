#!/usr/bin/env python3
"""Build candidate work dirs for the DL3DV bad-case view-selection study (task-14)."""
import json, os

REPO = "/workspace/organizer-projects/FFSplat-Hackathon"
B = os.path.join(REPO, "outputs/viewsel-dl3dv-20261030")
SCENES = [
    dict(scene="dl3dv-0b5635d5", case="dl3dv/0b5635d53b3b33294d7e49d75d96046c1157ad0713571138e51f1524a164cfce",
         budget=94.29, work="outputs/multi/dl3dv-0b5635d5/work",
         control_total=31.106, control_q=17.16, control_ctx_psnr=22.68949725981129),
    dict(scene="dl3dv-0f622eaa", case="dl3dv/0f622eaabc6dce5039205fc8828455202f40c6adab371ab707bd1e89a2f86a2e",
         budget=94.29, work="outputs/multi/dl3dv-0f622eaa/work",
         control_total=31.553, control_q=17.93, control_ctx_psnr=20.854213073193286),
    dict(scene="dl3dv-3e4dbe41", case="dl3dv/3e4dbe4135ad3ba14c9d7c89732eb0951e237e652a65007357b2d978c80cdae3",
         budget=94.29, work="outputs/multi/dl3dv-3e4dbe41/work",
         control_total=35.039, control_q=21.09, control_ctx_psnr=22.768642971971136),
    dict(scene="dl3dv-001dccbc", case="dl3dv/001dccbc1f78146a9f03861026613d8e73f39f372b545b26118e37a23c740d5f",
         budget=94.29, work="outputs/multi/dl3dv-001dccbc/work",
         control_total=78.942, control_q=64.46, control_ctx_psnr=30.947608092563776),
]
METHODS_BAD = ["uniform", "covis-fps", "covis-maxmin", "coverage-max", "extreme-baseline", "arc-uniform"]
METHODS_GOOD = ["uniform", "arc-uniform", "covis-fps"]
manifest = {"scenes": [dict(s) for s in SCENES], "candidates": []}
for sc in SCENES:
    sel = json.load(open(os.path.join(B, "selection", sc["scene"] + ".json")))
    methods = METHODS_GOOD if sc["scene"].endswith("001dccbc") else METHODS_BAD
    for meth in methods:
        ctx_names = sel["methods"][meth]["names"]
        assert len(ctx_names) == 16 and len(set(ctx_names)) == 16
        assert not (set(ctx_names) & set(sel["heldout_names"])), "heldout leaked into candidate"
        src = os.path.join(REPO, sc["work"])
        wd = os.path.join(B, "work", sc["scene"], meth)
        os.makedirs(wd, exist_ok=True)
        split = json.load(open(os.path.join(src, "split.json")))
        split["context"] = ctx_names
        json.dump(split, open(os.path.join(wd, "split.json"), "w"), indent=2)
        for name in ("images", "sparse", "module_a.json", "image_cache"):
            dst = os.path.join(wd, name)
            if os.path.islink(dst):
                os.unlink(dst)
            if os.path.exists(os.path.join(src, name)):
                os.symlink(os.path.join(src, name), dst)
        manifest["candidates"].append(dict(
            scene=sc["scene"], method=meth, case=sc["case"], budget=sc["budget"],
            work=os.path.relpath(wd, REPO), source_work=sc["work"], n_context=len(ctx_names),
            control_total=sc["control_total"], control_q=sc["control_q"],
            control_ctx_psnr=sc["control_ctx_psnr"],
            heldout_names=sel["heldout_names"], context_names=ctx_names,
            sel_stats={k: sel["methods"][meth].get(k) for k in
                       ("mean_pair_jaccard", "min_pair_jaccard", "union_visible_frac",
                        "center_min_pair", "center_span_max")}))
json.dump(manifest, open(os.path.join(B, "candidates.json"), "w"), indent=2)
print("candidates:", len(manifest["candidates"]))
for c in manifest["candidates"]:
    print(" ", c["scene"], c["method"], c["n_context"], c["work"])
