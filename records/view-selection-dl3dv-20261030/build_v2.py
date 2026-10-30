#!/usr/bin/env python3
"""Build extreme-baseline-v2 (tau=60pct) candidate work dirs (CPU only)."""
import json, os
REPO="/workspace/organizer-projects/FFSplat-Hackathon"; B=os.path.join(REPO,"outputs/viewsel-dl3dv-20261030")
man=json.load(open(os.path.join(B,"candidates.json")))
scenes={s["scene"]: s for s in man["scenes"]}
for scene in ("dl3dv-0b5635d5","dl3dv-0f622eaa","dl3dv-3e4dbe41"):
    sel=json.load(open(os.path.join(B,"selection-v2",scene+".json")))
    names=sel["methods"]["extreme-baseline"]["names"]
    src=os.path.join(REPO, scenes[scene]["work"]); wd=os.path.join(B,"work",scene,"extreme-baseline-v2")
    os.makedirs(wd,exist_ok=True)
    split=json.load(open(os.path.join(src,"split.json"))); split["context"]=names
    json.dump(split,open(os.path.join(wd,"split.json"),"w"),indent=2)
    for nm in ("images","sparse","module_a.json","image_cache"):
        dst=os.path.join(wd,nm)
        if os.path.islink(dst): os.unlink(dst)
        if os.path.exists(os.path.join(src,nm)): os.symlink(os.path.join(src,nm),dst)
    print(scene,"built",wd,"heldout:",len(sel["heldout_names"]))
