import hashlib, json, sys
from pathlib import Path
import torch
sys.path.insert(0, "/tmp/fastergs_export/FasterGSCudaBackend")
from FasterGSCudaBackend import _C
from FasterGSCudaBackend.torch_bindings.adam import FusedAdam

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

so = Path(_C.__file__)
adam_py = Path("/tmp/fastergs_export/FasterGSCudaBackend/FasterGSCudaBackend/torch_bindings/adam.py")
out = dict(torch=torch.__version__, cuda=torch.version.cuda,
           so_path=str(so), so_sha256=sha(so), so_bytes=so.stat().st_size,
           class_file=str(Path(FusedAdam.__module__.replace(".", "/"))), 
           class_py_sha256=sha(adam_py),
           class_bases=[c.__name__ for c in FusedAdam.__mro__[:4]],
           has_adam_step=hasattr(_C, "adam_step"))
# numeric/state smoke: FusedAdam vs torch.optim.Adam, 35 steps incl state expansion at 12
torch.manual_seed(0)
maxdiff = {"param": 0.0, "m": 0.0, "v": 0.0}
nsteps = 35
for trial in range(2):
    torch.manual_seed(100 + trial)
    p0 = [torch.randn(1000, device="cuda", dtype=torch.float32) for _ in range(3)]
    grads = [] 
    for step in range(nsteps):
        g = [torch.randn_like(p) * 0.5 for p in p0]
        grads.append(g)
    params_f = [torch.nn.Parameter(p.clone()) for p in p0]
    params_t = [torch.nn.Parameter(p.clone()) for p in p0]
    # official FusedAdam requires one tensor per param group (same shape as 3DGS training_setup)
    opt_f = FusedAdam([{"params": [p]} for p in params_f], lr=1e-3, eps=1e-15)
    opt_t = torch.optim.Adam([{"params": [p]} for p in params_t], lr=1e-3, eps=1e-15)
    for step in range(nsteps):
        if step == 12:  # state expansion: add a fourth parameter mid-run
            extra = torch.nn.Parameter(torch.randn(500, device="cuda", dtype=torch.float32))
            p0.append(extra.detach().clone()); grads_extra = [torch.randn_like(extra) * 0.5]
            params_f.append(extra); params_t.append(torch.nn.Parameter(extra.detach().clone()))
            grads.append(grads_extra + [torch.randn_like(extra) * 0.5])
        if step == 20:  # zero grads on one param
            grads[step][0] = torch.zeros_like(p0[0])
        for p, g in zip(params_f, grads[step]): p.grad = g.clone()
        for p, g in zip(params_t, grads[step]): p.grad = g.clone()
        opt_f.step(); opt_t.step()
        opt_f.zero_grad(set_to_none=True); opt_t.zero_grad(set_to_none=True)
        for i, (a, b) in enumerate(zip(params_f, params_t)):
            maxdiff["param"] = max(maxdiff["param"], float((a - b).abs().max()))
        for sf, st in zip(opt_f.state.values(), opt_t.state.values()):
            if "exp_avg" in sf and "exp_avg" in st:
                maxdiff["m"] = max(maxdiff["m"], float((sf["exp_avg"] - st["exp_avg"]).abs().max()))
                maxdiff["v"] = max(maxdiff["v"], float((sf["exp_avg_sq"] - st["exp_avg_sq"]).abs().max()))
out["numeric_steps"] = nsteps
out["numeric_max_abs"] = maxdiff
out["numeric_ok"] = max(maxdiff.values()) < 1e-5
print(json.dumps(out, indent=2))
