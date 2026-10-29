import argparse
import ast
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
from competition.budget import Deadline, validate_settings
from competition.contracts import read_json, write_json
from competition.pairing import freeze_inputs, clone_work, verify_inputs
from competition.selection import proposals, rank_candidates, evaluate_context
from competition.suite import build_suite
from competition.run import build_commands

ROOT = Path(__file__).parents[1]


def settings():
    return read_json(ROOT/"competition/policies/mcmc-ctf.json")


def test_deadline_uses_setup_reserve_and_shared_lr_ctf_progress():
    cfg = settings(); clock = Deadline(100, 10, cfg)
    assert clock.allowed == 80
    for elapsed, scale in ((0, .5), (20, .75), (40, 1.)):
        plan = clock.plan(elapsed, 0, 1000)
        assert plan["scale"] == scale
    assert clock.plan(72, 0, 1000)["position_lr"] == pytest.approx(cfg["position_lr_final"])
    assert clock.plan(80, 0, 1000)["stop_reason"] == "time_allocation"
    assert clock.plan(0, cfg["max_updates"], 1000)["stop_reason"] == "max_updates"
    assert Deadline(2, 2, cfg).plan(0, 0, 1000)["stop_reason"] == "time_allocation"


def test_resolution_and_population_jump_increase_cost_before_step():
    clock = Deadline(100, 10, settings())
    clock.observe(2, .5, 100)
    assert clock.plan(0, 1, 100)["predicted_step_seconds"] == 3
    assert clock.plan(40, 1, 200)["predicted_step_seconds"] == 24
    assert clock.plan(70, 1, 200)["stop_reason"] == "next_step_reserve"


@pytest.mark.parametrize("change", [dict(e2e_fraction=1), dict(max_updates=True), dict(noise_lr=float("nan")),
    dict(ctf=[dict(until=1, scale=.5)]), dict(unused_knob=1), dict(position_lr_final=.1)])
def test_invalid_or_unused_settings_fail(change):
    cfg = settings(); cfg.update(change)
    with pytest.raises(ValueError): validate_settings(cfg)


def training_views(count=80):
    result = []
    for i in range(count):
        pose = np.eye(4, dtype="f4"); pose[0, 3] = i
        result.append(dict(name=f"{i:03}.png", path=f"{i:03}.png", c2w=pose, K=np.eye(3, dtype="f4")))
    return result


def test_proposals_are_bounded_unique_and_do_not_use_probes():
    views = training_views()
    context = [v["name"] for v in views[:16]]
    probes, candidates = proposals(views, context)
    assert len(probes) == 8 and len(candidates) == 3
    assert candidates[0] == context
    for candidate in candidates:
        assert len(candidate) == len(set(candidate)) == 16
        assert set(candidate).isdisjoint(probes)
        assert len(set(candidate)-set(context)) <= 2
    assert proposals(views[:16], context) == ([], [context])


def test_ranker_requires_gain_and_rejects_invalid_loss():
    candidates = [["a"], ["b"]]
    assert rank_candidates(candidates, lambda c: dict(loss=1 if c == ["a"] else .99))[0] == 0
    assert rank_candidates(candidates, lambda c: dict(loss=1 if c == ["a"] else .9))[0] == 1
    with pytest.raises(ValueError): rank_candidates(candidates, lambda c: dict(loss=float("nan")))


def test_probe_rgb_never_enters_refinement_and_poses_use_same_pivot():
    import torch
    # Execute the pinned organizer's real batch builder without importing its CUDA model stack.
    tree = ast.parse((ROOT/"competition/upstream/resplat/scripts/infer_colmap.py").read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_batch")
    namespace = dict(torch=torch, move_to_device=lambda batch, device: batch,
                     camera_normalization=lambda pivot, poses: torch.linalg.inv(pivot) @ poses)
    exec(compile(ast.Module(body=[node], type_ignores=[]), "pinned_build_batch", "exec"), namespace)
    views = training_views(18)
    context, probes = [v["name"] for v in views[:16]], [v["name"] for v in views[16:]]
    expected = [8., 9.]
    class Decoder:
        def forward(self, gaussians, poses, ks, near, far, shape, depth_mode=None):
            assert poses[0, :, 0, 3].tolist() == expected
            return SimpleNamespace(color=torch.ones(1, len(probes), 3, *shape)*.8)
    seen = []
    def inference(encoder, decoder, batch, num_refine, **kwargs):
        seen.append(True)
        assert num_refine == 2
        assert torch.equal(batch["context"]["image"], batch["target"]["image"])
        assert torch.all(batch["target"]["image"] == .2)
        return object(), None, None, None
    def load(paths, h, w):
        return torch.stack([torch.full((3, h, w), .8 if n in probes else .2) for n in paths])
    upstream = SimpleNamespace(load_and_preprocess_images=load, build_batch=namespace["build_batch"], run_inference=inference)
    result = evaluate_context(upstream, (None, Decoder(), lambda x: x), views, context, probes, (64, 64), "cpu")
    assert seen and result["loss"] == pytest.approx(0, abs=1e-6)
    with pytest.raises(ValueError):
        evaluate_context(upstream, (None, Decoder(), lambda x: x), views, context, [context[0]], (64, 64), "cpu")


def prepared_fixture(path):
    names = [f"{i:03}.png" for i in range(24)]
    train = [n for i, n in enumerate(names) if i % 8]
    split = dict(images=names, train=train, test=names[::8], context=train[:16], llffhold=8, resolution=[128,192])
    write_json(path/"split.json", split); write_json(path/"module_a.json", dict(total_seconds=10))
    sparse = path/"sparse/0"; sparse.mkdir(parents=True)
    (sparse/"cameras.txt").write_text("synthetic camera bytes")
    (sparse/"images.txt").write_text("synthetic pose bytes")
    (path/"images").mkdir()
    for name in names: (path/"images"/name).write_bytes(b"tiny synthetic image")
    freeze_inputs(path, "fixture")


def test_pairing_clones_mutable_metadata_and_preserves_geometry(tmp_path):
    control, candidate = tmp_path/"control", tmp_path/"candidate"
    prepared_fixture(control)
    clone_work(control, candidate)
    before = verify_inputs(control)
    split = read_json(candidate/"split.json"); split["context"][-1] = split["train"][-1]
    write_json(candidate/"split.json", split)
    after = freeze_inputs(candidate, "fixture")
    assert after["geometry_sha256"] == before["geometry_sha256"]
    assert verify_inputs(control) == before
    assert read_json(control/"split.json") != read_json(candidate/"split.json")
    (candidate/"images/000.png").write_bytes(b"mutation must be detected")
    with pytest.raises(ValueError): verify_inputs(candidate)


def test_suite_pairs_each_b_with_matching_a_state(tmp_path):
    args = argparse.Namespace(spec=tmp_path/"spec.json", data=tmp_path/"data", work=tmp_path/"suite",
        checkpoint=tmp_path/"weights.pth", profile=ROOT/"competition/profiles/linux-cuda.json",
        b_settings=ROOT/"competition/policies/mcmc-ctf.json", groups=None)
    rows = build_suite(args)
    assert [r["cell"] for r in rows] == ["O0", "A", "B", "AB"]
    for i, parent in ((1,"O0"), (2,"O0"), (3,"A")):
        cmd = rows[i]["argv"]
        assert cmd[cmd.index("--from-control")+1] == str(args.work/parent)
    assert "--reuse-geometry-only" in rows[1]["argv"]
    args.upstream=ROOT/"competition/upstream"; args.stage="all"; args.a_policy="ranked16"
    args.policy="official-fastgs"; args.from_control=args.work/"O0"; args.reuse_geometry_only=True
    commands=build_commands(args, dict(cases=[dict(id="hw/x",group="hw",source="hw",scene="x")]), read_json(args.profile))
    assert [c["stage"] for c in commands] == ["resplat", "train", "render", "metrics"]
    assert "competition.rank_resplat" in commands[0]["argv"]


def test_training_loss_has_finite_gradients_and_adam_starts_fresh():
    import torch
    from competition.train_utils import ssim_loss, make_parameters, make_optimizers
    a = torch.rand(16, 16, 3, requires_grad=True); b = torch.rand_like(a)
    loss = .8*(a-b).abs().mean()+.2*ssim_loss(a, b); loss.backward()
    assert torch.isfinite(a.grad).all()
    arrays = [np.ones((2, n), dtype="f4") for n in (3,3,4,3,45,1)]
    params = make_parameters(arrays, "cpu"); opts = make_optimizers(params, settings())
    assert all(not opt.state for opt in opts.values())
    params["means"].sum().backward(); opts["means"].step()
    assert opts["means"].state[params["means"]]["step"] == 1
    assert np.all(arrays[0] == 1)


def test_opacity_endpoints_are_clamped_only_for_train_parameters(tmp_path):
    from competition.bridge import load_world, parameter_arrays
    world = dict(means=np.zeros((2,3)),scales=np.ones((2,3)), rotations_xyzw=np.array([[0,0,0,1]]*2),
                 harmonics=np.zeros((2,3,16)),opacities=np.array([0,1]))
    np.savez(tmp_path/"world.npz", **world)
    assert np.isfinite(parameter_arrays(load_world(tmp_path/"world.npz"), np.zeros(3), 1)[-1]).all()


def test_backend_report_rejects_camera_mismatch(tmp_path):
    from competition.render_probe import compare
    fixture=dict(names=np.array(["train.png"]),camera_identity=np.array("fixed"),renders=np.zeros((1,4,4,3)))
    np.savez(tmp_path/"a.npz", **fixture); np.savez(tmp_path/"b.npz", **fixture)
    assert compare(tmp_path/"a.npz", tmp_path/"b.npz")["rows"][0]["identical"]
    fixture["camera_identity"] = np.array("changed")
    np.savez(tmp_path/"b.npz", **fixture)
    with pytest.raises(ValueError): compare(tmp_path/"a.npz", tmp_path/"b.npz")


def test_seeded_model_load_does_not_shift_forward_random_state(tmp_path):
    import torch
    from competition.resplat import load_runner, seed_inference
    path=tmp_path/"baseline/run_resplat.py"; path.parent.mkdir()
    path.write_text("import torch\ndef build_model(**kwargs):\n    return torch.rand(10)\n")
    upstream=load_runner(tmp_path)
    first_model=upstream.build_model(); actual=torch.rand(5)
    seed_inference(); expected=torch.rand(5)
    assert torch.equal(actual,expected)
    assert torch.equal(first_model,upstream.build_model())


def test_ranked_wrapper_shares_model_and_charges_a_cost(tmp_path, monkeypatch):
    import sys
    import torch
    from competition import rank_resplat, bridge, selection
    work=tmp_path/"work"; out=tmp_path/"resplat"; prepared_fixture(work)
    split=read_json(work/"split.json")
    views=training_views(len(split["train"]))
    for view,name in zip(views,split["train"]): view.update(name=name,path=name)
    winner=split["context"][:-1]+[split["train"][-1]]
    probes=[split["train"][-2]]
    monkeypatch.setattr(bridge,"read_training_views",lambda path:(split,views))
    monkeypatch.setattr(selection,"proposals",lambda *args:(probes,[split["context"],winner]))
    monkeypatch.setattr(selection,"evaluate_context",lambda upstream,model,views,context,*args:dict(loss=.8 if context==winner else 1))
    monkeypatch.setattr(torch.cuda,"synchronize",lambda:None)
    counts=[]; model=(object(),object(),lambda x:x)
    upstream=SimpleNamespace(PRESET_OVERRIDES=["pinned"])
    def build(**kwargs): counts.append(kwargs); return model
    upstream.build_model=build
    def main():
        assert read_json(work/"split.json")["context"] == winner
        assert upstream.build_model(**counts[0]) is model
        write_json(out/"resplat.json",dict(total_seconds=1))
    upstream.main=main
    monkeypatch.setattr(rank_resplat,"load_runner",lambda _:upstream)
    monkeypatch.setattr(sys,"argv",["rank", "--work",str(work),"--out",str(out),"--checkpoint",str(tmp_path/"weights")])
    rank_resplat.main()
    assert len(counts) == 1
    assert upstream.build_model is build
    a=read_json(work/"module_a.json")
    assert a["total_seconds"] == pytest.approx(10+a["selection_seconds"])
    assert read_json(work/"selection.json")["selected"] == 1
