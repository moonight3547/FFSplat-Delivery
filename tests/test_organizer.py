import argparse
from pathlib import Path
import numpy as np
import pytest
from competition.contracts import validate_split, remaining_b_seconds, write_json
from competition.bridge import load_world, parameter_arrays, export_ply, read_training_views
from competition.run import build_commands


def split_fixture():
    names = [f"{i:03}.png" for i in range(24)]
    train = [n for i, n in enumerate(names) if i % 8]
    return dict(images=names, train=train, test=names[::8], context=train[:16], llffhold=8, resolution=[128, 192])


def test_split_rejects_leakage_and_wrong_resolution():
    split = split_fixture()
    validate_split(split)
    split["context"][0] = split["test"][0]
    with pytest.raises(ValueError, match="held-out"):
        validate_split(split)
    split = split_fixture(); split["resolution"] = [960, 640]
    with pytest.raises(ValueError, match="limit"):
        validate_split(split)


def test_e2e_budget_charges_a_and_resplat():
    assert remaining_b_seconds(100, 20, 10) == 50
    with pytest.raises(ValueError, match="exhausts"):
        remaining_b_seconds(100, 60, 20)
    with pytest.raises(ValueError):
        remaining_b_seconds(float("nan"), 0, 0)


def test_world_ply_roundtrip_preserves_frame_sh_and_covariance(tmp_path):
    from plyfile import PlyData
    from scipy.spatial.transform import Rotation
    rng = np.random.default_rng(7)
    world = dict(means=rng.normal(size=(5, 3)).astype("f4"), scales=np.exp(rng.normal(size=(5, 3))).astype("f4"),
        rotations_xyzw=Rotation.random(5, random_state=rng).as_quat().astype("f4"),
        harmonics=rng.normal(size=(5, 3, 16)).astype("f4"), opacities=np.linspace(.1, .9, 5).astype("f4"))
    np.savez(tmp_path/"world.npz", **world)
    loaded = load_world(tmp_path/"world.npz")
    center = np.array([1, 2, 3], dtype="f4"); scale = .7
    arrays = parameter_arrays(loaded, center, scale)
    export_ply(tmp_path/"result.ply", arrays, center, scale)
    data = PlyData.read(tmp_path/"result.ply")["vertex"].data
    np.testing.assert_allclose(np.column_stack([data[k] for k in ("x", "y", "z")]), world["means"], atol=4e-7)
    np.testing.assert_allclose(np.exp(np.column_stack([data[f"scale_{i}"] for i in range(3)])), world["scales"], rtol=2e-6)
    np.testing.assert_allclose(np.column_stack([data[f"rot_{i}"] for i in range(4)]), world["rotations_xyzw"][:, [3,0,1,2]])
    np.testing.assert_array_equal(np.column_stack([data[f"f_rest_{i}"] for i in range(45)]).reshape(5,3,15), world["harmonics"][:,:,1:])
    np.testing.assert_allclose(1/(1+np.exp(-data["opacity"])), world["opacities"], atol=1e-7)


def test_camera_bridge_only_exposes_train_images(tmp_path):
    split = split_fixture(); write_json(tmp_path/"split.json", split)
    sparse = tmp_path/"sparse/0"; sparse.mkdir(parents=True)
    (sparse/"cameras.txt").write_text("1 PINHOLE 192 128 100 100 96 64\n")
    (sparse/"images.txt").write_text("".join(f"{i+1} 1 0 0 0 {i} 0 0 1 {n}\n\n" for i,n in enumerate(split["images"])))
    _, views = read_training_views(tmp_path)
    assert [v["name"] for v in views] == split["train"]
    assert not set(v["name"] for v in views) & set(split["test"])
    np.testing.assert_array_equal(views[0]["c2w"][:3,3], [-1,0,0])


def test_runner_uses_fixed_official_parameters_and_conda(tmp_path):
    from competition.contracts import read_json
    root = Path(__file__).parents[1]
    spec = dict(cases=[dict(id=f"{group}/fixture", group=group, source=group, scene="fixture")
                      for group in ("dl3dv", "hw", "mipnerf360")])
    profile = read_json(root/"competition/profiles/linux-cuda.json")
    args = argparse.Namespace(upstream=root/"competition/upstream", work=tmp_path/"run", data=tmp_path/"data",
        checkpoint=tmp_path/"weights.pth", groups=None, stage="all", policy="official-fastgs")
    commands = build_commands(args, spec, profile)
    assert len(commands) == 18
    assert all(c["argv"][:3] == ["conda", "run", "--no-capture-output"] for c in commands)
    ref = next(c["argv"] for c in commands if c["stage"] == "reference")
    assert ref[ref.index("--iterations")+1] == "30000" and "--init_random" in ref
    assert any(str(x).endswith("train_from_ply_fastgs.py") for x in ref)
    assert "--init_ply" not in ref
    assert ref[ref.index("-m")+1].endswith("reference_fastgs")
    args.policy="mcmc-ctf"
    migrated = build_commands(args, spec, profile)
    assert next(c["argv"] for c in migrated if c["stage"] == "render")[-1] == "-1"
    args.from_control=tmp_path/"control"
    paired = build_commands(args, spec, profile)
    assert len(paired) == 9
    assert {c["stage"] for c in paired} == {"train", "render", "metrics"}


@pytest.mark.parametrize("change", [{"trainer": "classic3dgs"}, {"trainer": None},
    {"iterations": 10000}, {"init_mode": "ply:resplat_world.ply"},
    {"total_seconds": float("nan")}, {"total_seconds": 0}])
def test_fastgs_reference_rejects_wrong_algorithm_initialization_or_timing(change):
    from competition.contracts import validate_reference
    ref = dict(trainer="fastgs", iterations=30000, init_mode="random:99960", total_seconds=100)
    assert validate_reference(ref) == ref
    with pytest.raises(ValueError): validate_reference(dict(ref, **change))
