import json
from pathlib import Path
import pytest
from ffsplat_eval.organizer import load_spec, inventory, summarize, digest, geometry_digest, compare_suite


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def test_default_matches_organizer_three_cases():
    root=Path(__file__).parents[1]
    assert load_spec(root/"benchmark.json") == load_spec()
    assert [c["scene"] for c in load_spec()["cases"]][1:] == ["billboard", "counter"]


def test_inventory_is_path_independent_and_byte_sensitive(tmp_path):
    spec=load_spec()
    for prefix in ("one", "two"):
        for c in spec["cases"]:
            p=tmp_path/prefix/c["source"]/c["scene"]/"images/000.png"
            p.parent.mkdir(parents=True); p.write_bytes(b"synthetic fixture")
    assert inventory(spec,tmp_path/"one") == inventory(spec,tmp_path/"two")
    p.write_bytes(b"changed")
    assert inventory(spec,tmp_path/"one")["sha256"] != inventory(spec,tmp_path/"two")["sha256"]


def results_fixture(tmp_path):
    spec=load_spec(); write(tmp_path/"run_identity.json",dict(benchmark=spec,policy="official-fastgs"))
    for c in spec["cases"]:
        base=tmp_path/"results"/c["id"]
        work=tmp_path/"work"/c["id"]
        write(work/"module_a.json",dict(total_seconds=10))
        write(base/"resplat/resplat.json",dict(total_seconds=20))
        write(base/"module_b/module_b.json",dict(total_seconds=30))
        write(base/"reference_fastgs/module_b.json",dict(total_seconds=100,trainer="fastgs",iterations=30000,init_mode="random:100000"))
        split=dict(images=["000.png","001.png"],train=["001.png"],test=["000.png"],context=["001.png"],llffhold=8,resolution=[64,64])
        write(work/"split.json",split)
        files=["split.json","sparse/0/cameras.txt","sparse/0/images.txt","images/000.png","images/001.png"]
        for name in files[1:]:
            path=work/name; path.parent.mkdir(parents=True,exist_ok=True); path.write_bytes(b"synthetic")
        frozen=dict(split=split,files={n:digest(work/n) for n in files})
        frozen["geometry_sha256"]=geometry_digest(frozen)
        write(work/"input_identity.json",frozen)
        for suffix in ("npz","ply"): (base/f"resplat/resplat_world.{suffix}").write_bytes(b"synthetic initial state")
        write(base/"resplat/export_identity.json",{suffix+"_sha256":digest(base/f"resplat/resplat_world.{suffix}") for suffix in ("npz","ply")})
        write(base/"module_b/metrics.json",dict(num_views=1,mean=dict(psnr=30,ssim=.9,lpips=.1),per_view=[dict(name="000.png")]))
        for stage in ("prepare","resplat","train","reference"):
            write(tmp_path/"logs"/c["id"]/(stage+".json"),dict(subprocess_wall_seconds=110 if stage=="reference" else 25))
    return spec


def test_reporting_full_e2e_gate_and_no_invented_score(tmp_path):
    spec=results_fixture(tmp_path)
    report=summarize(spec,tmp_path)
    assert report["valid"] is True
    assert report["mean_end_to_end_seconds"] == 60
    assert report["official_score_100"] is None
    path=tmp_path/"results"/spec["cases"][0]["id"]/"reference_fastgs/module_b.json"
    write(path,dict(total_seconds=60,trainer="fastgs",iterations=30000,init_mode="random:100000"))
    assert summarize(spec,tmp_path)["valid"] is False  # equality is invalid


def test_missing_stage_never_receives_zero_time(tmp_path):
    spec=results_fixture(tmp_path)
    (tmp_path/"results"/spec["cases"][0]["id"]/"resplat/resplat.json").unlink()
    report=summarize(spec,tmp_path)
    assert report["complete"] is False and report["valid"] is None
    assert report["mean_metrics"] is None


def test_changed_split_bytes_rejected(tmp_path):
    spec=results_fixture(tmp_path)
    path=tmp_path/"work"/spec["cases"][0]["id"]/"split.json"
    path.write_text(path.read_text()+"\n")
    assert summarize(spec,tmp_path)["complete"] is False


def test_incomplete_hash_manifest_and_mutated_initial_state_rejected(tmp_path):
    spec=results_fixture(tmp_path)
    work=tmp_path/"work"/spec["cases"][0]["id"]
    frozen=json.loads((work/"input_identity.json").read_text())
    frozen["files"].pop("images/000.png")
    write(work/"input_identity.json",frozen)
    assert summarize(spec,tmp_path)["complete"] is False
    spec=results_fixture(tmp_path)
    (tmp_path/"results"/spec["cases"][0]["id"]/"resplat/resplat_world.npz").write_bytes(b"changed")
    assert summarize(spec,tmp_path)["complete"] is False


def suite_fixture(root):
    for cell,a,b in (("O0","uniform16","official-fastgs"),("A","ranked16","official-fastgs"),
                     ("B","uniform16","mcmc-ctf"),("AB","ranked16","mcmc-ctf")):
        spec=results_fixture(root/cell)
        identity=dict(benchmark=spec, a_policy=a, policy=b, hardware={"gpu":"fixture"},profile={"runtime":"fixture"},
                      checkpoint_sha256="fixture",source_sha256={"code":"fixture"},assets_sha256="fixture-assets",b_settings={"test":1} if b=="mcmc-ctf" else None)
        if cell != "O0":
            parent="A" if cell=="AB" else "O0"
            identity["paired_control"]=dict(identity_sha256=digest(root/parent/"run_identity.json"),
                reuse="geometry" if cell=="A" else "initial_gaussians")
        write(root/cell/"run_identity.json",identity)
        for c in spec["cases"]:
            base=root/cell/"results"/c["id"]
            if cell in ("A","AB"):
                write(root/cell/"work"/c["id"]/"module_a.json",dict(total_seconds=15,selection_seconds=5))
                (base/"resplat/resplat_world.npz").write_bytes(b"ranked initial state")
                write(base/"resplat/export_identity.json",{suffix+"_sha256":digest(base/f"resplat/resplat_world.{suffix}") for suffix in ("npz","ply")})
            gain={"O0":0,"A":1,"B":2,"AB":4}[cell]
            write(base/"module_b/metrics.json",dict(num_views=1,mean=dict(psnr=30+gain,ssim=.9,lpips=.1),per_view=[dict(name="000.png")]))
    return spec


def test_four_cell_comparison_reports_metrics_times_and_interaction(tmp_path):
    spec=suite_fixture(tmp_path)
    report=compare_suite(spec,tmp_path)
    assert report["complete"] and report["official_score_100"] is None
    assert report["interaction"]["psnr"] == 1
    assert report["comparison"][0]["deltas_vs_O0"]["AB"]["psnr"] == 4
    assert report["cells"]["AB"]["mean_stage_seconds"] == dict(A=15,ReSplat=20,B=30)
    assert report["cells"]["AB"]["mean_end_to_end_seconds"] == 65
    subset=compare_suite(spec,tmp_path,["hw"])
    assert len(subset["comparison"]) == 1 and subset["cells"]["O0"]["scope"] == "explicit group subset"


def test_comparison_rejects_hardware_and_initial_state_mismatch(tmp_path):
    spec=suite_fixture(tmp_path)
    path=tmp_path/"AB/run_identity.json"; identity=json.loads(path.read_text())
    identity["hardware"]={"gpu":"another"}; write(path,identity)
    with pytest.raises(ValueError,match="hardware"): compare_suite(spec,tmp_path)
    spec=suite_fixture(tmp_path)
    base=tmp_path/"AB/results"/spec["cases"][0]["id"]/"resplat"
    (base/"resplat_world.npz").write_bytes(b"another export")
    frozen=json.loads((base/"export_identity.json").read_text()); frozen["npz_sha256"]=digest(base/"resplat_world.npz")
    write(base/"export_identity.json",frozen)
    with pytest.raises(ValueError,match="exact initialization"): compare_suite(spec,tmp_path)


def test_partial_suite_is_not_a_comparison(tmp_path):
    spec=suite_fixture(tmp_path)
    (tmp_path/"AB/results"/spec["cases"][0]["id"]/"module_b/metrics.json").unlink()
    report=compare_suite(spec,tmp_path)
    assert report["complete"] is False and report["comparison"] is None


def test_comparison_rejects_changed_source_asset_registration(tmp_path):
    spec = suite_fixture(tmp_path)
    path = tmp_path/"AB/run_identity.json"
    value = json.loads(path.read_text()); value["assets_sha256"] = "different-source-bytes"
    write(path, value)
    with pytest.raises(ValueError, match="assets_sha256"):
        compare_suite(spec, tmp_path)


def test_host_path_bindings_are_relative_and_cli_takes_precedence(tmp_path):
    from argparse import Namespace
    from ffsplat_eval.organizer import apply_paths
    config=tmp_path/"config/paths.json"
    write(config,dict(data="../data",checkpoint="../weights.pth",assets="../manifest.json",profile="../profile.json"))
    args=Namespace(paths=config,data=None,checkpoint=Path("/explicit/weights.pth"),assets=None,profile=None)
    apply_paths(args)
    assert args.data == tmp_path/"data" and args.assets == tmp_path/"manifest.json"
    assert args.checkpoint == Path("/explicit/weights.pth")
    write(config,dict(data="../data",checkpoint="../weights.pth",assets="../manifest.json",unexpected="no"))
    with pytest.raises(ValueError,match="Path bindings"): apply_paths(args)


def test_default_three_and_optional_full_supplied_suite_pin_same_assets():
    root=Path(__file__).parents[1]
    default=load_spec(root/"benchmark.json"); extended=load_spec(root/"benchmarks/organizer-provided13-v1.json")
    assert len(default["cases"]) == 3 and len(extended["cases"]) == 13
    assert set(c["id"] for c in default["cases"]) <= set(c["id"] for c in extended["cases"])
    assert default["checkpoint"] == extended["checkpoint"] and len(default["checkpoint"]["sha256"]) == 64
    assert default["data"]["snapshot"] == extended["data"]["snapshot"]


@pytest.mark.parametrize("change", [{"trainer":"classic3dgs"}, {"trainer":None},
    {"init_mode":"ply:resplat_world.ply"}, {"iterations":10000}])
def test_reference_algorithm_and_initialization_are_enforced(tmp_path, change):
    spec=results_fixture(tmp_path)
    path=tmp_path/"results"/spec["cases"][0]["id"]/"reference_fastgs/module_b.json"
    ref=json.loads(path.read_text()); ref.update(change); write(path,ref)
    report=summarize(spec,tmp_path)
    assert report["complete"] is False and report["valid"] is None
    assert "FastGS random-init" in report["rows"][0]["error"]
