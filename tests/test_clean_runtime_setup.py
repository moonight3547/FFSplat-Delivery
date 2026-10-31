import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/setup_runtime.py'
spec = importlib.util.spec_from_file_location('clean_runtime_setup', SCRIPT)
setup = importlib.util.module_from_spec(spec)
spec.loader.exec_module(setup)


def args(tmp_path, **overrides):
    values = dict(root=tmp_path/'owned', resplat_prefix=None, fastgs_prefix=None,
                  conda='/opt/conda/bin/conda', cuda128_home=None, cuda124_home=None,
                  jobs=4, arch='8.9', resume=False)
    values.update(overrides)
    return argparse.Namespace(**values)


@pytest.fixture
def planning(monkeypatch):
    monkeypatch.setattr(setup, 'conda_executable', lambda _: '/opt/conda/bin/conda')
    monkeypatch.setattr(setup, 'find_toolkit', lambda version, explicit: '/usr/local/cuda-12.8' if version=='12.8' else None)


def test_two_versions_two_prefixes_and_no_shared_name(planning, tmp_path):
    plan = setup.build_plan(args(tmp_path))
    assert plan['prefixes']['resplat'] != plan['prefixes']['fastgs']
    assert all(Path(p).is_absolute() for p in plan['profile']['environments'].values())
    assert plan['lock']['runtimes']['resplat']['torch']=='2.7.0'
    assert plan['lock']['runtimes']['fastgs']['torch']=='2.4.0'
    assert plan['toolkits']['fastgs'] is None  # Provision into the NEW prefix only.
    assert not (tmp_path/'owned').exists()  # --plan has no filesystem mutation.


def test_same_prefix_is_rejected(planning, tmp_path):
    with pytest.raises(ValueError, match='distinct'):
        setup.build_plan(args(tmp_path,resplat_prefix=tmp_path/'same',fastgs_prefix=tmp_path/'same'))


def test_explicit_python_is_inside_requested_prefix():
    command=setup.conda_python('/opt/conda/bin/conda','/data/new-fastgs','-m','pip','check')
    assert command[3:6]==['-p','/data/new-fastgs','/data/new-fastgs/bin/python']


def test_whole_filesystem_reservations_are_aggregated(tmp_path,monkeypatch):
    monkeypatch.setattr(setup.shutil,'disk_usage',lambda _:SimpleNamespace(free=10*setup.GIB))
    result=setup.check_space([(tmp_path/'a',5*setup.GIB),(tmp_path/'b',4*setup.GIB)])
    assert len(result)==1
    assert result[0]['required']==11*setup.GIB
    assert not result[0]['sufficient']


def test_existing_shared_prefix_is_never_adopted(planning,tmp_path):
    shared=tmp_path/'shared';shared.mkdir();(shared/'precious.txt').write_text('old')
    a=args(tmp_path,fastgs_prefix=shared)
    with pytest.raises(FileExistsError,match='protected'):
        setup.Installer(a,setup.build_plan(a))
    assert (shared/'precious.txt').read_text()=='old'


def test_resume_requires_same_manifest_identity(planning,tmp_path):
    a=args(tmp_path)
    install=setup.Installer(a,setup.build_plan(a))
    a.resume=True
    resumed=setup.Installer(a,setup.build_plan(a))
    assert install.identity==resumed.identity
    a.fastgs_prefix=tmp_path/'different'
    with pytest.raises(FileExistsError,match='identical'):
        setup.Installer(a,setup.build_plan(a))


def test_runtime_overlay_and_caches_are_isolated(planning,tmp_path,monkeypatch):
    monkeypatch.setenv('PYTHONPATH','/old/overlay')
    monkeypatch.setenv('PIP_TARGET','/old/site')
    a=args(tmp_path);install=setup.Installer(a,setup.build_plan(a))
    assert 'PYTHONPATH' not in install.env and 'PIP_TARGET' not in install.env
    assert Path(install.env['CONDA_PKGS_DIRS']).is_relative_to(a.root)
    assert Path(install.env['TMPDIR']).is_relative_to(a.root)
    assert install.env['PYTHONNOUSERSITE']=='1'


def test_resume_does_not_repeat_successful_commands(planning,tmp_path,monkeypatch):
    a=args(tmp_path);install=setup.Installer(a,setup.build_plan(a))
    calls=[]
    monkeypatch.setattr(setup.subprocess,'run',lambda *a,**k:calls.append(a))
    install.run('one',['true']);install.run('one',['true'])
    assert len(calls)==1
    assert json.loads(install.manifest_path.read_text())['steps']['one']['status']=='complete'


def test_completed_command_cannot_silently_change_on_resume(planning,tmp_path,monkeypatch):
    a=args(tmp_path);install=setup.Installer(a,setup.build_plan(a))
    monkeypatch.setattr(setup.subprocess,'run',lambda *a,**k:None)
    install.run('one',['true'])
    with pytest.raises(RuntimeError,match='command changed'):
        install.run('one',['true','different'])


def test_resume_accepts_audited_installer_only_repair_but_not_dependency_change(planning,tmp_path):
    a=args(tmp_path);plan=setup.build_plan(a);setup.Installer(a,plan)
    a.resume=True
    plan=setup.build_plan(a);plan['source_sha256']['scripts/setup_runtime.py']='new-installer-hash'
    resumed=setup.Installer(a,plan)
    assert resumed.manifest['installer_revisions'][-1]['resumed_sha256']=='new-installer-hash'
    plan['source_sha256']['requirements/fastgs.txt']='different-dependency'
    with pytest.raises(FileExistsError):setup.Installer(a,plan)


def test_transient_git_error_is_bounded_and_recorded(planning,tmp_path,monkeypatch):
    a=args(tmp_path);install=setup.Installer(a,setup.build_plan(a));calls=[]
    def flaky(*argv,**kwargs):
        calls.append(argv)
        if len(calls)==1:raise setup.subprocess.CalledProcessError(128,argv[0])
    monkeypatch.setattr(setup.subprocess,'run',flaky)
    monkeypatch.setattr(setup.time,'sleep',lambda _:None)
    install.run('fetch',['git','fetch'],retries=3)
    assert [x['status'] for x in install.manifest['steps']['fetch']['attempts']]==['failed','complete']
