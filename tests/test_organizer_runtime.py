import json
import pytest
from competition.runtime import bind_python


def test_conda_command_uses_explicit_python_despite_inherited_path(tmp_path, monkeypatch):
    prefix = tmp_path/'wanted'
    (prefix/'bin').mkdir(parents=True)
    (prefix/'bin/python').touch()
    monkeypatch.setenv('PATH', '/different/conda/env/bin')
    monkeypatch.setattr('competition.runtime.subprocess.check_output', lambda *a, **k: json.dumps({'envs': [str(prefix)]}))
    command = ['conda', 'run', '--no-capture-output', '-n', 'wanted', 'python', '-m', 'pip', 'install', 'torch==2.4.0']
    bound = bind_python(command)
    assert bound[5] == str(prefix/'bin/python')
    assert command[5] == 'python'
    assert bound[6:] == command[6:]


def test_missing_or_ambiguous_runtime_fails_before_install(monkeypatch):
    for prefixes in ([], ['/one/wanted', '/two/wanted']):
        monkeypatch.setattr('competition.runtime.subprocess.check_output', lambda *a, **k: json.dumps({'envs': prefixes}))
        with pytest.raises(ValueError, match='Expected one Conda prefix'):
            bind_python(['conda', 'run', '-n', 'wanted', 'python', '-m', 'pip'])
