"""Bind Python to the named Conda prefix even if the inherited PATH is polluted."""
import json
from pathlib import Path
import subprocess


def bind_python(command):
    command = list(command)
    if len(command) < 5 or command[1] != 'run' or '-n' not in command:
        return command
    at = command.index('-n')
    env = command[at+1]
    if command[at+2] != 'python':
        return command
    prefixes = json.loads(subprocess.check_output([command[0], 'env', 'list', '--json'], text=True))['envs']
    matches = [Path(p) for p in prefixes if Path(p).name == env]
    if len(matches) != 1:
        raise ValueError(f'Expected one Conda prefix for {env}; found {len(matches)}')
    executable = matches[0]/'bin/python'
    if not executable.is_file():
        raise FileNotFoundError(executable)
    command[at+2] = str(executable)
    return command
