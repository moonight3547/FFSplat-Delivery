#!/usr/bin/env python3
"""Official setup entry: build isolated runtimes and publish their local profile."""
import argparse
from pathlib import Path
import shutil
import subprocess
import sys

REPO=Path(__file__).resolve().parents[1]


def main(argv=None):
    argv=list(sys.argv[1:] if argv is None else argv)
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--root',type=Path,default=REPO/'.runtime')
    known,_=parser.parse_known_args(argv)
    runtime_root=known.root.expanduser().resolve()
    forwarded=list(argv)
    if not any(a=='--root' or a.startswith('--root=') for a in argv):
        forwarded=['--root',str(runtime_root),*forwarded]
    result=subprocess.run([sys.executable,str(REPO/'scripts/setup_runtime.py'),*forwarded],check=False)
    if result.returncode or '--plan' in argv or '--help' in argv or '-h' in argv:
        return result.returncode
    source=runtime_root/'profile.json'
    if not source.is_file():raise FileNotFoundError(f'Installer did not produce a verified profile: {source}')
    destination=REPO/'.runtime/profile.json'
    if source.resolve()!=destination.resolve():
        destination.parent.mkdir(parents=True,exist_ok=True)
        temporary=destination.with_name('profile.json.tmp');shutil.copyfile(source,temporary);temporary.replace(destination)
    print(f'Runtime profile ready: {destination}',flush=True)
    return 0


if __name__=='__main__':raise SystemExit(main())
