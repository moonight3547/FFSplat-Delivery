#!/usr/bin/env bash
# VGG200 e2e on H1 (task-17).  PREPARED; run only once the Lead says "H1 free".
# Serial, quiet-GPU gated; uniform hyperparameters for every case.
set -u
cd /workspace/organizer-projects/FFSplat-Hackathon || exit 1
export PYTHONPATH=$PWD
mkdir -p outputs/vgg200-20261030/logs
python3 outputs/vgg200-20261030/run_vgg200.py --arms ctrl,vgg200
# If vgg200 is positive, follow with (best case only):
#   python3 outputs/vgg200-20261030/run_vgg200.py --only <best-tag> --arms vgg500w005
