"""JSON-serializable RNG checkpoints for reproducing native Scene loading."""
import hashlib
import json
import random

import numpy as np
import torch


def capture(cuda=True):
    state = np.random.get_state()
    return dict(python=random.getstate(), numpy=[state[0], state[1].tolist(), *state[2:]],
                torch_cpu=torch.get_rng_state().tolist(),
                torch_cuda=torch.cuda.get_rng_state().tolist() if cuda else None)


def identity(state):
    return hashlib.sha256(json.dumps(state, separators=(',', ':')).encode()).hexdigest()


def restore(state):
    py = state['python']
    random.setstate((py[0], tuple(py[1]), py[2]))
    np_state = state['numpy']
    np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(torch.tensor(state['torch_cpu'], dtype=torch.uint8))
    if state['torch_cuda'] is not None:
        torch.cuda.set_rng_state(torch.tensor(state['torch_cuda'], dtype=torch.uint8))
