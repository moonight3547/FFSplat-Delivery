"""Content identities for derived input caches; no CUDA dependency."""
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def image_sources(work):
    root = Path(work) / 'images'
    return sorted(path for path in root.iterdir() if path.suffix.lower() in ('.png', '.jpg', '.jpeg'))


def source_identity(work):
    """Actual source bytes, not only basenames or timestamps."""
    rows = [(path.name, path.stat().st_size, sha256(path)) for path in image_sources(work)]
    return hashlib.sha256(json.dumps(rows, separators=(',', ':')).encode()).hexdigest()


def scene_identity(work, seed, version):
    work = Path(work)
    paths = [work / 'split.json', work / 'image_cache/manifest.json']
    paths += sorted(path for path in (work / 'sparse/0').iterdir()
                    if path.suffix in ('.txt', '.bin', '.ply'))
    rows = [(str(path.relative_to(work)), sha256(path)) for path in paths]
    return hashlib.sha256(json.dumps([version, int(seed), source_identity(work), rows],
                                    separators=(',', ':')).encode()).hexdigest()
