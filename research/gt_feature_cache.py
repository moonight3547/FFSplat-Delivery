"""Exact GT-feature cache for the in-run VGG-LPIPS tail (N1 framework optimization).

The GT camera image is immutable, but ``lpips.LPIPS.forward`` extracts its VGG features
from scratch on every tail step.  This wrapper caches the *already normalized* GT feature
maps per camera (after the LPIPS scaling layer + VGG trunk + per-layer L2 normalization) and
evaluates only the render branch during the tail.  The loss math (weighted sum of the five
``lin`` layers' spatial means over ``(feat_render - feat_gt)^2``) is copied operation for
operation from the pinned ``lpips`` package (v0.1 normalization), so cached and live values
are bit-identical; only the GT forward pass is removed.  The render forward and the whole
LPIPS backward remain.

VRAM: the cached maps are ~60-67 MB per full-res 576x960 view (VGG16 features 64/128/256/512/
512 at 1/2…1/32 resolution); views are cached lazily on first use and the cache is bounded by
``max_views`` (LRU) to keep the peak in check.  Only the tail (200 steps in the candidate
recipe) touches it.
"""
import lpips as lpips_pkg
import torch
import torch.nn as nn

__all__ = ["GTCachedLPIPS"]


class GTCachedLPIPS(nn.Module):
    def __init__(self, base: nn.Module, max_views: int = 0):
        super().__init__()
        assert base.spatial is False, "cached path implements the non-spatial LPIPS only"
        self.base = base
        self.max_views = int(max_views)
        self._order = []
        self._cache = {}
        self.stats = dict(calls=0, hits=0, misses=0, build_seconds=0.0)

    @torch.no_grad()
    def _gt_features(self, in1):
        outs1 = self.base.net.forward(self.base.scaling_layer(in1))
        return [lpips_pkg.normalize_tensor(o) for o in outs1]

    def _put(self, key, feats):
        if self.max_views and key not in self._cache and len(self._order) >= self.max_views:
            old = self._order.pop(0)
            self._cache.pop(old, None)
            torch.cuda.empty_cache()
        self._cache[key] = feats
        self._order.append(key)

    def forward(self, in0, in1, key=None):
        """in0 = render*2-1, in1 = gt*2-1.  ``key`` should identify the immutable GT image."""
        self.stats["calls"] += 1
        feats1 = None
        if key is not None:
            feats1 = self._cache.get(key)
            if feats1 is None:
                torch.cuda.synchronize()
                t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True)
                t0.record()
                feats1 = self._gt_features(in1)
                t1.record()
                torch.cuda.synchronize()
                self.stats["build_seconds"] += t0.elapsed_time(t1) * 1e-3
                self.stats["misses"] += 1
                self._put(key, feats1)
            else:
                self.stats["hits"] += 1
        if feats1 is None:  # no key: behaviour identical to the package (no caching)
            feats1 = self._gt_features(in1)

        outs0 = self.base.net.forward(self.base.scaling_layer(in0))
        val = 0
        for kk in range(self.base.L):
            feats0 = lpips_pkg.normalize_tensor(outs0[kk])
            diff = (feats0 - feats1[kk]) ** 2
            res = lpips_pkg.spatial_average(self.base.lins[kk](diff), keepdim=True)
            val = val + res
        return val

    @torch.no_grad()
    def cache_bytes(self):
        n = 0
        for feats in self._cache.values():
            for f in feats:
                n += f.numel() * f.element_size()
        return int(n)

    @torch.no_grad()
    def cache_views(self):
        return len(self._cache)
