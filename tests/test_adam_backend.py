from types import SimpleNamespace

import pytest
import torch

from research.adam_backend import AdamBackend


def test_official_requires_manifest_and_cannot_silently_fall_back():
    with pytest.raises(ValueError):
        AdamBackend('official')
    with pytest.raises(ValueError):
        AdamBackend('auto')


def test_foreach_backend_keeps_parameters_and_uses_actual_dispatch():
    model = SimpleNamespace(**{name: torch.nn.Parameter(torch.ones(2, 3)) for name in
        ('_xyz', '_features_dc', '_features_rest', '_opacity', '_scaling', '_rotation')})
    before = {name: id(value) for name, value in vars(model).items()}
    def setup(model, options):
        model.optimizer = torch.optim.Adam([
            dict(params=[model._xyz], name='xyz'), dict(params=[model._features_dc], name='f_dc'),
            dict(params=[model._opacity], name='opacity'), dict(params=[model._scaling], name='scaling'),
            dict(params=[model._rotation], name='rotation')], lr=.01, eps=1e-15)
        model.shoptimizer = torch.optim.Adam([dict(params=[model._features_rest], name='f_rest')], lr=.01, eps=1e-15)
    with AdamBackend('foreach') as backend:
        backend.training_setup(model, None, setup)
        for _ in range(2):
            for name in before:
                getattr(model, name).grad = torch.ones_like(getattr(model, name))
            model.optimizer.step(); model.shoptimizer.step()
            model.optimizer.zero_grad(set_to_none=True); model.shoptimizer.zero_grad(set_to_none=True)
        proof = backend.audit(model)
    assert before == {name: id(getattr(model, name)) for name in before}
    assert proof['actual'] == 'foreach' and proof['calls']['_multi_tensor_adam'] > 0
    assert proof['calls']['_single_tensor_adam'] == proof['calls']['_fused_adam'] == 0
    assert all(group['state_step'] == 2 for group in proof['groups'].values())


def test_perceptual_extra_render_keeps_geometry_gradient_and_reuses_photo_when_equal():
    from research.perceptual_render import image_for_perceptual
    geometry = torch.tensor(2., requires_grad=True)
    photo = geometry * 2
    def renderer(*args):
        return dict(render=geometry * 3)
    reused, extra = image_for_perceptual(photo, renderer, None, None, None, None, .5, .5)
    assert reused is photo and not extra
    full, extra = image_for_perceptual(photo, renderer, None, None, None, None, 1., .5)
    full.backward()
    assert extra and geometry.grad == 3
