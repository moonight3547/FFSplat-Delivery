"""Lossless vectorized PLY row packing; no optimizer, SH or rasterizer changes."""
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement


def structured_rows(attributes, names):
    attributes=np.asarray(attributes)
    if attributes.ndim!=2 or attributes.shape[1]!=len(names) or len(set(names))!=len(names):
        raise ValueError('Attribute matrix and ordered unique field names must agree')
    dense=np.ascontiguousarray(attributes,dtype=np.float32)
    dtype=np.dtype([(name,'f4') for name in names])
    return dense.view(dtype).reshape(-1)


def save_ply_vectorized(model,path):
    """Same attribute extraction/order/header as pinned FastGS; vectorize row packing."""
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    xyz=model._xyz.detach().cpu().numpy()
    normals=np.zeros_like(xyz)
    dc=model._features_dc.detach().transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
    rest=model._features_rest.detach().transpose(1,2).flatten(start_dim=1).contiguous().cpu().numpy()
    opacity=model._opacity.detach().cpu().numpy()
    scale=model._scaling.detach().cpu().numpy()
    rotation=model._rotation.detach().cpu().numpy()
    attributes=np.concatenate((xyz,normals,dc,rest,opacity,scale,rotation),axis=1)
    elements=structured_rows(attributes,model.construct_list_of_attributes())
    PlyData([PlyElement.describe(elements,'vertex')]).write(path)
