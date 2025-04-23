import numpy as np
import numba as nb

from meds import _uberseg

# import ctypes as ct

# libuberseg = ct.CDLL(
#     "/Users/aguinot/miniconda3/envs/roman_sim/lib/python3.12/site-packages/meds/_uberseg.cpython-312-darwin.so"
# )
# uberseg_tree = libuberseg.uberseg_tree
# uberseg_tree.argtypes = [
#     np.ctypeslib.ndpointer(dtype=np.int32),
#     np.ctypeslib.ndpointer(dtype=np.float32),
#     np.int32,
#     np.int32,
#     np.int32,
#     np.ctypeslib.ndpointer(dtype=np.int32),
#     np.ctypeslib.ndpointer(dtype=np.int32),
#     np.int32,
# ]
# uberseg_tree.restype = None


@nb.njit()
def uberseg(
    seg,
    weight,
    object_number,
):
    obj_inds = np.where(seg != 0)

    for i, row in enumerate(seg):
        for j, element in enumerate(row):
            obj_dists = (i - obj_inds[0]) ** 2 + (j - obj_inds[1]) ** 2
            ind_min = np.argmin(obj_dists)

            segval = seg[obj_inds[0][ind_min], obj_inds[1][ind_min]]
            if segval != object_number:
                weight[i, j] = 0.0

    return weight


# @nb.njit()
def fast_uberseg(
    seg,
    weight,
    object_number,
):
    obj_inds = np.where(seg != 0)

    Nx, Ny = seg.shape
    Ninds = len(obj_inds[0])
    seg = seg.astype(np.int32)
    weight = weight.astype(np.float32, copy=False)
    obj_inds_x = obj_inds[0].astype(np.int32, copy=False)
    obj_inds_y = obj_inds[1].astype(np.int32, copy=False)
    _uberseg.uberseg_tree(
        seg, weight, Nx, Ny, object_number, obj_inds_x, obj_inds_y, Ninds
    )

    return weight
