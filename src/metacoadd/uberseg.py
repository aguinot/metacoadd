import numpy as np

from meds import _uberseg


def fast_uberseg(
    seg,
    weight,
    object_number,
):
    """Wrapper of the uber-segmentation from meds.

    Parameters.
    ----------
    seg : np.ndarray
        Segmentation map.
    weight : np.ndarray
        Weight map.
    object_number : int
        Object number to keep in the weight map.

    Returns
    -------
    weight : np.ndarray
        Weight map with only the object_number kept.

    """
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
