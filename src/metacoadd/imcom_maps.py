import numpy as np


def get_combine_func(combine_func_name):
    """
    Get the combine function based on the name.
    This is used to combine the pixel values of the IMCOM maps for each object.

    Parameters
    ----------
    combine_func_name : str
        The name of the combine function. Must be one of ['mean', 'median',
        'sum', 'min', 'max'].

    Returns
    -------
    combine_func : function
        The combine function corresponding to the name.
    """
    if combine_func_name == "mean":
        return np.mean
    elif combine_func_name == "median":
        return np.median
    elif combine_func_name == "sum":
        return np.sum
    elif combine_func_name == "min":
        return np.min
    elif combine_func_name == "max":
        return np.max
    else:
        raise ValueError(
            f"Combine function {combine_func_name} not recognized. Must be in "
            "['mean', 'median', 'sum', 'min', 'max']."
        )


def get_stat_func(stat_func_name):
    """
    Get the statistical function based on the name.
    This is used to compute statistics for the IMCOM maps for each object.

    Parameters
    ----------
    stat_func_name : str
        The name of the statistical function. Must be one of ['std', 'var',
        'mad'].

    Returns
    -------
    stat_func : function
        The statistical function corresponding to the name.
    """
    if stat_func_name == "std":
        return np.std
    elif stat_func_name == "var":
        return np.var
    elif stat_func_name == "mad":
        return lambda x: np.median(np.abs(x - np.median(x)))
    else:
        raise ValueError(
            f"Stat function {stat_func_name} not recognized. Must be in "
            "['std', 'var', 'mad']."
        )


def get_imcom_map_cat_dtype(imcom_map_config, nband):
    """
    Get the dtype for the output catalog of IMCOM maps.

    Parameters
    ----------
    imcom_map_config : dict
        The configuration dictionary for the IMCOM maps. Must contain a
        'layers' key with a dictionary of map names and their corresponding
        combine and stat functions.
    nband : int
        The number of bands in the input data.

    Returns
    -------
    dtype : list of tuples
        The dtype for the output catalog of IMCOM maps. Each tuple contains the
        name of the field and its corresponding numpy dtype.
    """
    dtype = []
    for i in range(nband):
        for map_name in imcom_map_config:
            combine_func_name = imcom_map_config[map_name].get(
                "combine", "mean"
            )
            stat_func_name = imcom_map_config[map_name].get("stat", "std")
            dtype += [
                (f"IMCOM_{map_name}_{combine_func_name}_{i}", np.float64),
                (f"IMCOM_{map_name}_{stat_func_name}_{i}", np.float64),
            ]
    return dtype


def get_output_imcom_map_cat(n_obj, imcom_map_config, nband):
    """
    Get the output catalog for the IMCOM maps.

    Parameters
    ----------
    n_obj : int
        The number of objects in the catalog.
    imcom_map_config : dict
        The configuration dictionary for the IMCOM maps.
    nband : int
        The number of bands in the input data.

    Returns
    -------
    out : numpy.ndarray
        The output catalog for the IMCOM maps.
    """
    CAT_DTYPE = get_imcom_map_cat_dtype(imcom_map_config, nband)
    out = np.array(
        list(map(tuple, np.ones((len(CAT_DTYPE), n_obj)).T * -10.0)),
        dtype=CAT_DTYPE,
    )
    return out


def extract_imcom_maps(
    mbobs,
    sep_cat,
    seg_map,
    config,
):
    """
    Extract the IMCOM maps for each object in the catalog.

    Parameters
    ----------
    mbobs : ngmix.MultiBandObsList
        The MultiBandObsList containing the observations for each band.
    sep_cat : numpy.ndarray
        The catalog of objects from SExtractor.
    seg_map : numpy.ndarray
        The segmentation map from SExtractor.
    config : dict
        The configuration dictionary for the IMCOM maps. Must contain a
        'layers' key with a dictionary of map names and their corresponding
        combine and stat functions.

    Returns
    -------
    output_cat : numpy.ndarray
        The output catalog containing the extracted IMCOM maps for each object.
    """
    n_band = len(mbobs)

    imcom_map_config = config["layers"]
    output_cat = get_output_imcom_map_cat(
        len(sep_cat), imcom_map_config, n_band
    )

    n_obj = len(sep_cat)
    seg_ids = np.arange(1, n_obj + 1, dtype=np.int32)
    flat_seg_map = seg_map.ravel()

    for map_name in imcom_map_config:
        combine_func_name = imcom_map_config[map_name].get("combine", "mean")
        combine_func = get_combine_func(combine_func_name)
        stat_func_name = imcom_map_config[map_name].get("stat", "std")
        stat_func = get_stat_func(stat_func_name)
        for band_ind in range(n_band):
            obs = mbobs[band_ind][0]
            if not hasattr(obs, map_name.lower()):
                raise ValueError(
                    f"IMCOM layer {map_name} was not propagated to the "
                    "observation. Make sure that the observation is "
                    "constructed with the correct IMCOM maps."
                )
            flat_imcom_map = getattr(obs, map_name.lower()).ravel()
            for i, seg_id_tmp in enumerate(seg_ids):
                seg_pix = flat_seg_map == seg_id_tmp
                pix_vals = flat_imcom_map[seg_pix]
                output_cat[f"IMCOM_{map_name}_{combine_func_name}_{band_ind}"][
                    i
                ] = combine_func(pix_vals)
                output_cat[f"IMCOM_{map_name}_{stat_func_name}_{band_ind}"][
                    i
                ] = stat_func(pix_vals)

    return output_cat
