import tempfile
import os
import subprocess
import shutil

import numpy as np

from astropy.io import fits


def reproject_swarp(
    input_list,
    coadd_center,
    coadd_size,
    coadd_scale,
    image_kinds,
    swarp_config,
):
    """Run SWARP on a directory of images.

    Parameters:
        coaddimage (str):
        run_dir (str): Directory containing the input images.

    Returns:
        img
        wght
        header
    """

    # Make working dir
    if "run_dir" not in swarp_config or not os.path.exists(
        swarp_config["run_dir"]
    ):
        tmp_dir = tempfile.gettempdir()
    else:
        tmp_dir = swarp_config["run_dir"]

    # Make resamp dir
    # run_dir = os.path.join(tmp_dir, "resamp")
    run_dir = tempfile.TemporaryDirectory(
        dir=os.path.expandvars(tmp_dir),
        prefix="resamp-",
    ).name
    os.makedirs(run_dir, exist_ok=True)

    # Set input images
    exps, exp_wcs = input_list
    header = exp_wcs.to_header(relax=True)
    exp_dict = {kind: exps[i] for i, kind in enumerate(image_kinds)}
    foot_in = np.ones(exps[0].shape)
    if "weight" not in image_kinds:
        image_kinds.append("footprint")
        exp_dict["footprint"] = foot_in

    # Run SWARP for each image kind
    out_imgs = []
    for kind in image_kinds:
        weight_path = None
        if kind == "weight":
            weight_path = os.path.join(run_dir, f"{kind}.fits")
            save_image(exp_dict[kind], header, weight_path)
            img_path = os.path.join(run_dir, "footprint.fits")
            save_image(foot_in, header, img_path)
        else:
            img_path = os.path.join(run_dir, f"{kind}.fits")
            save_image(exp_dict[kind], header, img_path)
        resamp_img, resamp_weight = run_swarp(
            img_path,
            swarp_config,
            coadd_center,
            coadd_size,
            coadd_scale,
            run_dir,
            weight_path=weight_path,
        )
        if kind == "weight":
            out_imgs.append(resamp_weight)
            footprint = resamp_img
        elif kind == "footprint":
            footprint = resamp_img
        else:
            out_imgs.append(resamp_img)

    # Remove temporary directory
    shutil.rmtree(run_dir, ignore_errors=True)

    return np.stack(out_imgs), footprint


def run_swarp(
    img_path,
    swarp_config,
    coadd_center,
    coadd_size,
    coadd_scale,
    output_dir,
    weight_path=None,
):
    # Build command
    cmd = (
        f"{swarp_config['exec']} "
        f"{img_path} "
        "-RESCALE_WEIGHTS N "
        "-COMBINE N "
        f"-CENTER {coadd_center.ra.deg},{coadd_center.dec.deg} "
        f"-IMAGE_SIZE {coadd_size[0]},{coadd_size[1]} "
        "-PIXELSCALE_TYPE MANUAL "
        f"-PIXEL_SCALE {coadd_scale} "
        f"-RESAMPLE_DIR {output_dir} "
        f"-RESAPLING_TYPE {swarp_config['resamp_method'].upper()} "
        "-FSCALASTRO_TYPE NONE "
        "-FSCALE_KEYWORD NONE "
        "-GAIN_KEYWORD NONE "
        "-SATLEV_KEYWORD NONE "
        "-SUBTRACT_BACK N "
        "-DELETE_TMPFILES N "
        "-WRITE_XML N "
        "-NTHREADS 1 "
        "-VERBOSE_TYPE QUIET "
    )

    if weight_path is not None:
        cmd += "-WEIGHT_TYPE MAP_WEIGHT "
        cmd += f"-WEIGHT_IMAGE {weight_path} "

    # Run SWARP
    subprocess.Popen(
        cmd.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE
    ).wait()

    # Get output
    img_out_path = os.path.splitext(img_path)[0] + ".resamp.fits"
    img = fits.getdata(img_out_path, 0, memmap=False)
    if weight_path is not None:
        wght_out_path = os.path.splitext(img_path)[0] + ".resamp.weight.fits"
        wght = fits.getdata(wght_out_path, 0, memmap=False)
    else:
        wght = None
    return img, wght


def save_image(arr, header, path):
    hdu = fits.PrimaryHDU(arr, header)
    hdu.writeto(path, overwrite=True)
