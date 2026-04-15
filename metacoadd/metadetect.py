import re
import gc
from copy import deepcopy

import numpy as np

import ngmix
from ngmix.metacal.convenience import (
    _replace_image_with_noise,
    _rotate_obs_image_square,
    _doadd_single_obs,
)

from .metacal_new import MetacalHandler, MetacalHandlerTest
from .detect import get_stamp_mbobs, get_cat, DET_CAT_DTYPE
from .fitting import get_fitters, get_gauss_psf_runner
from .fitters.fourier_fitting_nb import estimate_noise_ps_analytic


def get_shape_cat_dtype(runner_name):
    dtype = [
        (f"{runner_name}_flags", np.int32),
        (f"{runner_name}_nimage", np.int32),
        (f"{runner_name}_dx", np.float64),
        (f"{runner_name}_dy", np.float64),
        (f"{runner_name}_T", np.float64),
        (f"{runner_name}_T_err", np.float64),
        (f"{runner_name}_Tr", np.float64),
        (f"{runner_name}_Tpsf", np.float64),
        (f"{runner_name}_rho4", np.float64),
        (f"{runner_name}_rho4_err", np.float64),
        (f"{runner_name}_s2n", np.float64),
        (f"{runner_name}_e1", np.float64),
        (f"{runner_name}_e2", np.float64),
        (f"{runner_name}_e1err", np.float64),
        (f"{runner_name}_e2err", np.float64),
        (f"{runner_name}_g1", np.float64),
        (f"{runner_name}_g2", np.float64),
    ]
    return dtype


class MetaDetect:
    """MetaCoadd

    The class run the metacoaddition process.
    The inputs are multi-band multi-epoch images and PSFs.
    The processing goes as follow:
    0. Resize the exposures to fit within the planned coadded region.
    1. run metacalibration (deconvolutioon | shear | re-convolution) on each input single exposures.
    2. Re-sample each sheared single exposures to the coadd WCS.
    3. Coadd the exposures together (can do a multi-band coadd if the PSF is homogenized through all bands).
    4. Run dectection on each sheared coadded image.
    5. Run multi-band multi-epoch shape measurement at each detected object position.

    Parameters
    ----------
    """

    def __init__(
        self,
        rng,
        step=0.01,
        types=["1m", "1p", "2m", "2p", "noshear"],
        detect_thresh=1500,
        coadd_multiband=True,
        models=None,
        fwhm=None,
        stamp_size=101,
        mcal_config={},
        test_fixnoise=False,
        noise_boost_factor=1.0,
        path_mcal_transfer_func="/Users/aguinot/Documents/roman/test_metadetect/metacal_transfer_func.npy",
    ):
        self.rng = rng
        self.mcal_config = {
            "step": step,
            "types": types,
            "psf": "fitgauss",
            "use_noise_image": True,
            "fixnoise": True,
            "has_pixel": False,
        }
        if not isinstance(mcal_config, dict):
            raise ValueError("mcal_config must be a dictionary")
        self.mcal_config.update(mcal_config)

        self._detect_thresh = detect_thresh
        self._coadd_multiband = coadd_multiband
        self._models = models
        self._fwhm = fwhm
        self._stamp_size = stamp_size
        self.test_fixnoise = test_fixnoise
        self.noise_boost_factor = noise_boost_factor
        self.path_mcal_transfer_func = path_mcal_transfer_func

    def go(
        self,
        mb_obs,
    ):
        """
        Run metadetect.
        """

        if isinstance(mb_obs, ngmix.MultiBandObsList):
            nband = len(mb_obs)
            scale = mb_obs[0][0].jacobian.get_scale()
        else:
            raise ValueError(
                "mb_obs must be an instance of ngmix.MultiBandObsList"
            )
        self.gal_runners = get_fitters(
            models=self._models,
            fwhm=self._fwhm,
            rng=self.rng,
            nband=nband,
            scale=scale,
            stamp_size=self._stamp_size,
        )
        self._init_metacal(mb_obs)
        if self.test_fixnoise:
            self._set_power_spectrum_pseudo_fixnoise()

        final_cat = {}
        for mcal_key in self.mcal_config["types"]:
            mcal_mbobs = self.mcal_mbobs[mcal_key]

            T_psf = self.get_T_psf(mcal_mbobs)

            all_sep_cat, seg_map = self.get_cat(mcal_mbobs)

            if not self.test_fixnoise:
                self._set_power_spectrum(mcal_mbobs)

            all_shape_cat = self.get_shape_cat(
                mcal_mbobs,
                all_sep_cat,
                seg_map,
                T_psf,
            )
            self.all_shape_cat = all_shape_cat

            final_cat[mcal_key] = self.build_output_cat(
                mcal_mbobs, all_sep_cat, all_shape_cat
            )

            del mcal_mbobs, all_sep_cat, seg_map, all_shape_cat
            gc.collect()

        return final_cat

    def _init_metacal(self, mb_obs):
        if not self.test_fixnoise:
            mcal_handler = MetacalHandler(
                rng=self.rng,
                fixnoise=self.mcal_config["fixnoise"],
                use_noise_image=self.mcal_config["use_noise_image"],
                mcal_config={
                    "step": self.mcal_config["step"],
                    "has_pixel": self.mcal_config["has_pixel"],
                },
            )
        else:
            mcal_handler = MetacalHandlerTest(
                rng=self.rng,
                fixnoise=self.mcal_config["fixnoise"],
                use_noise_image=self.mcal_config["use_noise_image"],
                mcal_config={
                    "step": self.mcal_config["step"],
                    "has_pixel": self.mcal_config["has_pixel"],
                },
                noise_boost_factor=self.noise_boost_factor,
            )
        self.mcal_mbobs = mcal_handler.get_all(
            mb_obs, self.mcal_config["types"]
        )

    def _set_power_spectrum(self, mb_obs):
        do_ps = False
        for model_name in self.gal_runners:
            if "fourier" in model_name:
                do_ps = True
                break
        if not do_ps:
            return

        for obslist in mb_obs:
            for obs in obslist:
                if hasattr(obs, "ps"):
                    continue
                ps = estimate_noise_ps_analytic(
                    obs.noise,
                    101,
                )
                obs.ps = ps

    def _set_power_spectrum_pseudo_fixnoise(self):
        do_ps = False
        for model_name in self.gal_runners:
            if "fourier" in model_name:
                do_ps = True
                break
        if not do_ps:
            return

        type_to_ind = {
            "noshear": 0,
            "1p": 1,
            "1m": 2,
            "2p": 3,
            "2m": 4,
        }

        all_shear_T = np.load(self.path_mcal_transfer_func)
        ps_orig = None
        mcal_ps = {}
        for mcal_key, mcal_mbobs in self.mcal_mbobs.items():
            mcal_ps[mcal_key] = []
            for band_ind, obslist in enumerate(mcal_mbobs):
                mcal_ps[mcal_key].append([])
                for list_ind, obs in enumerate(obslist):
                    ps = estimate_noise_ps_analytic(
                        obs.noise,
                        101,
                    )
                    mcal_ps[mcal_key][-1].append(
                        ps / self.noise_boost_factor**2
                    )
                    # obs.ps = ps / self.noise_boost_factor**2
                    # if ps_orig is None:
                    #     ps_orig = estimate_noise_ps_analytic(
                    #         orig_mb_obs[band_ind][list_ind].noise,
                    #         101,
                    #     )
                    # if mcal_key == "noshear":
                    #     obs.ps = (
                    #         ps_orig * all_shear_T[type_to_ind["noshear"], :, :]
                    #     )
                    # elif "1" in mcal_key:
                    #     obs.ps = [
                    #         (ps_orig * all_shear_T[type_to_ind["1p"], :, :]),
                    #         (ps_orig * all_shear_T[type_to_ind["1m"], :, :]),
                    #     ]
                    # elif "2" in mcal_key:
                    #     obs.ps = [
                    #         (ps_orig * all_shear_T[type_to_ind["2p"], :, :]),
                    #         (ps_orig * all_shear_T[type_to_ind["2m"], :, :]),
                    #     ]

        # 1p
        if "1p" in self.mcal_config["types"]:
            mcal_mbobs = self.mcal_mbobs["1p"]
            for band_ind, obslist in enumerate(mcal_mbobs):
                for list_ind, obs in enumerate(obslist):
                    obs.ps = [
                        mcal_ps["1p"][band_ind][list_ind],
                        mcal_ps["1m"][band_ind][list_ind],
                    ]
        # 1m
        if "1m" in self.mcal_config["types"]:
            mcal_mbobs = self.mcal_mbobs["1m"]
            for band_ind, obslist in enumerate(mcal_mbobs):
                for list_ind, obs in enumerate(obslist):
                    obs.ps = [
                        mcal_ps["1p"][band_ind][list_ind],
                        mcal_ps["1m"][band_ind][list_ind],
                    ]
        # 2p
        if "2p" in self.mcal_config["types"]:
            mcal_mbobs = self.mcal_mbobs["2p"]
            for band_ind, obslist in enumerate(mcal_mbobs):
                for list_ind, obs in enumerate(obslist):
                    obs.ps = [
                        mcal_ps["2p"][band_ind][list_ind],
                        mcal_ps["2m"][band_ind][list_ind],
                    ]
        # 2m
        if "2m" in self.mcal_config["types"]:
            mcal_mbobs = self.mcal_mbobs["2m"]
            for band_ind, obslist in enumerate(mcal_mbobs):
                for list_ind, obs in enumerate(obslist):
                    obs.ps = [
                        mcal_ps["2p"][band_ind][list_ind],
                        mcal_ps["2m"][band_ind][list_ind],
                    ]
        # noshear
        if "noshear" in self.mcal_config["types"]:
            mcal_mbobs = self.mcal_mbobs["noshear"]
            for band_ind, obslist in enumerate(mcal_mbobs):
                for list_ind, obs in enumerate(obslist):
                    obs.ps = mcal_ps["noshear"][band_ind][list_ind]

    def get_T_psf(self, mb_obs):

        psf_runner = get_gauss_psf_runner(self.rng)

        T_psf_avg = 0.0
        W_psf = 0.0
        for obslist in mb_obs:
            for obs in obslist:
                psf_res = psf_runner.go(obs.psf)
                w_psf = np.median(obs.weight[obs.weight != 0])
                T_psf_avg += psf_res["T"] * w_psf
                W_psf += w_psf
        return T_psf_avg / W_psf

    def get_coadd_multiband(self, mb_obs):

        img_final = np.zeros_like(mb_obs[0][0].image)
        weight_final = np.zeros_like(mb_obs[0][0].weight)
        for obslist in mb_obs:
            img_final += obslist[0].image * obslist[0].weight
            weight_final += obslist[0].weight
        img_final[weight_final != 0] /= weight_final[weight_final != 0]

        return img_final, weight_final

    def get_cat(self, mb_obs):
        if self._coadd_multiband:
            img, weight = self.get_coadd_multiband(mb_obs)
        else:
            img = mb_obs[0][0].image
            weight = mb_obs[0][0].weight
        cat, seg_map = get_cat(
            img,
            weight,
            thresh=self._detect_thresh,
            wcs=None,
        )
        del img, weight
        gc.collect()

        return cat, seg_map

    def get_shape_cat(
        self,
        in_mbobs,
        sep_cat,
        seg_map,
        T_psf,
        do_uberseg=False,
    ):

        all_shape_cat = {name: [] for name in self.gal_runners}
        for obj_ind, det_obj in enumerate(sep_cat):
            cutout_size = self._stamp_size

            mb_obs = get_stamp_mbobs(
                in_mbobs,
                det_obj,
                min_stamp_size=cutout_size,
                max_stamp_size=cutout_size,
                do_uberseg=do_uberseg,
                seg_map=seg_map,
            )

            for name, runner in self.gal_runners.items():
                res_ = runner.go(mb_obs)
                res = {k: v for k, v in res_.items()}
                if "g" in res:
                    res["g1"] = res["g"][0]
                    res["g2"] = res["g"][1]
                elif "e" in res:
                    res["g1"] = res["e"][0]
                    res["g2"] = res["e"][1]
                res["Tpsf"] = T_psf

                all_shape_cat[name].append(res)
        return all_shape_cat

    def build_output_cat(self, mb_obs, all_sep_cat, all_shape_cat):
        SHAPE_CAT_DTYPE = []
        for name in self.gal_runners:
            SHAPE_CAT_DTYPE += get_shape_cat_dtype(name)
            for i in range(len(mb_obs)):
                SHAPE_CAT_DTYPE.append((f"{name}_flux_" + str(i), np.float64))
                SHAPE_CAT_DTYPE.append(
                    (f"{name}_flux_err_" + str(i), np.float64)
                )
        final_cat_dtype = DET_CAT_DTYPE + SHAPE_CAT_DTYPE
        final_cat = np.zeros(
            len(all_sep_cat),
            dtype=final_cat_dtype,
        )
        for i, sep_obj in enumerate(all_sep_cat):
            for key in sep_obj.dtype.names:
                final_cat[i][key] = sep_obj[key]
            for key in np.dtype(SHAPE_CAT_DTYPE).names:
                try:
                    runner_name = key.split("_")[0]
                    if runner_name == "fourier":
                        runner_name = "_".join(key.split("_")[:2])
                    shape_key = key.split(f"{runner_name}_")[1]
                    if shape_key == "dx":
                        final_cat[i][key] = all_shape_cat[runner_name][i][
                            "pars"
                        ][0]
                    elif shape_key == "dy":
                        final_cat[i][key] = all_shape_cat[runner_name][i][
                            "pars"
                        ][1]
                    elif "flux_err" in shape_key:
                        flux_ind = int(re.findall(r"\d+", shape_key)[0])
                        final_cat[i][key] = np.atleast_1d(
                            all_shape_cat[runner_name][i]["flux_err"]
                        )[flux_ind]
                    elif "flux" in shape_key:
                        flux_ind = int(re.findall(r"\d+", shape_key)[0])
                        final_cat[i][key] = np.atleast_1d(
                            all_shape_cat[runner_name][i]["flux"]
                        )[flux_ind]
                    else:
                        final_cat[i][key] = all_shape_cat[runner_name][i][
                            shape_key
                        ]
                except:
                    continue
        return final_cat
