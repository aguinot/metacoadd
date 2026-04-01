import re

import numpy as np

import ngmix
from .metacal_new import MetacalFitGaussPSF
from .detect import get_stamp_mbobs, get_cat, DET_CAT_DTYPE


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
        mbobs,
        rng,
        step=0.01,
        types=["1m", "1p", "2m", "2p", "noshear"],
        detect_thresh=1500,
        coadd_multiband=True,
        gal_runners=None,
        psf_runner=None,
        mcal_config={},
    ):
        self.rng = rng
        self.mcal_config = {
            "step": step,
            "types": types,
            # "psf": "fitgauss_UR",
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

        if isinstance(gal_runners, dict):
            for _, runner in gal_runners.items():
                if not isinstance(runner, ngmix.runners.RunnerBase):
                    raise ValueError(
                        "All gal_runners must be instances of ngmix.runners.RunnerBase"
                    )
        elif isinstance(gal_runners, ngmix.runners.RunnerBase):
            gal_runners = {"default": gal_runners}
        else:
            raise ValueError(
                "gal_runners must be either a dict of ngmix.runners.RunnerBase "
                "or a single ngmix.runners.RunnerBase instance. "
                f"Got {type(gal_runners)}"
            )
        self.gal_runners = gal_runners
        if not isinstance(psf_runner, ngmix.runners.RunnerBase):
            raise ValueError(
                "psf_runner must be an instance of ngmix.runners.RunnerBase. "
                f"Got {type(psf_runner)}"
            )
        self.psf_runner = psf_runner

        # # Set observations
        self.mbobs = mbobs

    def go(
        self,
    ):
        """
        Run metadetect.
        """

        self._init_metacal()

        final_cat = {}
        for mcal_key in self.mcal_config["types"]:
            mb_obs = self.get_mcal(mcal_key)
            # mb_obs = mcal_mbobs[mcal_key]

            T_psf = self.get_T_psf(mb_obs)

            all_sep_cat, seg_map = self.get_cat(mb_obs)

            all_shape_cat = self.get_shape_cat(
                mb_obs,
                all_sep_cat,
                seg_map,
                mcal_key,
                T_psf,
            )

            final_cat[mcal_key] = self.build_output_cat(
                all_sep_cat, all_shape_cat
            )

            del mb_obs, all_sep_cat, seg_map, all_shape_cat

        return final_cat

    def _init_metacal(self):
        self.mcal_makers = []
        for i, obs_list in enumerate(self.mbobs):
            self.mcal_makers.append([])
            for j, obs in enumerate(obs_list):
                obs_rng = np.random.RandomState(self.rng.randint(2**32))
                mcal_maker = MetacalFitGaussPSF(
                    obs,
                    step=self.mcal_config["step"],
                    has_pixel=self.mcal_config["has_pixel"],
                    fixnoise=self.mcal_config["fixnoise"],
                    rng=obs_rng,
                )
                self.mcal_makers[i].append(mcal_maker)

    def get_mcal(self, mcal_type):
        mcal_mbobs = ngmix.MultiBandObsList()
        for i, obs_list in enumerate(self.mbobs):
            mcal_obs_list = ngmix.ObsList()
            for j, _ in enumerate(obs_list):
                mcal_obs = self.mcal_makers[i][j].get_obs_galshear(mcal_type)
                mcal_obs_list.append(mcal_obs)
            mcal_mbobs.append(mcal_obs_list)
        return mcal_mbobs

    def get_T_psf(self, mb_obs):

        T_psf_avg = 0.0
        W_psf = 0.0
        for obslist in mb_obs:
            for obs in obslist:
                psf_res = self.psf_runner.go(obs.psf)
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
            img = np.copy(mb_obs[0][0].image)
            weight = np.copy(mb_obs[0][0].weight)
        cat, seg_map = get_cat(
            img,
            weight,
            thresh=self._detect_thresh,
            wcs=None,
        )

        return cat, seg_map

    def get_shape_cat(
        self,
        in_mbobs,
        sep_cat,
        seg_map,
        T_psf,
        do_uberseg=False,
    ):

        self.all_obs = []
        all_shape_cat = {name: [] for name in self.gal_runners}
        k = 0
        for det_obj in sep_cat:
            cutout_size = 101

            mb_obs = get_stamp_mbobs(
                in_mbobs,
                det_obj,
                min_stamp_size=cutout_size,
                max_stamp_size=cutout_size,
                do_uberseg=do_uberseg,
                seg_map=seg_map,
            )

            self.all_obs.append(mb_obs)

            for name, runner in self.gal_runners.items():
                res = runner.go(mb_obs[0][0])
                res = {k: v for k, v in res.items()}
                if "g" in res:
                    res["g1"] = res["g"][0]
                    res["g2"] = res["g"][1]
                elif "e" in res:
                    res["g1"] = res["e"][0]
                    res["g2"] = res["e"][1]
                res["Tpsf"] = T_psf

                all_shape_cat[name].append(res)
            k += 1
        return all_shape_cat

    def build_output_cat(self, all_sep_cat, all_shape_cat):
        SHAPE_CAT_DTYPE = []
        for name in self.gal_runners:
            SHAPE_CAT_DTYPE += get_shape_cat_dtype(name)
            for i in range(len(self.mbobs)):
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
                        # final_cat[i][key] = all_shape_cat[runner_name][i]["flux_err"][
                        #     flux_ind
                        # ]
                        final_cat[i][key] = all_shape_cat[runner_name][i][
                            "flux_err"
                        ]
                    elif "flux" in shape_key:
                        flux_ind = int(re.findall(r"\d+", shape_key)[0])
                        # final_cat[i][key] = all_shape_cat[runner_name][i]["flux"][flux_ind]
                        final_cat[i][key] = all_shape_cat[runner_name][i][
                            "flux"
                        ]
                    else:
                        final_cat[i][key] = all_shape_cat[runner_name][i][
                            shape_key
                        ]
                except:
                    continue
        return final_cat
