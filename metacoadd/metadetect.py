import re

import numpy as np

import galsim

import ngmix
from ngmix.metacal.convenience import (
    _replace_image_with_noise,
    _rotate_obs_image_square,
    _doadd_single_obs,
)

from .metacal_oversampling import MetacalFitGaussPSFUnderRes
from .detect import get_cutout_size, get_cutout, get_cat, DET_CAT_DTYPE
from .uberseg import fast_uberseg
from .moments.galsim_admom import GAdmomFitter


_SHAPE_CAT_DTYPE = [
    ("wmom_flags", np.int32),
    ("wmom_nimage", np.int32),
    ("wmom_dx", np.float64),
    ("wmom_dy", np.float64),
    # ("wmom_flux", np.float64),
    # ("wmom_flux_err", np.float64),
    ("wmom_T", np.float64),
    ("wmom_T_err", np.float64),
    ("wmom_Tr", np.float64),
    ("wmom_Tpsf", np.float64),
    ("wmom_rho4", np.float64),
    ("wmom_rho4_err", np.float64),
    ("wmom_s2n", np.float64),
    ("wmom_e1", np.float64),
    ("wmom_e2", np.float64),
    ("wmom_e1err", np.float64),
    ("wmom_e2err", np.float64),
    ("wmom_g1", np.float64),
    ("wmom_g2", np.float64),
]


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

    # @profile
    def __init__(
        self,
        mbobs,
        rng,
        coadd_method="weighted",
        step=0.01,
        types=["1m", "1p", "2m", "2p", "noshear"],
        psf_true=None,
    ):
        self.rng = rng
        self.mcal_config = {
            "step": step,
            "types": types,
            "psf": "fitgauss_UR",
            "use_noise_image": True,
            "fixnoise": True,
        }

        # # Set observations
        self.mbobs = mbobs

        # Set mcal shear dict
        self._shear_dict = {
            "1m": galsim.Shear(g1=-step, g2=0),
            "1p": galsim.Shear(g1=step, g2=0),
            "2m": galsim.Shear(g1=0, g2=-step),
            "2p": galsim.Shear(g1=0, g2=step),
            "noshear": galsim.Shear(g1=0, g2=0),
        }
        self._true_psf = psf_true

    # @profile
    def go(
        self,
    ):
        """
        Run the coaddition process.
        """

        # mcal_mbobs = self.get_mcal_(self.mcal_config["types"])
        self._init_metacal()

        final_cat = {}
        for mcal_key in self.mcal_config["types"]:
            mb_obs = self.get_mcal(mcal_key)

            all_sep_cat, seg_map = self.get_cat(mb_obs, do_multiband=True)

            all_shape_cat = self.get_shape_cat(
                mb_obs,
                all_sep_cat,
                seg_map,
                mcal_key,
            )

            final_cat[mcal_key] = self.build_output_cat(
                all_sep_cat, all_shape_cat
            )

        return final_cat

    def _init_metacal(self):
        # mbobs = exp2obs(
        #     self.coaddimage.mb_explist,
        #     self.psf_mb_explist,
        #     use_resamp=False,
        # )
        self.mcal_makers = []
        for i, obs_list in enumerate(self.mbobs):
            self.mcal_makers.append([])
            for j, obs in enumerate(obs_list):
                obs_rng = np.random.RandomState(self.rng.randint(2**32))
                if self.mcal_config["fixnoise"]:
                    if self.mcal_config["use_noise_image"]:
                        noise_obs = _replace_image_with_noise(obs)
                    else:
                        noise_obs = ngmix.simobs.simulate_obs(
                            gmix=None, obs=obs, rng=obs_rng
                        )
                    _rotate_obs_image_square(noise_obs, k=1)
                mcal_maker = MetacalFitGaussPSFUnderRes(
                    obs,
                    self.mcal_config["step"],
                    obs_rng,
                )
                if self.mcal_config["fixnoise"]:
                    mcal_maker_noise = MetacalFitGaussPSFUnderRes(
                        noise_obs,
                        self.mcal_config["step"],
                        obs_rng,
                    )
                else:
                    mcal_maker_noise = None
                self.mcal_makers[i].append(
                    {"img": mcal_maker, "noise": mcal_maker_noise}
                )

    def get_mcal(self, mcal_type):
        mcal_mbobs = ngmix.MultiBandObsList()
        for i, obs_list in enumerate(self.mbobs):
            mcal_obs_list = ngmix.ObsList()
            for j, obs in enumerate(obs_list):
                mcal_obs = self.mcal_makers[i][j]["img"].get_obs_galshear(
                    mcal_type
                )
                if self.mcal_config["fixnoise"]:
                    noise_obs = self.mcal_makers[i][j][
                        "noise"
                    ].get_obs_galshear(mcal_type)
                    _rotate_obs_image_square(noise_obs, k=3)

                    _doadd_single_obs(mcal_obs, noise_obs)

                mcal_obs_list.append(mcal_obs)
            mcal_mbobs.append(mcal_obs_list)
        return mcal_mbobs

    def get_mcal_(self, mcal_types):
        if self._do_psf_sed_corr:
            self._psf_corr_dict = []

        if self.mcal_config["use_noise_image"]:
            noise_mbobs = _replace_image_with_noise(self.mbobs)
        else:
            noise_obs = ngmix.simobs.simulate_obs(
                gmix=None, obs=self.mbobs, rng=self.rng
            )
        _rotate_obs_image_square(noise_mbobs, k=1)

        mcal_mbobs = {}
        for mcal_type in mcal_types:
            mcal_mbobs_ = ngmix.MultiBandObsList()
            for i, obs_list in enumerate(self.mbobs):
                mcal_obs_list = ngmix.ObsList()
                if self._do_psf_sed_corr and mcal_type == "noshear":
                    self._psf_corr_dict.append([])
                for j, obs in enumerate(obs_list):
                    obs_rng = np.random.RandomState(self.rng.randint(2**32))

                    mcal_maker = MetacalFitGaussPSFUnderRes(
                        obs,
                        self.mcal_config["step"],
                        obs_rng,
                    )
                    mcal_maker_noise = MetacalFitGaussPSFUnderRes(
                        noise_mbobs[i][j],
                        self.mcal_config["step"],
                        obs_rng,
                    )

                    if self._do_psf_sed_corr and mcal_type == "noshear":
                        self._psf_corr_dict[i].append(
                            {
                                "psf_int_nopix_inv": mcal_maker.psf_int_nopix_inv,
                            }
                        )

                    mcal_obs = mcal_maker.get_obs_galshear(mcal_type)
                    noise_obs = mcal_maker_noise.get_obs_galshear(mcal_type)
                    _rotate_obs_image_square(noise_obs, k=3)

                    _doadd_single_obs(mcal_obs, noise_obs)

                    mcal_obs_list.append(mcal_obs)
                mcal_mbobs_.append(mcal_obs_list)
            mcal_mbobs[mcal_type] = mcal_mbobs_

        # del mbobs
        return mcal_mbobs

    def get_cat(self, mb_obs, do_multiband=True):
        if do_multiband:
            cat, seg_map = get_cat(
                np.copy(mb_obs[0][0].image),
                np.copy(mb_obs[0][0].weight),
                thresh=1500,
                # thresh=1e3,
                wcs=None,
            )

        return cat, seg_map

    def get_shape_cat(
        self,
        in_mbobs,
        sep_cat,
        seg_map,
        mcal_key,
        do_uberseg=False,
    ):
        # scale_sq = self.coaddimage.coadd_pixel_scale**2
        scale_sq = in_mbobs[0][0].jacobian.area

        # fitter = ReGaussFitter(guess_fwhm=0.6)
        # fitter = GAdmomFitter(guess_fwhm=0.6)
        fitter = ngmix.gaussmom.GaussMom(fwhm=1.2)
        psf_fitter = GAdmomFitter(guess_fwhm=0.6)

        self.all_obs = []
        all_shape_cat = []
        T_psf_avg = 0.0
        W_psf = 0.0
        k = 0
        for det_obj in sep_cat:
            # cutout_size = np.int64(
            #     np.ceil(np.sqrt(det_obj["npix"] / np.pi) * 2)
            # )
            cutout_size = get_cutout_size(
                det_obj["xx"],
                det_obj["xy"],
                det_obj["yy"],
                n_sigma=5.0,
            )
            cutout_size = np.int64(np.ceil(cutout_size))
            if cutout_size % 2 == 0:
                cutout_size += 1
            # cutout_size = max(31, cutout_size)
            cutout_size = 31

            mb_obs = ngmix.MultiBandObsList()
            for i, obslist in enumerate(in_mbobs):
                obs_list = ngmix.ObsList()
                for j, obs in enumerate(obslist):
                    x = det_obj["x"]
                    y = det_obj["y"]
                    img_pos = galsim.PositionD(x, y)

                    img, dx, dy = get_cutout(
                        np.copy(obs.image), img_pos.x, img_pos.y, cutout_size
                    )
                    wgt, _, _ = get_cutout(
                        np.copy(obs.weight), img_pos.x, img_pos.y, cutout_size
                    )
                    noise, _, _ = get_cutout(
                        np.copy(obs.noise), img_pos.x, img_pos.y, cutout_size
                    )
                    if do_uberseg:
                        seg, _, _ = get_cutout(
                            seg_map, det_obj["x"], det_obj["y"], cutout_size
                        )
                        wgt = fast_uberseg(seg, wgt, det_obj["number"])

                    jac = ngmix.Jacobian(
                        row=dx,
                        col=dy,
                        dudrow=obs.jacobian.get_dudrow(),
                        dudcol=obs.jacobian.get_dudcol(),
                        dvdrow=obs.jacobian.get_dvdrow(),
                        dvdcol=obs.jacobian.get_dvdcol(),
                    )

                    # Fit the PSF
                    psf_res = psf_fitter.go(obs.psf)
                    w_psf = np.median(wgt)
                    T_psf_avg += psf_res["T"] * w_psf
                    W_psf += w_psf

                    newobs = ngmix.Observation(
                        image=img,
                        weight=wgt,
                        jacobian=jac,
                        noise=noise,
                        psf=obs.psf,
                    )
                    obs_list.append(newobs)
                mb_obs.append(obs_list)

            # Get guess from sep moments
            guess = np.array(
                [
                    0.0,
                    0.0,
                    det_obj["xx"] * scale_sq,
                    det_obj["xy"] * scale_sq,
                    det_obj["yy"] * scale_sq,
                ]
            )
            self.all_obs.append(mb_obs)

            res = fitter.go(newobs)  # , guess=guess)
            res = {k: v for k, v in res.items()}
            # res["g1"] = res["g"][0]
            # res["g2"] = res["g"][1]
            res["g1"] = res["e"][0]
            res["g2"] = res["e"][1]
            res["Tpsf"] = T_psf_avg / W_psf

            all_shape_cat.append(res)
            k += 1
        return all_shape_cat

    def build_output_cat(self, all_sep_cat, all_shape_cat):
        SHAPE_CAT_DTYPE = _SHAPE_CAT_DTYPE.copy()
        for i in range(len(self.mbobs)):
            SHAPE_CAT_DTYPE.append(("wmom_flux_" + str(i), np.float64))
            SHAPE_CAT_DTYPE.append(("wmom_flux_err_" + str(i), np.float64))
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
                    shape_key = key.split("wmom_")[1]
                    if shape_key == "dx":
                        final_cat[i][key] = all_shape_cat[i]["pars"][0]
                    elif shape_key == "dy":
                        final_cat[i][key] = all_shape_cat[i]["pars"][1]
                    elif "flux_err" in shape_key:
                        flux_ind = int(re.findall(r"\d+", shape_key)[0])
                        # final_cat[i][key] = all_shape_cat[i]["flux_err"][
                        #     flux_ind
                        # ]
                        final_cat[i][key] = all_shape_cat[i]["flux_err"]
                    elif "flux" in shape_key:
                        flux_ind = int(re.findall(r"\d+", shape_key)[0])
                        # final_cat[i][key] = all_shape_cat[i]["flux"][flux_ind]
                        final_cat[i][key] = all_shape_cat[i]["flux"]
                    else:
                        final_cat[i][key] = all_shape_cat[i][shape_key]
                except:
                    continue
        return final_cat
