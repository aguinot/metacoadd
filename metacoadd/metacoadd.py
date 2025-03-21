import copy

import numpy as np

import galsim

import ngmix

from .exposure import CoaddImage, Exposure, ExpList, MultiBandExpList, exp2obs

from .metacal_oversampling import get_all_metacal
from .detect import get_cutout, get_cat, DET_CAT_DTYPE
from .moments.galsim_regauss import ReGaussFitter


TEST_METADETECT_CONFIG = {
    "model": "wmom",
    "weight": {
        "fwhm": 1.2,  # arcsec
    },
    "metacal": {
        "psf": "gauss",
        "types": ["noshear", "1p", "1m", "2p", "2m"],
        "use_noise_image": True,
    },
    "sx": {
        # in sky sigma
        # DETECT_THRESH
        "detect_thresh": 0.8,
        # Minimum contrast parameter for deblending
        # DEBLEND_MINCONT
        # 'deblend_cont': 0.00001,
        "deblend_cont": 0.01,
        # minimum number of pixels above threshold
        # DETECT_MINAREA: 6
        "minarea": 6,
        "filter_type": "conv",
        # 7x7 convolution mask of a gaussian PSF with FWHM = 3.0 pixels.
        "filter_kernel": [
            [
                0.004963,
                0.021388,
                0.051328,
                0.068707,
                0.051328,
                0.021388,
                0.004963,
            ],  # noqa: E501
            [
                0.021388,
                0.092163,
                0.221178,
                0.296069,
                0.221178,
                0.092163,
                0.021388,
            ],  # noqa: E501
            [
                0.051328,
                0.221178,
                0.530797,
                0.710525,
                0.530797,
                0.221178,
                0.051328,
            ],  # noqa: E501
            [
                0.068707,
                0.296069,
                0.710525,
                0.951108,
                0.710525,
                0.296069,
                0.068707,
            ],  # noqa: E501
            [
                0.051328,
                0.221178,
                0.530797,
                0.710525,
                0.530797,
                0.221178,
                0.051328,
            ],  # noqa: E501
            [
                0.021388,
                0.092163,
                0.221178,
                0.296069,
                0.221178,
                0.092163,
                0.021388,
            ],  # noqa: E501
            [
                0.004963,
                0.021388,
                0.051328,
                0.068707,
                0.051328,
                0.021388,
                0.004963,
            ],  # noqa: E501
        ],
    },
    "meds": {
        "min_box_size": 32,
        "max_box_size": 32,
        "box_type": "iso_radius",
        "rad_min": 4,
        "rad_fac": 2,
        "box_padding": 2,
    },
    # needed for PSF symmetrization
    "psf": {
        "model": "gauss",
        "ntry": 2,
        "lm_pars": {
            "maxfev": 2000,
            "ftol": 1.0e-5,
            "xtol": 1.0e-5,
        },
    },
    # check for an edge hit
    "bmask_flags": 2**30,
    "nodet_flags": 2**0,
}


_available_method = ["weighted"]


SHAPE_CAT_DTYPE = [
    ("regauss_flags", np.int32),
    ("regauss_nimage", np.int32),
    ("regauss_flux", np.float64),
    ("regauss_flux_err", np.float64),
    ("regauss_T", np.float64),
    ("regauss_T_err", np.float64),
    ("regauss_Tr", np.float64),
    ("regauss_Tpsf", np.float64),
    ("regauss_rho4", np.float64),
    ("regauss_rho4_err", np.float64),
    ("regauss_s2n", np.float64),
    ("regauss_e1", np.float64),
    ("regauss_e2", np.float64),
    ("regauss_e1err", np.float64),
    ("regauss_e2err", np.float64),
    ("regauss_g1", np.float64),
    ("regauss_g2", np.float64),
]


class SimpleCoadd:
    """SimpleCoadd

    This class handle the coaddition of an `Exposure` list.
    It will transform the coordinates and shifts each single exposures to match
    the coadd center.
    At the moment only weithed average coadding is handled.

    Args:
        coaddimage (metacoadd.CoaddImage): CoaddImage instance to stack.
        coadd_method (str, optional): Kind of stacking method to use. Only
        'weighted' is implemented. Defaults to 'weighted'.
            - `'weighted'`: Weighted average coadd.
    """

    def __init__(
        self,
        coaddimage,
        coadd_method="weighted",
        do_border=True,
        border_size=20,
    ):
        if isinstance(coaddimage, CoaddImage):
            self.coaddimage = coaddimage
        else:
            raise TypeError("coaddimage must be a metacoadd.CoaddImage.")

        if isinstance(coadd_method, str):
            if coadd_method in _available_method:
                self._coadd_method = coadd_method
            else:
                raise ValueError(
                    f"coadd_method must be in {_available_method}."
                )
        else:
            raise TypeError("coadd_method must be of type str.")

        # NOTE: Not sure if this should be accessible by the user
        self._do_border = do_border
        if self._do_border:
            self._border_size = border_size

    def go(self, **resamp_kwargs):
        """
        Run the coaddition process.
        """

        self.coaddimage.setup_coadd()
        for i, explist in enumerate(self.coaddimage.mb_explist):
            if len(explist) == 0:
                raise ValueError("No exposure find to make the coadd.")
            if not explist[0]._resamp:
                # raise ValueError('Exposure must be resampled first.')
                self.coaddimage.get_all_resamp_images(**resamp_kwargs)

            for exp in explist:
                all_stamp = self._process_one_exp(exp)

                # Check bounds, it should always pass. Just for safety.
                # We check only 'image' because it we always be there and the
                # property are shared with the other kind.
                b = all_stamp["image"].bounds & self.coaddimage.image[i].bounds
                if b.isDefined():
                    if self._coadd_method == "weighted":
                        # NOTE: check for the presence of a 'weight' for the
                        # weighted average coadding
                        self.coaddimage.image[i][b] += (
                            all_stamp["image"]
                            * all_stamp["weight"]
                            * all_stamp["border"]
                        )
                        if "noise" in list(all_stamp.keys()):
                            self.coaddimage.noise[i][b] += (
                                all_stamp["noise"]
                                * all_stamp["weight"]
                                * all_stamp["border"]
                            )
                        self.coaddimage.weight[i] += (
                            all_stamp["weight"] * all_stamp["border"]
                        )
            non_zero_weights = np.where(self.coaddimage.weight[i].array != 0)
            self.coaddimage.image[i].array[non_zero_weights] /= (
                self.coaddimage.weight[i].array[non_zero_weights]
            )
            if "noise" in list(all_stamp.keys()):
                self.coaddimage.noise[i].array[non_zero_weights] /= (
                    self.coaddimage.weight[i].array[non_zero_weights]
                )

    def _process_one_exp(self, exp):
        """Process one exposure

        Make the coadding step for one exposure (deconv, reconv, ...).
        NOTE: find a better way to do this.

        Args:
            exp (metacoadd): Exposure to coadd.
        Returns:
            (dict): Dict containing all the images link to an exposure in the
                coadd referential.
        """

        if not isinstance(exp, Exposure):
            raise TypeError("exp must be a metacoadd.Exposure.")

        stamp_dict = {}

        stamp_dict["image"] = exp.image_resamp
        # The noise image goes through the same process.
        if hasattr(exp, "noise"):
            stamp_dict["noise"] = exp.noise_resamp

        if hasattr(exp, "weight"):
            stamp_dict["weight"] = exp.weight_resamp

        if self._do_border:
            border_image = self._get_border(exp)
            border_stamp = galsim.Image(
                border_image,
                bounds=exp.image_resamp.bounds,
            )
        else:
            border_stamp = galsim.Image(
                bounds=exp.image_resamp.bounds,
            )
            border_stamp.fill(1)
        stamp_dict["border"] = border_stamp

        return stamp_dict

    def _get_border(self, exp):
        """Set border

        This method handle the CCD border to avoid issues in case the edge of
        a CCD falls in the coadd footprint.
        This step is necessarry due to the interpolation.

        Args:
            exp (metacoadd.Exposure): Exposure for which we create the border.

        Returns:
            galsim.Image: Galsim.Image representing te border.
        """

        full_bounds = exp._meta["image_bounds"]

        border_wcs = exp.wcs

        border_image = galsim.Image(
            bounds=full_bounds,
            wcs=border_wcs,
        )
        border_image.fill(0)

        border_bounds = galsim.BoundsI(
            xmin=full_bounds.xmin + self._border_size,
            xmax=full_bounds.xmax - self._border_size,
            ymin=full_bounds.ymin + self._border_size,
            ymax=full_bounds.ymax - self._border_size,
        )
        common_bound = border_image.bounds & border_bounds
        if common_bound.isDefined():
            border_image[border_bounds].fill(1)

        border_exp = Exposure(border_image, wcs=border_wcs)
        if self.coaddimage._relax_resize is None:
            resized_border = self.coaddimage._resize_exp(border_exp, 0)
        else:
            relax_resize = self.coaddimage._relax_resize
            resized_border = self.coaddimage._resize_exp(
                border_exp,
                relax_resize,
            )

        input_reproj = (
            np.array([resized_border.image.array]),
            resized_border.wcs.astropy,
        )
        # NOTE: need to change this part!
        # border_resamp, _ = self.coaddimage._do_resamp(
        #     input_reproj,
        #     "nearest",
        #     image_kinds=["border"],
        # )
        border_resamp, _ = self.coaddimage._do_resamp(
            input_reproj,
            "classic",
            image_kinds=["border"],
            resamp_algo="swarp",
        )

        return border_resamp[0]


class MetaCoadd(SimpleCoadd):
    def __init__(
        self,
        coaddimage,
        psfs,
        resamp_config,
        rng,
        coadd_method="weighted",
        do_border=True,
        step=0.01,
        types=["1m", "1p", "2m", "2p", "noshear"],
    ):
        super().__init__(coaddimage=coaddimage, coadd_method=coadd_method)

        self.psf_mb_explist = psfs
        self.resamp_config = resamp_config
        self.rng = rng
        self.mcal_config = {
            "step": step,
            "types": types,
            "psf": "fitgauss_UR",
            "use_noise_image": True,
        }

        # Set observations
        self.mbobs = exp2obs(
            self.coaddimage.mb_explist,
            self.psf_mb_explist,
            use_resamp=False,
        )

        self._do_border = do_border

    def go(
        self,
    ):
        """
        Run the coaddition process.
        """

        self.coaddimage.setup_coadd_metacal(self.mcal_config["types"])

        mcal_dict = self._run_metacal()

        self._mcal_coadd_image = {}
        self.mcal_mb_explist = {}
        self.mcal_mb_explist_psf = {}
        all_sep_cat = {}
        all_shape_cat = {}
        final_cat = {}
        for mcal_key in mcal_dict.keys():
            mb_obs = mcal_dict[mcal_key]

            mcal_coadd_image, mcal_mb_explist, mcal_mb_explist_psf = (
                self._get_resamp(mb_obs)
            )
            self._mcal_coadd_image[mcal_key] = mcal_coadd_image

            self.make_coadds(
                mcal_coadd_image.mb_explist, mcal_key, do_multiband=True
            )
            all_sep_cat[mcal_key] = self.get_cat(mcal_key, do_multiband=True)
            all_shape_cat[mcal_key] = self.get_shape_cat(
                mcal_mb_explist, mcal_mb_explist_psf, all_sep_cat[mcal_key]
            )
            final_cat[mcal_key] = self.build_output_cat(
                all_sep_cat[mcal_key], all_shape_cat[mcal_key]
            )
            self.mcal_mb_explist[mcal_key] = mcal_mb_explist
            self.mcal_mb_explist_psf[mcal_key] = mcal_mb_explist_psf

        self.all_sep_cat = all_sep_cat
        self.all_shape_cat = all_shape_cat
        return final_cat

    def _run_metacal(self):
        mcal_obs = get_all_metacal(
            self.mbobs,
            rng=self.rng,
            **self.mcal_config,
        )

        return mcal_obs

    def _get_resamp(self, mb_obs):
        mb_explist = MultiBandExpList()
        mb_explist_psf = MultiBandExpList()
        for i, obs_list in enumerate(mb_obs):
            exp_list = ExpList()
            exp_list_psf = ExpList()
            for j, obs in enumerate(obs_list):
                exp = Exposure(
                    image=obs.image,
                    weight=obs.weight,
                    noise=obs.noise,
                    wcs=self.coaddimage.mb_explist[i][j].wcs,
                )
                exp_list.append(exp)

                exp_psf = Exposure(
                    image=obs.psf.image,
                    weight=obs.psf.weight,
                    wcs=self.psf_mb_explist[i][j].wcs,
                )
                exp_list_psf.append(exp_psf)
            mb_explist.append(exp_list)
            mb_explist_psf.append(exp_list_psf)

        coadd_image = CoaddImage(
            copy.deepcopy(mb_explist),
            world_coadd_center=self.coaddimage.world_coadd_center,
            scale=self.coaddimage.coadd_pixel_scale,
            image_coadd_size=self.coaddimage.image_coadd_size,
        )

        coadd_image.get_all_resamp_images(**self.resamp_config)

        return coadd_image, mb_explist, mb_explist_psf

    def make_coadds(self, mb_explist, mcal_key, do_multiband=False):
        if do_multiband:
            if not self.coaddimage._mb_coadd_set:
                self.coaddimage.setup_mb_coadd_metacal(
                    self.mcal_config["types"]
                )
        for i, exp_list in enumerate(mb_explist):
            for exp in exp_list:
                if self._do_border:
                    border_image = galsim.Image(
                        self._get_border(exp),
                        bounds=exp.image_resamp.bounds,
                    )
                else:
                    border_image = galsim.Image(
                        bounds=exp.image_resamp.bounds,
                    )
                    border_image.fill(1)
                b = (
                    exp.image_resamp.bounds
                    & self.coaddimage.image[i][mcal_key].bounds
                )
                if b.isDefined():
                    if self._coadd_method == "weighted":
                        # NOTE: check for the presence of a 'weight' for the
                        # weighted average coadding
                        self.coaddimage.image[i][mcal_key][b] += (
                            exp.image_resamp * exp.weight_resamp * border_image
                        )
                        self.coaddimage.weight[i][mcal_key] += (
                            exp.weight_resamp * border_image
                        )
                        if do_multiband:
                            self.coaddimage.mb_image[mcal_key][b] += (
                                exp.image_resamp
                                * exp.weight_resamp
                                * border_image
                            )
                            self.coaddimage.mb_weight[mcal_key] += (
                                exp.weight_resamp * border_image
                            )
            non_zero_weights = np.where(
                self.coaddimage.weight[i][mcal_key].array != 0
            )
            self.coaddimage.image[i][mcal_key].array[non_zero_weights] /= (
                self.coaddimage.weight[i][mcal_key].array[non_zero_weights]
            )
        if do_multiband:
            non_zero_weights = np.where(
                self.coaddimage.mb_weight[mcal_key].array != 0
            )
            self.coaddimage.mb_image[mcal_key].array[non_zero_weights] /= (
                self.coaddimage.mb_weight[mcal_key].array[non_zero_weights]
            )

    def get_cat(self, mcal_key, do_multiband=False):
        if do_multiband:
            cat, seg_map = get_cat(
                self.coaddimage.mb_image[mcal_key].array,
                self.coaddimage.mb_weight[mcal_key].array,
                thresh=1.5,
                wcs=self.coaddimage.mb_image[mcal_key].wcs.astropy,
            )
        return cat

    def get_shape_cat(self, mcal_mb_explist, mcal_mb_explist_psf, sep_cat):
        cutout_size = 51
        fitter = ReGaussFitter(guess_fwhm=0.3)

        all_shape_cat = []
        for det_obj in sep_cat:
            mb_obs = ngmix.MultiBandObsList()
            for explist, explist_psf in zip(
                mcal_mb_explist, mcal_mb_explist_psf
            ):
                obs_list = ngmix.ObsList()
                for exp, exp_psf in zip(explist, explist_psf):
                    obj_world_pos = galsim.CelestialCoord(
                        ra=det_obj["ra"] * galsim.degrees,
                        dec=det_obj["dec"] * galsim.degrees,
                    )
                    img_pos = exp.wcs.toImage(obj_world_pos)
                    img, dx, dy = get_cutout(
                        exp.image.array, img_pos.x, img_pos.y, cutout_size
                    )
                    wgt, _, _ = get_cutout(
                        exp.weight.array, img_pos.x, img_pos.y, cutout_size
                    )
                    noise, _, _ = get_cutout(
                        exp.noise.array, img_pos.x, img_pos.y, cutout_size
                    )
                    jac = ngmix.Jacobian(
                        row=(cutout_size - 1) / 2.0 + dy,
                        col=(cutout_size - 1) / 2.0 + dx,
                        wcs=exp.wcs.local(world_pos=obj_world_pos),
                    )

                    psf_img = exp_psf.image.array
                    psf_wgt = exp_psf.weight.array
                    psf_jac = ngmix.Jacobian(
                        row=(psf_img.shape[0] - 1) / 2.0,
                        col=(psf_img.shape[1] - 1) / 2.0,
                        wcs=exp_psf.wcs.local(world_pos=obj_world_pos),
                    )
                    psf_obs = ngmix.Observation(
                        image=psf_img,
                        weight=psf_wgt,
                        jacobian=psf_jac,
                    )
                    obs = ngmix.Observation(
                        image=img,
                        weight=wgt,
                        jacobian=jac,
                        noise=noise,
                        psf=psf_obs,
                    )
                    obs_list.append(obs)
            mb_obs.append(obs_list)
            res = fitter.go(mb_obs)
            all_shape_cat.append(res)
        return all_shape_cat

    def build_output_cat(self, all_sep_cat, all_shape_cat):
        final_cat_dtype = DET_CAT_DTYPE + SHAPE_CAT_DTYPE
        final_cat = np.zeros(
            len(all_sep_cat),
            dtype=final_cat_dtype,
        )
        for i, sep_obj in enumerate(all_sep_cat):
            for key in sep_obj.dtype.names:
                final_cat[i][key] = sep_obj[key]
            for key in np.dtype(SHAPE_CAT_DTYPE).names:
                final_cat[i][key] = all_shape_cat[i][key.split("regauss_")[1]]
        return final_cat
