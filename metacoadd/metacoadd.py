import esutil as eu
import galsim
import ngmix
import numpy as np
from metadetect import detect, procflags, shearpos
from metadetect.fitting import fit_mbobs_list_wavg
from metadetect.mfrac import measure_mfrac

from metacoadd.exposure import CoaddImage, Exposure
from metacoadd.utils import exp2obs

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
            ],  # noqa
            [
                0.021388,
                0.092163,
                0.221178,
                0.296069,
                0.221178,
                0.092163,
                0.021388,
            ],  # noqa
            [
                0.051328,
                0.221178,
                0.530797,
                0.710525,
                0.530797,
                0.221178,
                0.051328,
            ],  # noqa
            [
                0.068707,
                0.296069,
                0.710525,
                0.951108,
                0.710525,
                0.296069,
                0.068707,
            ],  # noqa
            [
                0.051328,
                0.221178,
                0.530797,
                0.710525,
                0.530797,
                0.221178,
                0.051328,
            ],  # noqa
            [
                0.021388,
                0.092163,
                0.221178,
                0.296069,
                0.221178,
                0.092163,
                0.021388,
            ],  # noqa
            [
                0.004963,
                0.021388,
                0.051328,
                0.068707,
                0.051328,
                0.021388,
                0.004963,
            ],  # noqa
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
            self._border_size = 20

    def go(self):
        """
        Run the coaddition process.
        """

        if len(self.coaddimage.explist) == 0:
            raise ValueError("No exposure find to make the coadd.")
        if not self.coaddimage.explist[0]._resamp:
            # raise ValueError('Exposure must be resampled first.')
            self.coaddimage.get_all_resamp_images()

        self.coaddimage.setup_coadd()
        stamps = []
        for exp in self.coaddimage.explist:
            all_stamp = self._process_one_exp(exp)

            # Check bounds, it should always pass. Just for safety.
            # We check only 'image' because it we always be there and the
            # property are shared with the other kind.
            b = all_stamp["image"].bounds & self.coaddimage.image.bounds
            if b.isDefined():
                if self._coadd_method == "weighted":
                    # NOTE: check for the presence of a 'weight' for the
                    # weighted average coadding
                    self.coaddimage.image[b] += (
                        all_stamp["image"]
                        * all_stamp["weight"]
                        * all_stamp["border"]
                    )
                    if "noise" in list(all_stamp.keys()):
                        self.coaddimage.noise[b] += (
                            all_stamp["noise"]
                            * all_stamp["weight"]
                            * all_stamp["border"]
                        )
                    self.coaddimage.weight += (
                        all_stamp["weight"] * all_stamp["border"]
                    )
            stamps.append(all_stamp)
        self.stamps = stamps
        non_zero_weights = np.where(self.coaddimage.weight.array != 0)
        self.coaddimage.image.array[
            non_zero_weights
        ] /= self.coaddimage.weight.array[non_zero_weights]
        if "noise" in list(all_stamp.keys()):
            self.coaddimage.noise.array[
                non_zero_weights
            ] /= self.coaddimage.weight.array[non_zero_weights]

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
            border_image = self._set_border(exp)
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

    def _set_border(self, exp):
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

        border_wcs = exp.wcs_bundle.galsim

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
            resized_border.image.array,
            resized_border.wcs_bundle.astropy,
        )
        border_resamp, _ = self.coaddimage._do_resamp(
            input_reproj,
            "nearest",
        )

        return border_resamp


class MetaCoadd(SimpleCoadd):
    def __init__(
        self,
        coaddimage,
        psfs,
        coadd_method="weighted",
        step=0.01,
        types=["1m", "1p", "2m", "2p", "noshear"],
    ):
        super().__init__(coaddimage, coadd_method)

        self.psf_coaddimage = psfs

        self.step = step
        self.types = types

        self._do_border = True

    def go(
        self,
    ):
        """
        Run the coaddition process.
        """

        if len(self.coaddimage.explist) == 0:
            raise ValueError("No exposure find to make the coadd.")
        # if not self.coaddimage.explist[0]._interp:
        #     raise ValueError('Exposure must be interpolated first.')

        self.coaddimage.setup_coadd_metacal(self.types)
        self.psf_coaddimage.setup_coadd_metacal(self.types)
        stamps = []
        for n, exp in enumerate(self.coaddimage.explist):
            # print(n)
            # all_stamp = self._process_one_exp(exp, self._inv_psflist[n])
            all_stamp = self._process_one_exp(
                exp, self.psf_coaddimage.explist[n]
            )

            # Check bounds, it should always pass. Just for safety.
            # We check only 'image' because it we always be there and the
            # property are shared with the other kind.
            for type in self.types:
                b = (
                    all_stamp["image"][type].bounds
                    & self.coaddimage.image[type].bounds
                )
                if b.isDefined():
                    if self._coadd_method == "weighted":
                        # NOTE: check for the presence of a 'weight' for the
                        # weighted average coadding
                        self.coaddimage.image[type][b] += (
                            all_stamp["image"][type]
                            * all_stamp["weight"][type]
                            * all_stamp["border"]
                        )
                        if "noise" in list(all_stamp.keys()):
                            self.coaddimage.noise[type][b] += (
                                all_stamp["noise"][type]
                                * all_stamp["weight"][type]
                                * all_stamp["border"]
                            )
                        self.coaddimage.weight[type] += (
                            all_stamp["weight"][type] * all_stamp["border"]
                        )

            # Now we coadd the PSF
            for type in self.types:
                b = (
                    all_stamp["psf"][type].bounds
                    & self.psf_coaddimage.image[type].bounds
                )
                if b.isDefined():
                    self.psf_coaddimage.image[type][b] += all_stamp["psf"][type]

            stamps.append(all_stamp)
        self.stamps = stamps

        # Finish the stacking
        for type in self.types:
            non_zero_weights = np.where(self.coaddimage.weight[type].array != 0)
            self.coaddimage.image[type].array[
                non_zero_weights
            ] /= self.coaddimage.weight[type].array[non_zero_weights]
            if "noise" in list(all_stamp.keys()):
                self.coaddimage.noise[type].array[
                    non_zero_weights
                ] /= self.coaddimage.weight[type].array[non_zero_weights]
            self.psf_coaddimage.image[type].array[:, :] /= len(
                self.psf_coaddimage.explist
            )

        # Run detection + shape measurement
        self.results = {}
        self.mcal_dict = {}
        for type in self.types:
            img_jac = ngmix.Jacobian(
                row=self.coaddimage.image_coadd_center.x,
                col=self.coaddimage.image_coadd_center.y,
                wcs=self.coaddimage.coadd_wcs_bundle.galsim.jacobian(
                    world_pos=self.coaddimage.world_coadd_center
                ),
            )
            psf_jac = ngmix.Jacobian(
                row=self.psf_coaddimage.image_coadd_center.x,
                col=self.psf_coaddimage.image_coadd_center.y,
                wcs=self.psf_coaddimage.coadd_wcs_bundle.galsim.jacobian(
                    world_pos=self.psf_coaddimage.world_coadd_center
                ),
            )
            psf_obs = ngmix.Observation(
                image=self.psf_coaddimage.image[type].array,
                jacobian=psf_jac,
            )
            obs = ngmix.Observation(
                image=self.coaddimage.image[type].array,
                weight=self.coaddimage.weight[type].array,
                bmask=np.zeros_like(
                    self.coaddimage.image[type].array,
                    dtype=np.int32,
                ),
                ormask=np.zeros_like(
                    self.coaddimage.image[type].array,
                    dtype=np.int32,
                ),
                noise=self.coaddimage.noise[type].array,
                jacobian=img_jac,
                psf=psf_obs,
            )
            mbobs = ngmix.MultiBandObsList()
            obslist = ngmix.ObsList()
            obslist.append(obs)
            mbobs.append(obslist)
            self.mcal_dict[type] = mbobs
            self.results[type] = self._measure(
                mbobs,
                type,
                nonshear_mbobs=None,
            )
            self.obs = obs

    def _run_metacal(self, exp, psf_exp, use_resamp=True):
        obs = exp2obs(exp, psf_exp, use_resamp=use_resamp)

        rng = np.random.RandomState(1234)

        mcal_obs = ngmix.metacal.get_all_metacal(
            obs,
            step=self.step,
            types=self.types,
            use_noise_image=True,
            # psf="gauss",
            psf=galsim.Gaussian(sigma=0.6, flux=1.0),
            rng=rng,
        )

        return mcal_obs

    def _process_one_exp(self, exp, psf_exp):
        if self._do_border:
            border_image = self._set_border(exp)
            border_stamp = galsim.Image(
                border_image,
                bounds=exp.image_resamp.bounds,
            )
        else:
            border_stamp = galsim.Image(
                bounds=exp.image_resamp.bounds,
            )
            border_stamp.fill(1)

        mcal_obs = self._run_metacal(exp, psf_exp, use_resamp=True)

        stamp_dict = {"image": {}, "psf": {}, "border": {}}
        if hasattr(exp, "noise"):
            stamp_dict["noise"] = {}
        if hasattr(exp, "weight"):
            stamp_dict["weight"] = {}

        for key in stamp_dict.keys():
            if key == "border":
                stamp_dict[key] = border_stamp
                continue
            for type in self.types:
                if key == "psf":
                    img = getattr(mcal_obs[type], key).image
                    # print("after")
                    # print(galsim.hsm.FindAdaptiveMom(galsim.Image(img)))
                    wcs = getattr(mcal_obs[type], key).jacobian.get_galsim_wcs()
                else:
                    img = getattr(mcal_obs[type], key)
                    wcs = mcal_obs[type].jacobian.get_galsim_wcs()
                gs_img = galsim.Image(
                    img,
                    wcs=wcs,
                )
                stamp_dict[key][type] = gs_img

        return stamp_dict

    def _render_coord(self, coord):
        return [coord.ra.deg, coord.dec.deg]

    def _get_shear(self, type):
        if type == "1m":
            return -self.step, 0.0
        elif type == "1p":
            return self.step, 0.0
        elif type == "2m":
            return 0.0, -self.step
        elif type == "2p":
            return 0.0, self.step
        elif type == "noshear":
            return 0.0, 0.0
        else:
            raise ValueError(
                'type must be in ["1m", "1p", "2m", "2p", "noshear"].'
            )

    def _process_psf(self, psfs):
        if isinstance(psfs, list):
            if all(isinstance(n, np.ndarray) for n in psfs):
                (
                    self._inv_psflist,
                    self._sigma_psflist,
                    self._flux_psflist,
                ) = self._init_invpsf_nparray(psfs)
            elif all(isinstance(n, galsim.Image) for n in psfs):
                (
                    self._inv_psflist,
                    self._sigma_psflist,
                    self._flux_psflist,
                ) = self._init_invpsf_galimage(psfs)
            else:
                raise TypeError(
                    "psf should be a list of numpy.ndarray or galsim.Image."
                )
        else:
            raise TypeError(
                "psf should be a list of numpy.ndarray or galsim.Image."
            )

    def _init_invpsf_nparray(self, psfs):
        inv_psflist = []
        sigma_psflist = []
        flux_psflist = []
        for n, psf in enumerate(psfs):
            local_wcs = self._get_exposures_local_wcs(
                self.coaddimage.explist[n]
            )
            psf_img = galsim.Image(
                psf,
                wcs=local_wcs,
            )
            psf_interp = galsim.InterpolatedImage(
                psf_img,
                x_interpolant="lanczos15",
            )
            exp = self.coaddimage.explist[0]
            sigma_psflist.append(
                self._get_exp_reconv_psf_size(
                    self.deconvolve(psf_interp, self._get_exposures_pixel(exp))
                )
            )
            flux_psflist.append(psf_interp.flux)
            inv_psf_interp = galsim.Deconvolve(psf_interp)

            inv_psflist.append(inv_psf_interp)

        return inv_psflist, sigma_psflist, flux_psflist

    def _init_invpsf_galimage(self, psfs):
        inv_psflist = []
        sigma_psflist = []
        flux_psflist = []
        for n, psf in enumerate(psfs):
            local_wcs = self._get_exposures_local_wcs(
                self.coaddimage.explist[n]
            )
            if hasattr(psf, "wcs"):
                if psf.wcs != local_wcs:
                    raise ValueError(
                        "The provided psf WCS are different from the derived "
                        "local WCS on the corresponding single exposure."
                    )
            else:
                psf.wcs = local_wcs
            psf_interp = galsim.InterpolatedImage(
                psf,
                x_interpolant="lanczos15",
            )
            sigma_psflist.append(
                self._get_exp_reconv_psf_size(
                    psf_interp,
                )
            )
            flux_psflist.append(psf_interp.flux)
            inv_psf_interp = galsim.Deconvolve(psf_interp)

            inv_psflist.append(inv_psf_interp)

        return inv_psflist, sigma_psflist, flux_psflist

    def _get_exposures_local_wcs(self, exp):
        """
        Get the local WCS for the single exposure images.

        Args:
            exp (metacoadd.Exposure): Exposure from which we want the pixel
                information.
        Returns
            galsim.BaseWCS: Local WCS.
        """

        wcs = exp.image.wcs
        if not wcs.isLocal():
            wcs = wcs.local(
                world_pos=self.coaddimage.world_coadd_center,
            )

        return wcs

    def _get_exp_reconv_psf_size(self, psf):
        """
        taken from galsim/tests/test_metacal.py
        assumes the psf is centered
        """

        dk = psf.stepk / 4.0

        small_kval = 1.0e-2  # Find the k where the given psf hits this kvalue
        smaller_kval = 3.0e-3  # Target PSF will have this kvalue at the same k

        kim = psf.drawKImage(scale=dk)
        karr_r = kim.real.array
        # Find the smallest r where the kval < small_kval
        nk = karr_r.shape[0]
        kx, ky = np.meshgrid(
            np.arange(-nk / 2, nk / 2), np.arange(-nk / 2, nk / 2)
        )
        ksq = (kx**2 + ky**2) * dk**2
        ksq_max = np.min(ksq[karr_r < small_kval * psf.flux])

        # We take our target PSF to be the (round) Gaussian that is even
        # smaller at this ksq
        # exp(-0.5 * ksq_max * sigma_sq) = smaller_kval
        sigma_sq = -2.0 * np.log(smaller_kval) / ksq_max

        return np.sqrt(sigma_sq)

    def _get_reconv_psf(self, g1, g2, method="max"):
        g = np.sqrt(g1**2.0 + g2**2.0)
        dilation = 1.0 + 2.0 * g

        if method == "max":
            sigma = np.max(self._sigma_psflist)
            flux = np.mean(self._flux_psflist)

            psf = galsim.Gaussian(sigma=sigma, flux=flux)
            psf = psf.dilate(dilation)

        return psf

    def _measure(self, mbobs, shear_str, nonshear_mbobs=None):
        """
        perform measurements on the input mbobs. This involves running
        detection as well as measurements.
        we only detect on the shear bands in mbobs.
        we then do flux measurements on the nonshear_mbobs as well if it is
        given.
        """

        medsifier = self._do_detect(mbobs)
        mbm = medsifier.get_multiband_meds()
        mbobs_list = mbm.get_mbobs_list()

        if nonshear_mbobs is not None:
            nonshear_medsifier = detect.CatalogMEDSifier(
                nonshear_mbobs,
                medsifier.cat["x"],
                medsifier.cat["y"],
                medsifier.cat["box_size"],
            )
            nonshear_mbm = nonshear_medsifier.get_multiband_meds()
            nonshear_mbm.get_mbobs_list()
        else:
            pass

        self._fitter = ngmix.gaussmom.GaussMom(fwhm=1.2)

        res = fit_mbobs_list_wavg(
            mbobs_list=mbobs_list,
            fitter=self._fitter,
            bmask_flags=TEST_METADETECT_CONFIG.get("bmask_flags", 0),
        )

        if res is not None:
            res = self._add_positions_and_psf(
                mbobs, medsifier.cat, res, shear_str
            )

        return res

    def _do_detect(self, mbobs):
        """
        use a MEDSifier to run detection
        """
        return detect.MEDSifier(
            mbobs=mbobs,
            sx_config=TEST_METADETECT_CONFIG["sx"],
            meds_config=TEST_METADETECT_CONFIG["meds"],
            # nodet_flags=TEST_METADETECT_CONFIG['nodet_flags'],
        )

    def _add_positions_and_psf(self, mbobs, cat, res, shear_str):
        """
        add catalog positions to the result
        """

        new_dt = [
            ("sx_row", "f4"),
            ("sx_col", "f4"),
            ("sx_row_noshear", "f4"),
            ("sx_col_noshear", "f4"),
            ("ormask", "i4"),
            ("mfrac", "f4"),
            ("bmask", "i4"),
            ("ormask_det", "i4"),
            ("mfrac_det", "f4"),
            ("bmask_det", "i4"),
        ]
        if "psfrec_flags" not in res.dtype.names:
            new_dt += [
                ("psfrec_flags", "i4"),  # psfrec is the original psf
                ("psfrec_g", "f8", 2),
                ("psfrec_T", "f8"),
            ]
        newres = eu.numpy_util.add_fields(
            res,
            new_dt,
        )
        if "psfrec_flags" not in res.dtype.names:
            newres["psfrec_flags"] = procflags.NO_ATTEMPT

        if hasattr(self, "psf_stats"):
            newres["psfrec_flags"][:] = self.psf_stats["flags"]
            newres["psfrec_g"][:, 0] = self.psf_stats["g1"]
            newres["psfrec_g"][:, 1] = self.psf_stats["g2"]
            newres["psfrec_T"][:] = self.psf_stats["T"]

        if cat.size > 0:
            obs = mbobs[0][0]

            newres["sx_col"] = cat["x"]
            newres["sx_row"] = cat["y"]

            rows_noshear, cols_noshear = shearpos.unshear_positions_obs(
                newres["sx_row"],
                newres["sx_col"],
                shear_str,
                obs,
                # an example for jacobian and image shape
                # default is 0.01 but make sure to use the passed in default
                # if needed
                step=TEST_METADETECT_CONFIG["metacal"].get(
                    "step", shearpos.DEFAULT_STEP
                ),
            )

            newres["sx_row_noshear"] = rows_noshear
            newres["sx_col_noshear"] = cols_noshear

            self._set_ormask_and_bmask(mbobs)
            if (
                "ormask_region" in TEST_METADETECT_CONFIG
                and TEST_METADETECT_CONFIG["ormask_region"] > 1
            ):
                ormask_region = TEST_METADETECT_CONFIG["ormask_region"]
            elif (
                "mask_region" in TEST_METADETECT_CONFIG
                and TEST_METADETECT_CONFIG["mask_region"] > 1
            ):
                ormask_region = TEST_METADETECT_CONFIG["mask_region"]
            else:
                ormask_region = 1

            if (
                "mask_region" in TEST_METADETECT_CONFIG
                and TEST_METADETECT_CONFIG["mask_region"] > 1
            ):
                bmask_region = TEST_METADETECT_CONFIG["mask_region"]
            else:
                bmask_region = 1

            newres["ormask"] = _fill_in_mask_col(
                mask_region=ormask_region,
                rows=newres["sx_row_noshear"],
                cols=newres["sx_col_noshear"],
                mask=self.ormask,
            )
            newres["ormask_det"] = _fill_in_mask_col(
                mask_region=ormask_region,
                rows=newres["sx_row"],
                cols=newres["sx_col"],
                mask=self.ormask,
            )

            newres["bmask"] = _fill_in_mask_col(
                mask_region=bmask_region,
                rows=newres["sx_row_noshear"],
                cols=newres["sx_col_noshear"],
                mask=self.bmask,
            )
            newres["bmask_det"] = _fill_in_mask_col(
                mask_region=bmask_region,
                rows=newres["sx_row"],
                cols=newres["sx_col"],
                mask=self.bmask,
            )

            self._set_mfrac(mbobs)
            if np.any(self.mfrac > 0):
                newres["mfrac"] = measure_mfrac(
                    mfrac=self.mfrac,
                    x=newres["sx_col_noshear"],
                    y=newres["sx_row_noshear"],
                    box_sizes=cat["box_size"],
                    obs=obs,
                    fwhm=self.get("mfrac_fwhm", None),
                )

                newres["mfrac_det"] = measure_mfrac(
                    mfrac=self.mfrac,
                    x=newres["sx_col"],
                    y=newres["sx_row"],
                    box_sizes=cat["box_size"],
                    obs=obs,
                    fwhm=self.get("mfrac_fwhm", None),
                )
            else:
                newres["mfrac"] = 0

        return newres

    def _set_ormask_and_bmask(self, mbobs):
        """
        set the ormask and bmask, ored from all epochs
        """

        for band, obslist in enumerate(mbobs):
            nepoch = len(obslist)
            assert nepoch == 1, "expected 1 epoch, got %d" % nepoch

            obs = obslist[0]

            if band == 0:
                ormask = obs.ormask.copy()
                bmask = obs.bmask.copy()
            else:
                ormask |= obs.ormask
                bmask |= obs.bmask

        self.ormask = ormask
        self.bmask = bmask

    def _set_mfrac(self, mbobs):
        """
        set the masked fraction image, averaged over all bands
        """
        wgts = []
        mfrac = np.zeros_like(mbobs[0][0].image)
        for band, obslist in enumerate(mbobs):
            nepoch = len(obslist)
            assert nepoch == 1, "expected 1 epoch, got %d" % nepoch

            obs = obslist[0]
            msk = obs.weight > 0
            if not np.any(msk):
                wgt = 0
            else:
                wgt = np.median(obs.weight[msk])
            if hasattr(obs, "mfrac"):
                mfrac += obs.mfrac * wgt
            wgts.append(wgt)

        if np.sum(wgts) > 0:
            mfrac = mfrac / np.sum(wgts)
        else:
            mfrac[:, :] = 1.0

        self.mfrac = mfrac


def _clip_and_round(vals_in, dim):
    """
    clip values and round to nearest integer
    """

    vals = vals_in.copy()

    np.rint(vals, out=vals)
    vals.clip(min=0, max=dim - 1, out=vals)

    return vals.astype("i4")


def _fill_in_mask_col(*, mask_region, rows, cols, mask):
    dims = mask.shape
    rclip = _clip_and_round(rows, dims[0])
    cclip = _clip_and_round(cols, dims[1])

    if mask_region > 1:
        res = np.zeros_like(rows, dtype=np.int32)
        for ind in range(rows.size):
            lr = int(min(dims[0] - 1, max(0, rclip[ind] - mask_region)))
            ur = int(min(dims[0] - 1, max(0, rclip[ind] + mask_region)))

            lc = int(min(dims[1] - 1, max(0, cclip[ind] - mask_region)))
            uc = int(min(dims[1] - 1, max(0, cclip[ind] + mask_region)))

            res[ind] = np.bitwise_or.reduce(
                mask[lr : ur + 1, lc : uc + 1],
                axis=None,
            )
    else:
        res = mask[rclip, cclip].astype(np.int32)

    return res
