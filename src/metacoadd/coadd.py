import numpy as np

COADD_TYPES = ["average", "weighted_average", "median"]


class Coadd:
    """Base class for coadding multi-band observations.

    We assume that the images are already aligned.
    Each images are also rescaled to share the same zeropoint.

    Note:
    At the moment, all images multi-band and multi-epoch are combined into a
    single image.

    """

    def __init__(self, mb_obs, fscale=None, zeropoints=None, target_zp=30.0):
        self._set_data(
            mb_obs, fscale=fscale, zeropoints=zeropoints, target_zp=target_zp
        )

    def _set_data(self, mb_obs, fscale=None, zeropoints=None, target_zp=30.0):
        self.mb_obs = mb_obs
        self._n_band = len(mb_obs)
        self._n_obs = [len(obs) for obs in mb_obs]

        if zeropoints is None and fscale is not None:
            if len(fscale) != self._n_band:
                raise ValueError(
                    "fscale must have the same length as the number of bands."
                )
            passes = True
            self.fscale = []
            for i in range(self._n_band):
                obs_fscale = np.atleast_1d(fscale[i])
                if len(obs_fscale) != self._n_obs[i]:
                    passes = False
                    break
                self.fscale.append(obs_fscale)
            if not passes:
                raise ValueError(
                    "fscale must have the same length as the number of "
                    "observations in each band."
                )
        elif zeropoints is not None and fscale is None:
            if len(zeropoints) != self._n_band:
                raise ValueError(
                    "zeropoints must have the same length as the number of "
                    "bands."
                )
            passes = True
            self.fscale = []
            for i in range(self._n_band):
                obs_zeropoints = np.atleast_1d(zeropoints[i])
                if len(obs_zeropoints) != self._n_obs[i]:
                    passes = False
                    break
                self.fscale.append(
                    [10 ** (-0.4 * (zp - target_zp)) for zp in obs_zeropoints]
                )
            if not passes:
                raise ValueError(
                    "zeropoints must have the same length as the number of "
                    "observations in each band."
                )
        elif zeropoints is None and fscale is None:
            self.fscale = [[1.0] * self._n_obs[i] for i in range(self._n_band)]
        else:
            raise ValueError(
                "Either fscale or zeropoints must be provided, but not both."
            )
        self.fscale = np.asarray(self.fscale, dtype=np.float64)

    def make_masks(self):
        """Combine bmask and ormask across all observations.

        Each mask type is treated independently. If a mask is present on every
        observation, its values are combined pixel-by-pixel using bitwise OR.
        If it is absent from every observation, None is returned for that mask
        type.

        A mask present on only a subset of observations is considered
        inconsistent and raises an error.

        Returns
        -------
        bmask : np.ndarray or None
            Combined bmask, or None if no observations contain
            a bmask.
        ormask : np.ndarray or None
            Combined ormask, or None if no observations contain
            an ormask.
        """

        n_image = 0
        n_bmask = 0
        n_ormask = 0
        for i in range(self._n_band):
            for j in range(self._n_obs[i]):
                obs = self.mb_obs[i][j]
                if obs.has_bmask():
                    if n_bmask == 0:
                        bmask = np.zeros_like(obs.bmask)
                    bmask |= obs.bmask
                    n_bmask += 1
                if obs.has_ormask():
                    if n_ormask == 0:
                        ormask = np.zeros_like(obs.ormask)
                    ormask |= obs.ormask
                    n_ormask += 1
                n_image += 1
        if n_bmask == 0:
            bmask = None
        else:
            if n_bmask != n_image:
                raise ValueError(
                    "bmask must be set on all observations or none"
                )
        if n_ormask == 0:
            ormask = None
        else:
            if n_ormask != n_image:
                raise ValueError(
                    "ormask must be set on all observations or none"
                )

        return bmask, ormask

    def make(self):
        """Make the coadd image, noise and weight.
        Implemented in subclasses.
        """
        raise NotImplementedError(
            "This method should be implemented in subclasses."
        )


class CoaddAverage(Coadd):
    """Coadd multi-band observations using a simple average."""

    def make(self):
        """Make the coadd image, noise and weight using a simple average.

        Returns
        -------
        image : np.ndarray
            The coadded image.
        noise : np.ndarray
            The coadded noise.
        weight : np.ndarray
            The coadded weight.

        """
        image = np.zeros_like(self.mb_obs[0][0].image)
        noise = np.zeros_like(image)
        weight = np.zeros_like(image)
        n_image = np.zeros_like(image)
        for i in range(self._n_band):
            for j in range(self._n_obs[i]):
                msk = self.mb_obs[i][j].weight != 0
                image[msk] += self.mb_obs[i][j].image[msk] * self.fscale[i][j]
                noise[msk] += self.mb_obs[i][j].noise[msk] * self.fscale[i][j]
                weight[msk] += (
                    1 / self.mb_obs[i][j].weight[msk] * self.fscale[i][j] ** 2
                )
                n_image[msk] += 1
        image[n_image != 0] /= n_image[n_image != 0]
        noise[n_image != 0] /= n_image[n_image != 0]
        weight[n_image != 0] = n_image[n_image != 0] ** 2 / weight[n_image != 0]

        return image, noise, weight


class CoaddWeightedAverage(Coadd):
    """Coadd multi-band observations using a weighted average."""

    def make(self):
        """Make the coadd image, noise and weight using a weighted average.
        It follows the method from SWARP.

        Returns
        -------
        image : np.ndarray
            The coadded image.
        noise : np.ndarray
            The coadded noise.
        weight : np.ndarray
            The coadded weight.

        """
        image = np.zeros_like(self.mb_obs[0][0].image)
        noise = np.zeros_like(image)
        weight = np.zeros_like(image)
        n_image = np.zeros_like(image)
        for i in range(self._n_band):
            for j in range(self._n_obs[i]):
                msk = self.mb_obs[i][j].weight != 0
                image[msk] += (
                    self.mb_obs[i][j].image[msk]
                    * self.mb_obs[i][j].weight[msk]
                    * self.fscale[i][j]
                )
                noise[msk] += (
                    self.mb_obs[i][j].noise[msk]
                    * self.mb_obs[i][j].weight[msk]
                    * self.fscale[i][j]
                )
                weight[msk] += self.mb_obs[i][j].weight[msk]
                n_image[msk] += 1

        image[n_image != 0] /= weight[n_image != 0]
        noise[n_image != 0] /= weight[n_image != 0]

        return image, noise, weight


class CoaddMedian(Coadd):
    """Coadd multi-band observations using a median."""

    def make(self):
        """Make the coadd image, noise and weight using a median.
        It follows the method from SWARP.

        Returns
        -------
        image : np.ndarray
            The coadded image.
        noise : np.ndarray
            The coadded noise.
        weight : np.ndarray
            The coadded weight.

        """
        image = []
        noise = []
        weight = np.zeros_like(self.mb_obs[0][0].image)
        n_image = np.zeros_like(self.mb_obs[0][0].image)
        for i in range(self._n_band):
            for j in range(self._n_obs[i]):
                msk = self.mb_obs[i][j].weight != 0
                image.append(self.mb_obs[i][j].image * self.fscale[i][j])
                noise.append(self.mb_obs[i][j].noise * self.fscale[i][j])
                weight[msk] += np.sqrt(
                    self.mb_obs[i][j].weight[msk] / self.fscale[i][j] ** 2
                )
                n_image[msk] += 1

        image = np.median(image, axis=0)
        noise = np.median(noise, axis=0)
        weight[n_image != 0] = (
            2.0 / np.pi * (weight[n_image != 0] / n_image[n_image != 0]) ** 2
        )
        msk_even = n_image % 2 == 0
        weight[msk_even] *= n_image[msk_even] + np.pi / 2 - 1
        msk_odd = n_image % 2 != 0
        weight[msk_odd] *= n_image[msk_odd] + np.pi - 2

        return image, noise, weight


def get_coadd_class(coadd_type):
    """Get the coadd class based on the coadd type."""
    if coadd_type == "average":
        return CoaddAverage
    elif coadd_type == "weighted_average" or coadd_type == "weighted":
        return CoaddWeightedAverage
    elif coadd_type == "median":
        return CoaddMedian
    else:
        raise ValueError(
            f"Unknown coadd type: {coadd_type}. Must be one of {COADD_TYPES}."
        )
