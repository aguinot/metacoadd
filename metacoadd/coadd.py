import numpy as np

COADD_TYPES = ["average", "weighted_average", "median"]


class Coadd:
    """Base class for coadding multi-band observations.

    We assume that the images are already aligned.
    Each images are also rescaled to share the same zeropoint.

    NOTE:
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
                obq_fscale = np.atleast_1d(fscale[i])
                if len(obq_fscale) != self._n_obs[i]:
                    passes = False
                    break
                self.fscale.append(obq_fscale)
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
                if len(zeropoints[i]) != self._n_obs[i]:
                    passes = False
                    break
                self.fscale.append(
                    [10 ** (0.4 * (zp - target_zp)) for zp in zeropoints[i]]
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

    def make(self):
        raise NotImplementedError(
            "This method should be implemented in subclasses."
        )


class CoaddAverage(Coadd):
    def make(self):
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
        image /= n_image
        noise /= n_image
        weight = n_image**2 / weight

        return image, noise, weight


class CoaddWeightedAverage(Coadd):
    def make(self):
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
    def make(self):
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
