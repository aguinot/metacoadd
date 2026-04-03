from metadetect.fitting import fit_mbobs_wavg

from ..utils import atleast_mbobs


class MBMomRunner:
    """
    This is essentially a wrapper around the `fit_mbobs_wavg` from metadetect.
    """

    def __init__(
        self,
        fitter,
        fitter_name,
        bmask_flag=None,
        fwhm_reg=0.0,
        symmetrize=True,
    ):
        self.fitter = fitter
        self.fitter_name = fitter_name
        if bmask_flag is None:
            self.bmask_flag = 2**30
        else:
            self.bmask_flag = bmask_flag
        self.fwhm_reg = fwhm_reg
        self.symmetrize = symmetrize

    def go(self, obs):
        mbobs = atleast_mbobs(obs)
        res_ = fit_mbobs_wavg(
            mbobs=mbobs,
            fitter=self.fitter,
            bmask_flags=self.bmask_flag,
            fwhm_reg=self.fwhm_reg,
            symmetrize=self.symmetrize,
        )
        res = {}
        for k, v in zip(res_[0].dtype.names, res_[0]):
            if self.fitter_name in k:
                new_k = k.split(f"{self.fitter_name}_")[1]
                if "band" in new_k:
                    new_k = new_k.split("band_")[1]
                res[new_k] = v
        self._res_ = res_
        self._res = res

        return res
