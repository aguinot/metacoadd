import ngmix
from ngmix.gmix.gmix import _gmix_model_dict

from .moments.wmom_runner import MBMomRunner
from .fitters.fourier_fitting import FourierFitter


def get_fitters(models, fwhm=None, rng=None, nband=None, scale=None):

    if isinstance(models, str):
        models = [models]
    elif not isinstance(models, list):
        raise ValueError("models must be a string or a list of strings")

    fitters = {}
    for model in models:
        fitters[model] = get_runner(
            model, fwhm=fwhm, rng=rng, nband=nband, scale=scale
        )
    return fitters


def parse_model(model):

    if model in ["wmom", "pgauss", "am"]:
        pass
    elif model in _gmix_model_dict:
        pass
    elif "fourier" in model:
        base_model = model.split("fourier_")[-1]
        if base_model not in _gmix_model_dict:
            raise ValueError(
                f"Model {model} not recognized. Must be one of {list(_gmix_model_dict.keys())} or 'wmom'"
            )
    else:
        raise ValueError(
            f"Model {model} not recognized. Must be one of {list(_gmix_model_dict.keys())} or 'wmom'"
        )
    return model


def get_runner(model, fwhm=None, rng=None, nband=None, scale=None):

    model = parse_model(model)

    if model in ["wmom", "pgauss", "am"]:
        runner = build_mb_wmom_runner(model, fwhm=fwhm, rng=rng)
    elif "fourier" in model:
        runner = build_model_fitting_fourier_runner(
            model, rng=rng, nband=nband, scale=scale
        )
    else:
        runner = build_model_fitting_runner(
            model, rng=rng, nband=nband, scale=scale
        )
    return runner


def build_mb_wmom_runner(model, fwhm=None, rng=None):
    if model == "wmom":
        runner = ngmix.gaussmom.GaussMom(fwhm=fwhm)
    elif model == "pgauss":
        runner = ngmix.prepsfmom.PGaussMom(fwhm=fwhm)
    elif model == "am":
        fitter = ngmix.admom.AdmomFitter()
        guesser = ngmix.guessers.GMixPSFGuesser(
            rng=rng,
            ngauss=1,
            guess_from_moms=True,
        )
        runner = ngmix.runners.Runner(
            fitter=fitter,
            guesser=guesser,
            ntry=2,
        )
    return MBMomRunner(
        fitter=runner,
        fitter_name=model,
    )


def build_model_fitting_runner(model, rng, nband, scale):
    if model in ["gauss", "exp", "dev"]:
        gal_runner = get_single_model_runner(model, rng, nband, scale)
    else:
        raise NotImplementedError(f"Model {model} not implemented yet")
    psf_runner = get_gauss_psf_runner(rng)
    boot = ngmix.bootstrap.Bootstrapper(
        runner=gal_runner,
        psf_runner=psf_runner,
    )
    return boot


def build_model_fitting_fourier_runner(fourier_model, rng, nband, scale):
    model = fourier_model.split("fourier_")[-1]
    if model in ["gauss", "exp", "dev"]:
        gal_runner = get_single_fourier_model_runner(model, rng, nband, scale)
    else:
        raise NotImplementedError(f"Model {model} not implemented yet")
    psf_runner = get_gauss_psf_runner(rng)
    boot = ngmix.bootstrap.Bootstrapper(
        runner=gal_runner,
        psf_runner=psf_runner,
    )
    return boot


def get_gauss_psf_runner(rng):
    psf_guesser = ngmix.guessers.SimplePSFGuesser(
        rng=rng,
        guess_from_moms=True,
    )
    psf_fitter = ngmix.fitting.Fitter(
        model="gauss",
        fit_pars={
            "maxfev": 2000,
            "xtol": 1.0e-5,
            "ftol": 1.0e-5,
        },
    )
    psf_runner = ngmix.runners.PSFRunner(
        fitter=psf_fitter,
        guesser=psf_guesser,
        ntry=2,
    )
    return psf_runner


def get_single_model_runner(model, rng, nband, scale):
    prior = _make_prior(rng, scale, nband)

    fitter = ngmix.fitting.Fitter(
        model=model,
        prior=prior,
        fit_pars={
            "maxfev": 2000,
            "xtol": 5.0e-5,
            "ftol": 5.0e-5,
        },
    )
    guesser = ngmix.guessers.TPSFFluxGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
    )
    runner = ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )

    return runner


def get_single_fourier_model_runner(model, rng, nband, scale):
    prior = _make_prior(rng, scale, nband)

    fitter = FourierFitter(
        model=model,
        prior=prior,
        fit_pars={
            "maxfev": 2000,
            "xtol": 5.0e-5,
            "ftol": 5.0e-5,
        },
    )
    guesser = ngmix.guessers.TPSFFluxGuesser(
        rng=rng,
        T=0.25,
        prior=prior,
    )
    runner = ngmix.runners.Runner(
        fitter=fitter,
        guesser=guesser,
        ntry=2,
    )

    return runner


def _make_prior(rng, scale, nband):
    """make the prior for the fitter.

    Parameters
    ----------
    rng: np.random.RandomState
        The random number generator
    scale: float
        Pixel scale
    nband: int
        number of bands
    """
    g_prior = ngmix.priors.GPriorBA(sigma=0.3, rng=rng)
    cen_prior = ngmix.priors.CenPrior(
        cen1=0,
        cen2=0,
        sigma1=scale,
        sigma2=scale,
        rng=rng,
    )
    T_prior = ngmix.priors.TwoSidedErf(
        minval=-10.0,
        width_at_min=0.03,
        maxval=1.0e6,
        width_at_max=1.0e5,
        rng=rng,
    )
    F_prior = ngmix.priors.TwoSidedErf(
        minval=-1.0e4,
        width_at_min=1.0,
        maxval=1.0e9,
        width_at_max=0.25e8,
        rng=rng,
    )
    F_prior = [F_prior] * nband

    prior = ngmix.joint_prior.PriorSimpleSep(
        cen_prior=cen_prior,
        g_prior=g_prior,
        T_prior=T_prior,
        F_prior=F_prior,
    )

    return prior
