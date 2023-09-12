import time

import joblib
import numpy as np
import tqdm

from metacoadd.simu import make_sim, run_metacoadd, run_metadetect

# input_headers_dir = '../data/pre_selection_1/'
# input_headers_dir = '../data/input_headers/'
# input_headers_dir = '../data/pre_selection_2/'
# input_headers_dir = '../data/pre_selection_3/'

# Those 2 tests does not work at the moment due to CCDs that are upside down.
# input_headers_dir = '../data/pre_selection_1/'
# ra_center = 110.4990693800      # Deg
# dec_center = 53.0               # Deg

# input_headers_dir = '../data/pre_selection_2/'
# ra_center = 110.7181643         # Deg
# dec_center = 53.2210355         # Deg

input_headers_dir = "../data/pre_selection_3/"
ra_center = 110.1844991
dec_center = 52.8002126
scale = 0.185768447408928  # Arcsec
cell_size = 1.0 / 60.0  # Arcmin

noise = 1e-3

params_obj = {
    "n_obj": 25,  # ignored if 'grid' is used
    "hlr": 0.7,
    "flux": 100,
    "g1": 0.01,
    "g2": 0.0,
}

params_single = {
    "n_obj": 25,  # ignored if 'grid' is used
    "psf_fwhm": 0.7,
    "psf_fwhm_std": 0.0,
    "psf_g1": 0.0,
    "psf_g2": 0.0,
    "noise": noise,
}


def run_sim(
    coadd_ra,
    coadd_dec,
    coadd_scale,
    coadd_size,
    params_obj,
    params_single,
    seed,
    mdet_seed,
):
    params_obj_p = params_obj.copy()
    params_obj_p["g1"] = 0.02
    explist_p, explist_psf_p = make_sim(
        input_headers_dir,
        coadd_ra,
        coadd_dec,
        coadd_scale,
        coadd_size,
        params_obj_p,
        params_single,
        seed=seed,
        get_obj_dict=False,
    )
    (
        simplecoadd_p,
        simplecoadd_psf_p,
    ) = run_metacoadd(
        explist_p,
        coadd_ra,
        coadd_dec,
        coadd_scale,
        coadd_size,
        explist_psf=explist_psf_p,
    )
    _pres = run_metadetect(
        simplecoadd_p,
        simplecoadd_psf_p,
        mdet_seed,
    )
    if _pres is None:
        return None

    params_obj_m = params_obj.copy()
    params_obj_m["g1"] = -0.02
    explist_m, explist_psf_m = make_sim(
        input_headers_dir,
        coadd_ra,
        coadd_dec,
        coadd_scale,
        coadd_size,
        params_obj_m,
        params_single,
        seed=seed,
        get_obj_dict=False,
    )
    simplecoadd_m, simplecoadd_psf_m = run_metacoadd(
        explist_m,
        coadd_ra,
        coadd_dec,
        coadd_scale,
        coadd_size,
        explist_psf=explist_psf_m,
    )
    _mres = run_metadetect(
        simplecoadd_m,
        simplecoadd_psf_m,
        mdet_seed,
    )
    if _mres is None:
        return None

    return (_meas_shear_data(_pres, "wmom"), _meas_shear_data(_mres, "wmom"))


# From https://github.com/esheldon/metadetect/blob/master/shear_meas_test/test_shear_meas.py
def _shear_cuts(arr, model):
    if model == "wmom":
        tmin = 1.2
    else:
        tmin = 0.5
    msk = (
        (arr["flags"] == 0)
        & (arr[f"{model}_s2n"] > 10)
        & (arr[f"{model}_T_ratio"] > tmin)
    )
    return msk


def _meas_shear_data(res, model):
    msk = _shear_cuts(res["noshear"], model)
    g1 = np.mean(res["noshear"][f"{model}_g"][msk, 0])
    g2 = np.mean(res["noshear"][f"{model}_g"][msk, 1])

    msk = _shear_cuts(res["1p"], model)
    g1_1p = np.mean(res["1p"][f"{model}_g"][msk, 0])
    msk = _shear_cuts(res["1m"], model)
    g1_1m = np.mean(res["1m"][f"{model}_g"][msk, 0])
    R11 = (g1_1p - g1_1m) / 0.02

    msk = _shear_cuts(res["2p"], model)
    g2_2p = np.mean(res["2p"][f"{model}_g"][msk, 1])
    msk = _shear_cuts(res["2m"], model)
    g2_2m = np.mean(res["2m"][f"{model}_g"][msk, 1])
    R22 = (g2_2p - g2_2m) / 0.02

    dt = [("g1", "f8"), ("g2", "f8"), ("R11", "f8"), ("R22", "f8")]
    return np.array([(g1, g2, R11, R22)], dtype=dt)


def boostrap_m_c(pres, mres):
    m, c = meas_m_c_cancel(pres, mres)
    bdata = _bootstrap_stat(pres, mres, meas_m_c_cancel, 14324, nboot=500)
    merr, cerr = np.std(bdata, axis=0)
    return m, merr, c, cerr


def meas_m_c_cancel(pres, mres):
    x = np.mean(pres["g1"] - mres["g1"]) / 2
    y = np.mean(pres["R11"] + mres["R11"]) / 2
    m = x / y / 0.02 - 1

    x = np.mean(pres["g2"] + mres["g2"]) / 2
    y = np.mean(pres["R22"] + mres["R22"]) / 2
    c = x / y

    return m, c


def _bootstrap_stat(d1, d2, func, seed, nboot=500):
    dim = d1.shape[0]
    rng = np.random.RandomState(seed=seed)
    stats = []
    for _ in tqdm.trange(nboot, leave=False):
        ind = rng.choice(dim, size=dim, replace=True)
        stats.append(func(d1[ind], d2[ind]))
    return stats


def test_shear_meas(
    coadd_ra,
    coadd_dec,
    coadd_scale,
    coadd_size,
    params_obj,
    params_single,
    seed=1234,
    ntrial=50,
    njob=-1,
):
    nsub = max(ntrial // 100, 50)
    # nsub = ntrial
    nitr = ntrial // nsub
    # nitr = 1
    rng = np.random.RandomState(seed=seed)
    seeds = rng.randint(low=1, high=2**29, size=ntrial)
    mdet_seeds = rng.randint(low=1, high=2**29, size=ntrial)

    tm0 = time.time()

    print("")

    pres = []
    mres = []
    loc = 0
    for itr in tqdm.trange(nitr):
        jobs = [
            joblib.delayed(run_sim)(
                coadd_ra,
                coadd_dec,
                coadd_scale,
                coadd_size,
                params_obj,
                params_single,
                seeds[loc + i],
                mdet_seeds[loc + i],
            )
            for i in range(nsub)
        ]

        outputs = joblib.Parallel(
            n_jobs=njob,
            verbose=100,
            backend="loky",
        )(jobs)

        for out in outputs:
            if out is None:
                continue
            pres.append(out[0])
            mres.append(out[1])
        loc += nsub

        m, merr, c, cerr = boostrap_m_c(
            np.concatenate(pres),
            np.concatenate(mres),
        )
        print(
            (
                "\n"
                "nsims: %d\n"
                "m [1e-3, 3sigma]: %s +/- %s\n"
                "c [1e-5, 3sigma]: %s +/- %s\n"
                "\n"
            )
            % (
                len(pres),
                m / 1e-3,
                3 * merr / 1e-3,
                c / 1e-5,
                3 * cerr / 1e-5,
            ),
            flush=True,
        )

    total_time = time.time() - tm0
    print("time per:", total_time / ntrial, flush=True)

    pres = np.concatenate(pres)
    mres = np.concatenate(mres)
    m, merr, c, cerr = boostrap_m_c(pres, mres)

    print(
        ("\n\nm [1e-3, 3sigma]: {} +/- {}" "\nc [1e-5, 3sigma]: {} +/- {}").format(
            m / 1e-3,
            3 * merr / 1e-3,
            c / 1e-5,
            3 * cerr / 1e-5,
        ),
        flush=True,
    )


def get_args():
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
        help="seed for rng",
    )
    parser.add_argument(
        "--ntrial",
        type=int,
        default=50,
        help="number of trials",
    )
    parser.add_argument(
        "--njob",
        type=int,
        default=-1,
        help="number of jobs. '-1' for all. '-2' for all-1",
    )

    return parser.parse_args()


def main():
    """ """

    args = get_args()
    print(f"seed: {args.seed}")
    print(f"ntrial: {args.ntrial}")
    print(f"njob: {args.njob}")

    test_shear_meas(
        ra_center,
        dec_center,
        scale,
        cell_size,
        params_obj,
        params_single,
        seed=args.seed,
        ntrial=args.ntrial,
        njob=args.njob,
    )


if __name__ == "__main__":
    main()
