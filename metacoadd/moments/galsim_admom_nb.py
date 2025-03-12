"""
This an implementation of the Galsim adaptive moments algorithm in ngmix format.
To see the original implementation, please visit:
https://github.com/GalSim-developers/GalSim/blob/releases/2.7/src/hsm/PSFCorr.cpp
"""

from math import atan2, cos, exp, sin, sqrt

import ngmix
import numpy as np
from numba import njit


@njit(fastmath=True)
def find_ellipmom1(pixels, x0, y0, Mxx, Mxy, Myy, res, conf, do_cov=False):
    F = res["F"]

    detM = Mxx * Myy - Mxy * Mxy
    res["wnorm"] = 1.0 / (2 * np.pi * sqrt(detM))

    Minv_xx = Myy / detM
    TwoMinv_xy = -Mxy / detM * 2.0
    Minv_yy = Mxx / detM

    n_pixels = pixels.size
    for i_pix in range(n_pixels):
        pixel = pixels[i_pix]

        umod = pixel["u"] - x0
        vmod = pixel["v"] - y0

        Minv_xx__x_x0__x_x0 = Minv_xx * umod * umod
        TwoMinv_xy__y_y0__x_x0 = TwoMinv_xy * vmod * umod
        Minv_yy__y_y0__y_y0 = Minv_yy * vmod * vmod

        rho2 = (
            Minv_yy__y_y0__y_y0 + TwoMinv_xy__y_y0__x_x0 + Minv_xx__x_x0__x_x0
        )

        res["npix"] += 1
        if rho2 < conf["max_moment_nsig2"]:
            weight = exp(-0.5 * rho2) * res["wnorm"] * pixel["area"]
            intensity = weight * pixel["val"]

            res["wsum"] += weight

            if not do_cov:
                res["sums"][0] += umod * intensity
                res["sums"][1] += vmod * intensity
                res["sums"][2] += umod * umod * intensity
                res["sums"][3] += umod * vmod * intensity
                res["sums"][4] += vmod * vmod * intensity
                res["sums"][5] += 1.0 * intensity
                res["sums"][6] += rho2 * rho2 * intensity

            else:
                w2 = weight * weight
                var = 1.0 / (pixel["ierr"] * pixel["ierr"])
                F[0] = umod
                F[1] = vmod
                F[2] = umod * umod
                F[3] = umod * vmod
                F[4] = vmod * vmod
                F[5] = 1.0
                F[6] = rho2 * rho2
                for i in range(7):
                    res["sums"][i] += intensity * F[i]
                    for j in range(7):
                        res["sums_cov"][i, j] += w2 * var * F[i] * F[j]


@njit(fastmath=True)
def find_ellipmom2(
    pixels,
    guess,
    resarray,
    confarray,
):
    """ """

    conf = confarray[0]
    res = resarray[0]

    convergence_factor = 1.0
    shiftscale0 = 0.0

    x0, y0, Mxx, Mxy, Myy, _ = guess
    x00 = x0
    y00 = y0
    do_cov = False
    for i in range(conf["maxiter"]):
        clear_result(res)
        find_ellipmom1(pixels, x0, y0, Mxx, Mxy, Myy, res, conf, do_cov)
        Bx, By, Cxx, Cxy, Cyy, Amp, rho4 = res["sums"]

        if Amp <= 0:
            res["flags"] = ngmix.flags.NONPOS_FLUX

        two_psi = atan2(2 * Mxy, Mxx - Myy)
        semi_a2 = 0.5 * ((Mxx + Myy) + (Mxx - Myy) * cos(two_psi)) + Mxy * sin(
            two_psi
        )
        semi_b2 = Mxx + Myy - semi_a2

        if semi_b2 <= 0:
            res["flags"] = ngmix.flags.NONPOS_SIZE

        shiftscale = sqrt(semi_b2)
        if res["numiter"] == 0:
            shiftscale0 = shiftscale

        dx = 2.0 * Bx / (Amp * shiftscale)
        dy = 2.0 * By / (Amp * shiftscale)
        dxx = 4 * (Cxx / Amp - 0.5 * Mxx) / semi_b2
        dxy = 4 * (Cxy / Amp - 0.5 * Mxy) / semi_b2
        dyy = 4 * (Cyy / Amp - 0.5 * Myy) / semi_b2

        if dx > conf["bound_correct_wt"]:
            dx = conf["bound_correct_wt"]
        if dx < -conf["bound_correct_wt"]:
            dx = -conf["bound_correct_wt"]
        if dy > conf["bound_correct_wt"]:
            dy = conf["bound_correct_wt"]
        if dy < -conf["bound_correct_wt"]:
            dy = -conf["bound_correct_wt"]
        if dxx > conf["bound_correct_wt"]:
            dxx = conf["bound_correct_wt"]
        if dxx < -conf["bound_correct_wt"]:
            dxx = -conf["bound_correct_wt"]
        if dxy > conf["bound_correct_wt"]:
            dxy = conf["bound_correct_wt"]
        if dxy < -conf["bound_correct_wt"]:
            dxy = -conf["bound_correct_wt"]
        if dyy > conf["bound_correct_wt"]:
            dyy = conf["bound_correct_wt"]
        if dyy < -conf["bound_correct_wt"]:
            dyy = -conf["bound_correct_wt"]

        if abs(dx) > abs(dy):
            convergence_factor = abs(dx)
        else:
            convergence_factor = abs(dy)
        convergence_factor = convergence_factor * convergence_factor

        if abs(dxx) > convergence_factor:
            convergence_factor = abs(dxx)
        if abs(dxy) > convergence_factor:
            convergence_factor = abs(dxy)
        if abs(dyy) > convergence_factor:
            convergence_factor = abs(dyy)

        convergence_factor = sqrt(convergence_factor)
        if shiftscale < shiftscale0:
            convergence_factor *= shiftscale0 / shiftscale

        x0 += dx * shiftscale
        y0 += dy * shiftscale
        Mxx += dxx * semi_b2
        Mxy += dxy * semi_b2
        Myy += dyy * semi_b2

        if (abs(x0 - x00) > conf["shiftmax"]) | (
            abs(y0 - y00) > conf["shiftmax"]
        ):
            res["flags"] = ngmix.flags.CEN_SHIFT

        res["numiter"] = i + 1

        if convergence_factor < conf["tol"]:
            if not do_cov:
                do_cov = True
                continue
            A = Amp
            rho4 /= Amp
            res["pars"][0] = x0
            res["pars"][1] = y0
            res["pars"][2] = Mxx
            res["pars"][3] = Mxy
            res["pars"][4] = Myy
            res["pars"][5] = A * rho4
            res["pars"][6] = rho4
            break

        if res["numiter"] == conf["maxiter"]:
            res["flags"] = ngmix.flags.MAXITER


@njit
def clear_result(res):
    """
    clear some fields in the result structure
    """
    res["npix"] = 0
    res["wsum"] = 0.0
    res["sums"][:] = 0.0
    res["sums_cov"][:, :] = 0.0
    res["pars"][:] = np.nan
