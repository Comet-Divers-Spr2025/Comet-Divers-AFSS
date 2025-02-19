import astropy.units as u

try:
    import comet_tools
    rust_lambert = comet_tools.lambert
except ImportError:
    rust_lambert = None

import time
import numpy as np
from poliastro.iod import lambert as python_lambert


def lambert(
    mu: u.Quantity, r0: u.Quantity, r1: u.Quantity, tof: u.Quantity, direction: bool
):
    if rust_lambert is not None:
        try:
            v_i, v_f = rust_lambert(
                r0.to_value(u.km),
                r1.to_value(u.km),
                tof.to_value(u.s),
                mu.to_value(u.km**3 / u.s**2),
                direction,
                1e-8,
                35,
            )
        except:
            v_i = np.array([np.nan] * 3)
            v_f = np.array([np.nan] * 3)

        v_i = u.Quantity(v_i, u.km / u.s)
        v_f = u.Quantity(v_f, u.km / u.s)
    else:
        v_i, v_f = python_lambert(mu, r0, r1, tof, prograde=direction)

    return v_i, v_f


if __name__ == "__main__":
    N = 1000

    start = time.time()

    for _ in range(N):
        rust_lambert(
            [-1.48388777e08, -1.61790846e07, 1.98691395e03],
            [-5.10925714e07, -2.16051165e08, 4.38815199e08],
            8640000.0,
            132712442099.00002,
            True,
            1e-8,
            35,
        )

    end = time.time()
    dur = end - start

    print(f"{1e9 * dur/N} ns/iter")
