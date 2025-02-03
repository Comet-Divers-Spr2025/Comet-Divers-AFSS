import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

import astropy.units as u
from astropy.time import Time
from astroquery.jplhorizons import Horizons

from poliastro.bodies import Sun, Earth
from poliastro.frames import Planes

from poliastro.ephem import Ephem

from lambert import lambert


def generate_porkchop(name: str, epoch: Time, date_range: float = 200):
    start = epoch - date_range * u.day
    stop = epoch + date_range * u.day

    obj = Horizons(
        id=name,
        location="@10",
        epochs={"start": start.iso, "stop": stop.iso, "step": "5d"},
        id_type="designation",
    )
    vec = obj.vectors(refplane="ecliptic")

    tofs = np.linspace(100, 1000, 200) * u.day
    epochs = Time(vec["datetime_jd"], format="jd", scale="tdb")

    results = []

    start = time.time()
    for i, t_comet in enumerate(epochs):
        # Get comet states
        r_comet = u.Quantity(list(vec[i]["x", "y", "z"]), unit=vec["x"].unit)
        v_comet = u.Quantity(list(vec[i]["vx", "vy", "vz"]), unit=vec["vx"].unit)

        res = []

        # Get earth states
        ts_earth = t_comet - tofs
        earth_ephem = Ephem.from_body(
            Earth, ts_earth, attractor=Sun, plane=Planes.EARTH_ECLIPTIC
        )
        rs_earth, vs_earth = earth_ephem.rv(ts_earth)

        for j in range(len(tofs)):
            # Run lambert solver in each direction
            v_i_1, v_f_1 = lambert(Sun.k, rs_earth[j], r_comet, tofs[j], direction=True)
            v_i_2, v_f_2 = lambert(
                Sun.k, rs_earth[j], r_comet, tofs[j], direction=False
            )

            # Find v_inf in each direction
            v_inf_e_1 = np.linalg.norm(v_i_1 - vs_earth[j])
            v_inf_c_1 = np.linalg.norm(v_f_1 - v_comet)

            v_inf_e_2 = np.linalg.norm(v_i_2 - vs_earth[j])
            v_inf_c_2 = np.linalg.norm(v_f_2 - v_comet)

            # Choose most efficient orbit and compute C3
            if v_inf_e_1 > v_inf_e_2:
                v_inf_arr = v_inf_c_2
                c3 = np.linalg.norm(v_inf_e_2) ** 2
            else:
                v_inf_arr = v_inf_c_1
                c3 = np.linalg.norm(v_inf_e_1) ** 2

            # Store in results array
            res.append([c3.to_value(u.km**2 / u.s**2), v_inf_arr.to_value(u.km / u.s)])

        results.append(res)
    stop = time.time()

    results = np.array(results)

    return results, epochs, tofs


# Generate a porkchop contour plot from lambert grid results
def graph_porkchop(results, epochs, epoch, tofs, name, filename: str):
    c3 = results[:, :, 0]
    v_inf = results[:, :, 1]

    c3[c3 > 200] = np.nan
    v_inf[v_inf > 60] = np.nan

    plt.figure(figsize=(6, 6), dpi=300)

    t = [e.mjd for e in epochs]

    CS = plt.contour(t, tofs, c3.T, levels=10, colors="blue", linewidths=0.5)
    plt.clabel(CS, CS.levels, inline=True)

    CS = plt.contour(t, tofs, v_inf.T, levels=10, colors="red", linewidths=0.5)
    plt.clabel(CS, CS.levels, inline=True)

    # Dummy points for labels
    plt.plot(t[0], tofs[0], lw=0.5, color="b", label="C3 (km2/s2)")
    plt.plot(t[0], tofs[0], lw=0.5, color="r", label="v_inf (km/s)")

    plt.legend(loc=4)
    plt.title(name)
    plt.xlabel("Arrival Date (MJD)")
    plt.ylabel("TOF (days)")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

    idx = np.unravel_index(np.nanargmin(c3), c3.shape)
    best = results[idx]

    print(name)
    print(
        f"Best trajectory: arrival {epochs[idx[0]].mjd:.2f} MJD C3={best[0]:0.2f} km2/s2, v_inf={best[1]:0.2f} km/s, tof={tofs[idx[1]]:0.1f} days"
    )
    print(f"Days after perihelion: {epochs[idx[0]].mjd - epoch.mjd:.2f}")
    print()


def run(name: str, id: str, epoch: Time):
    results, epochs, tofs = generate_porkchop(id, epoch, date_range=500)
    graph_porkchop(
        results,
        epochs,
        epoch,
        tofs,
        name,
        f"output/{name.replace('/', ' ')}_porkchop.png",
    )


if __name__ == "__main__":
    df = pd.read_csv("comets.csv")

    # Run for each comet in CSV
    for row in df.iterrows():
        comet = row[1]
        name = comet["full_name"]
        id = comet["spkid"]
        tp = comet["tp"]

        run(name, id, Time(tp, format="jd"))
