import matplotlib.pyplot as plt
import numpy as np
import time
import multiprocessing
import functools
from diskcache import Cache

import astropy.units as u
from astropy.time import Time
from astroquery.jplhorizons import Horizons
from poliastro.bodies import Sun, Earth
from poliastro.frames import Planes
from poliastro.ephem import Ephem
from poliastro.twobody import Orbit

import fh_c3
from lambert import lambert


# Configure cache directory
cache = Cache(".cache")

# Maximum number of epochs per request
JPL_MAX_SIZE = 25


def obj_type(obj: str):
    if obj[0].lower() == "e":
        return "earth"
    elif obj[0].lower() == "c":
        return "comet"
    elif obj[0].lower() == "l":
        return "lagrange"
    elif obj[0:4].lower() == "jwst":
        return "jwst"
    elif obj.lower() == "ref":
        return "ref"
    else:
        raise ValueError(f"Unknown object type {obj}")


# Get earth R and V
@cache.memoize()
def earth_states(times: Time):
    r = list()
    v = list()

    for i in range(0, len(times), JPL_MAX_SIZE):
        queried_times = times[i : i + JPL_MAX_SIZE]
        earth_ephem = Ephem.from_body(
            Earth, epochs=queried_times, attractor=Sun, plane=Planes.EARTH_ECLIPTIC
        )
        r1, v1 = earth_ephem.rv()
        r.extend(r1)
        v.extend(v1)

    return r, v


@cache.memoize()
def get_states_horizon(name: str, times: Time):
    r = list()
    v = list()

    for i in range(0, len(times), JPL_MAX_SIZE):
        queried_times = times[i : i + JPL_MAX_SIZE]

        # Lock is currently not working, but would allow multiprocessing on
        # first run by only allowing once process to query Horizons at a given time.

        # try:
        #     lock.acquire()
        # except NameError:
        #     pass
        ephem = Ephem.from_horizons(
            name=name,
            epochs=queried_times,
            attractor=Sun,
            plane=Planes.EARTH_ECLIPTIC,
        )
        # try:
        #     lock.release()
        # except NameError:
        #     pass
        r1, v1 = ephem.rv()
        r.extend(r1)
        v.extend(v1)

    return u.Quantity(r), u.Quantity(v)


def lagrange_states(times: Time, L: int):
    # SEMB = Sun Earth-Moon Barycenter lagrange points
    return get_states_horizon(f"SEMB-L{L}", times)


def jwst_states(times: Time):
    # Search for JWST COSPAR ID
    return get_states_horizon("2021-130A", times)


class Porkchop:
    def __init__(
        self,
        origin: str,
        target: str,
        t_peri: Time,
        search_days: int = 200,
        search_resolution: int = 5,
        min_tof: float = 100,
        max_tof: float = 1000,
        tof_count: int = 100,
        elements: dict = None,
    ):
        # origin/target format: Earth, L1/2/3/4/5, C/XXXX (comet name), JWST
        self.origin = origin
        self.target = target
        self.t_peri = t_peri
        self.search_days = search_days
        self.search_resolution = search_resolution
        self.min_tof = min_tof
        self.max_tof = max_tof
        self.tof_count = tof_count
        self.elements = elements

    def process(self) -> dict:
        start = self.t_peri - self.search_days * u.day
        end = self.t_peri + self.search_days * u.day

        tofs = np.linspace(self.min_tof, self.max_tof, self.tof_count) * u.day

        # Get target states
        if obj_type(self.target) == "comet":
            obj = Horizons(
                id=self.target,
                location="@10",  # Origin at the sun
                epochs={
                    "start": start.iso,
                    "stop": end.iso,
                    "step": f"{self.search_resolution}d",
                },
                id_type="designation",
            )
            vec = obj.vectors(refplane="ecliptic")

            ts_arrive = Time(vec["datetime_jd"], format="jd", scale="tdb")

            rs_target = u.Quantity(vec["x", "y", "z"], unit=vec["x"].unit)
            vs_target = u.Quantity(
                [vec[x] for x in ["vx", "vy", "vz"]], unit=vec["vx"].unit
            ).T
        elif obj_type(self.target) == "lagrange":
            count = int((end.jd - start.jd) / self.search_resolution) + 1
            ts_arrive = Time(
                np.linspace(start.jd, end.jd, count), format="jd", scale="tdb"
            )

            rs_target, vs_target = lagrange_states(ts_arrive, int(self.target[1]))
        elif obj_type(self.target) == "jwst":
            count = int((end.jd - start.jd) / self.search_resolution) + 1
            ts_arrive = Time(
                np.linspace(start.jd, end.jd, count), format="jd", scale="tdb"
            )

            rs_target, vs_target = jwst_states(ts_arrive)
        elif obj_type(self.target) == "ref":
            count = int((end.jd - start.jd) / self.search_resolution) + 1
            ts_arrive = Time(
                np.linspace(start.jd, end.jd, count), format="jd", scale="tdb"
            )

            orbit = Orbit.from_classical(
                Sun, **self.elements, epoch=self.t_peri, plane=Planes.EARTH_ECLIPTIC
            )

            states = [orbit.propagate(t) for t in ts_arrive]
            rs_target = u.Quantity([s.r for s in states])
            vs_target = u.Quantity([s.v for s in states])
        else:
            raise ValueError("Invalid target")

        v_launch = np.zeros((len(ts_arrive), len(tofs))) * (u.km / u.s)
        v_arrive = np.zeros((len(ts_arrive), len(tofs))) * (u.km / u.s)

        # Loop through arrival times
        for i, t_arrive in enumerate(ts_arrive):
            # Get origin states
            ts_origin = t_arrive - tofs

            if obj_type(self.origin) == "earth":
                rs_origin, vs_origin = earth_states(ts_origin)
            elif obj_type(self.origin) == "lagrange":
                rs_origin, vs_origin = lagrange_states(ts_origin, int(self.origin[1]))
            elif obj_type(self.origin) == "jwst":
                rs_origin, vs_origin = jwst_states(ts_origin)
            else:
                raise ValueError("Invalid origin")

            # Loop through launch times
            for j in range(len(tofs)):
                # Run lambert solver in each direction
                v_i_1, v_f_1 = lambert(
                    Sun.k, rs_origin[j], rs_target[i], tofs[j], direction=True
                )
                v_i_2, v_f_2 = lambert(
                    Sun.k, rs_origin[j], rs_target[i], tofs[j], direction=False
                )

                # Find v_inf in each direction
                v_inf_origin_1 = np.linalg.norm(v_i_1 - vs_origin[j])
                v_inf_target_1 = np.linalg.norm(v_f_1 - vs_target[i])

                v_inf_origin_2 = np.linalg.norm(v_i_2 - vs_origin[j])
                v_inf_target_2 = np.linalg.norm(v_f_2 - vs_target[i])

                # Choose most efficient orbit and compute C3
                if v_inf_origin_1 > v_inf_origin_2:
                    v_launch[i, j] = v_inf_origin_2
                    v_arrive[i, j] = v_inf_target_2
                else:
                    v_launch[i, j] = v_inf_origin_1
                    v_arrive[i, j] = v_inf_target_1

        self.v_launch: u.Quantity = v_launch
        self.v_arrive: u.Quantity = v_arrive
        self.tofs = tofs
        self.ts_arrive = ts_arrive

        self.c3: u.Quantity = self.v_launch**2

        return self.v_launch, self.v_arrive, self.c3

    def graph(self, filename: str):
        c3 = (self.v_launch.to_value(u.km / u.s)) ** 2
        v_inf = self.v_arrive.to_value(u.km / u.s)

        c3[c3 > 200] = np.nan
        v_inf[v_inf > 60] = np.nan
        t = self.ts_arrive.to_datetime()

        plt.figure(figsize=(6, 6), dpi=300)

        CS = plt.contour(t, self.tofs, c3.T, levels=10, colors="blue", linewidths=0.5)
        plt.clabel(CS, CS.levels, inline=True)

        # CS = plt.contour(t, self.tofs, v_inf.T, levels=10, colors="red", linewidths=0.5)
        # plt.clabel(CS, CS.levels, inline=True)

        # Dummy points for labels
        plt.plot(t[0], self.tofs[0], lw=0.5, color="b", label="C3 (km2/s2)")
        # plt.plot(t[0], self.tofs[0], lw=0.5, color="r", label="v_inf (km/s)")

        plt.legend(loc=4)
        plt.title(f"{self.origin} - {self.target}")
        plt.xlabel("Arrival Date")
        plt.ylabel("TOF (days)")
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def print_results(self):
        best_idx = np.nanargmin(self.c3)
        best_idx = np.unravel_index(best_idx, self.c3.shape)

        c3 = self.c3[best_idx]
        payload = fh_c3.fh_expendable_mass(c3)

        print(f"Origin:                     {self.origin}")
        print(f"Target:                     {self.target}")
        print(f"Best arrival time:          {self.ts_arrive[best_idx[0]].iso}")
        print(
            f"Time of flight (days):      {self.tofs[best_idx[1]].to_value(u.d):0.0f}"
        )
        print(
            f"Arrival velocity (km/s):    {self.v_arrive[best_idx].to_value(u.km/u.s):0.2f}"
        )

        if self.origin == "Earth":
            print(f"C3 (km2/s2):                {c3.to_value(u.km**2/u.s**2):0.2f}")
            print(f"FH payload (kg):            {payload:0.0f}")
        else:
            print(
                f"Departure velocity (km/s):  {self.v_launch[best_idx].to_value(u.km/u.s):0.2f}"
            )

        print()


def analyze_lagrange_points():
    for L in [1, 2, 4, 5]:
        p = Porkchop(
            "Earth",
            f"L{L}",
            Time("2460000", format="jd"),
            search_days=0,
            min_tof=1,
            max_tof=500,
            tof_count=2000,
        )
        p.process()

        best_idx = np.nanargmin(p.v_launch + p.v_arrive)
        best_idx = np.unravel_index(best_idx, p.v_arrive.shape)
        payload = fh_c3.fh_expendable_mass(p.c3[best_idx])

        plt.figure(dpi=200)
        plt.suptitle(f"Earth - L{L} Transfer", fontsize="x-large")
        plt.title(
            f"Best: {p.tofs[best_idx[1]]:0.2f}, C3={p.c3[best_idx]:0.3f}\n{payload:0.1f} kg, V_arrive={p.v_arrive[best_idx]:0.2f}"
        )
        plt.xlabel("Time of Flight (days)")
        plt.grid()

        plt.plot(p.tofs, p.v_arrive.T, color="tab:blue")
        plt.ylabel("Arrival Velocity (km/s)", color="tab:blue")
        plt.tick_params(axis="y", labelcolor="tab:blue")

        plt.twinx()
        plt.plot(p.tofs, fh_c3.fh_expendable_mass(p.c3.T), color="tab:red")
        plt.ylabel("Falcon Heavy Payload (kg)", color="tab:red")
        plt.tick_params(axis="y", labelcolor="tab:red")

        plt.tight_layout()
        plt.savefig(f"output/earth-L{L}.png")


def analyze_orbit(origin: str, name: str, epoch: Time, elements: dict = None):
    p = Porkchop(
        origin,
        name,
        epoch,
        elements=elements,
        search_days=50,
        search_resolution=1,
        max_tof=500,
    )
    p.process()
    p.graph(f"output/{origin}-{name.replace('/', '')}.png")
    p.print_results()


def jwst_sweep_work_func(offset: float, origin: str, elements: dict):
    # print(f"Start {offset}")
    # Intercept is always within 20 days of perihelion
    p = Porkchop(
        origin,
        "REF",
        epoch + offset * u.day,
        search_resolution=0.5,
        search_days=25,
        max_tof=500,
        elements=elements,
    )
    p.process()
    print(f"Done {offset}")

    return p


# Initialize lock in each subprocess
def pool_init(l):
    global lock
    lock = l


def jwst_comet_sweep(
    epoch: Time, origin: str = "JWST", elements: dict = None, points: int = 24
):
    v_depart = []
    v_arr = []
    tofs = []
    t_arr_peri = []

    offsets = np.linspace(0, 365, points + 1)

    l = multiprocessing.Lock()

    # Only safe to use multiple processes once ephemerides are cached (ie, not on first run)
    PROCESSES = 1

    with multiprocessing.Pool(PROCESSES, initializer=pool_init, initargs=(l,)) as pool:
        work_func = functools.partial(
            jwst_sweep_work_func, origin=origin, elements=elements
        )

        for p in pool.map(work_func, offsets):
            best_idx = np.nanargmin(p.v_launch)
            best_idx = np.unravel_index(best_idx, p.v_launch.shape)

            v_depart.append(p.v_launch[best_idx].to_value(u.km / u.s))
            v_arr.append(p.v_arrive[best_idx].to_value(u.km / u.s))
            tofs.append(p.tofs[best_idx[1]].to_value(u.day))
            t_arr_peri.append((p.ts_arrive[best_idx[0]] - p.t_peri).to_value(u.day))

    plt.figure(dpi=300)
    plt.grid()
    plt.title("Orbit Phase")

    plt.plot(offsets, v_depart, color="tab:blue")
    plt.xlabel("Offset (days)")
    plt.ylabel("Departure Velocity (km/s)", color="tab:blue")
    plt.tick_params(axis="y", labelcolor="tab:blue")

    plt.twinx()
    plt.plot(offsets, tofs, color="tab:red")
    plt.ylabel("Time of Flight (days)", color="tab:red")
    plt.tick_params(axis="y", labelcolor="tab:red")

    plt.tight_layout()
    plt.savefig("jwst_phase.png")
    plt.close()

    plt.scatter(offsets, t_arr_peri)
    plt.savefig("jwst_arrival_peri.png")
    plt.close()

    best_idx = np.nanargmin(v_depart)
    worst_idx = np.nanargmax(v_depart)

    print(
        f"Best: v_depart = {v_depart[best_idx]:0.2f} km/s, v_arr = {v_arr[best_idx]:0.2f} km/s"
    )
    print(
        f"Worst: v_depart = {v_depart[worst_idx]:0.2f} km/s, v_arr = {v_arr[worst_idx]:0.2f} km/s"
    )
    print(
        f"Average: v_depart = {np.mean(v_depart):0.2f} km/s, v_arr = {np.mean(v_arr):0.2f} km/s"
    )


def compute_node_crossing(elements: dict):
    f = 2 * np.pi - elements["argp"].to_value(u.rad)
    e = elements["ecc"].to_value(u.one)

    if f < np.pi:
        E = np.arccos((e + np.cos(f)) / (e * np.cos(f) + 1))
    else:
        E = 2 * np.pi - np.arccos((e + np.cos(f)) / (e * np.cos(f) + 1))

    M = E - e * np.sin(E)

    n = np.sqrt(Sun.k / elements["a"] ** 3)

    print(f"Node crossing at Tp + {(M / n).to(u.day):0.2f}")


if __name__ == "__main__":
    # analyze_lagrange_points()

    elements = {
        "a": 10474.06 * u.au,  # Semi-major axis
        "ecc": 0.9998956528860143 * u.one,  # Eccentricity
        "inc": 100.88290133343503 * u.deg,  # Inclination
        "raan": 232.4318366430011 * u.deg,  # Right ascension of ascending node
        "argp": 335.5337773673682 * u.deg,  # Argument of periapsis
        "nu": 0 * u.deg,  # True anomaly, 0 because epoch at perihelion
    }

    # Using an earlier time than the given data because horizons doesn't have
    # data that far in the future
    epoch = Time("2025-06-15T12:00:00", format="isot", scale="utc")

    jwst_comet_sweep(epoch, elements=elements)

    compute_node_crossing(elements)

    # analyze_orbit("Earth", "JWST", epoch)
    # analyze_orbit("Earth", "REF", epoch, elements)
    # analyze_orbit("L2", "REF", epoch, elements)
    # analyze_orbit("JWST", "REF", epoch, elements)
