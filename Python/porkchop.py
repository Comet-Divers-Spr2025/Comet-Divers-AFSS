import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
from poliastro.plotting import StaticOrbitPlotter

import fh_c3
from lambert import lambert


# Configure cache directory
cache = Cache(".cache")

# Maximum number of epochs per request
JPL_MAX_SIZE = 25


def vector_angle_deg(v1, v2):
    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)
    dot = np.dot(v1, v2)
    angle = np.arccos(dot)

    return np.rad2deg(angle)


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

    return u.Quantity(r), u.Quantity(v)


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

        # Reverse output if necessary, since horizons returns results in order
        if queried_times[0].mjd > queried_times[-1].mjd:
            r.extend(r1[::-1])
            v.extend(v1[::-1])
        else:
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
        node_crossing: Time = None,
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
        self.node_crossing = node_crossing
        self.search_days = search_days
        self.search_resolution = search_resolution
        self.min_tof = min_tof
        self.max_tof = max_tof
        self.tof_count = tof_count
        self.elements = elements

    def process(self) -> dict:
        if self.node_crossing:
            start = self.node_crossing - self.search_days * u.day
            end = self.node_crossing + self.search_days * u.day
        else:
            start = self.t_peri - self.search_days * u.day
            end = self.t_peri + self.search_days * u.day

        tofs = (
            np.linspace(self.min_tof, self.max_tof, self.tof_count, endpoint=False)
            * u.day
        )

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

        initial_states = np.zeros((len(ts_arrive), len(tofs), 6))

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

                    initial_states[i, j] = np.concatenate(
                        [rs_origin[j].to_value(u.km), v_i_2.to_value(u.km / u.s)]
                    )
                else:
                    v_launch[i, j] = v_inf_origin_1
                    v_arrive[i, j] = v_inf_target_1

                    initial_states[i, j] = np.concatenate(
                        [rs_origin[j].to_value(u.km), v_i_1.to_value(u.km / u.s)]
                    )

        self.v_launch: u.Quantity = v_launch
        self.v_arrive: u.Quantity = v_arrive
        self.tofs = tofs
        self.ts_arrive = ts_arrive

        self.c3: u.Quantity = self.v_launch**2

        self.initial_states = initial_states

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


def plot_orbit(
    t_arr: Time,
    comet_orbit: Orbit,
    sc_orbit: Orbit,
    filename: str,
    title: str = None,
    earth_plane: bool = True,
):
    comet_orbit = comet_orbit.propagate(t_arr)
    sc_orbit = sc_orbit.propagate(t_arr)

    _, ax = plt.subplots(figsize=(4, 9), dpi=150)

    plotter = StaticOrbitPlotter(ax, plane=Planes.EARTH_ECLIPTIC)

    if earth_plane:
        plotter.plot_body_orbit(Earth, t_arr, color="g", trail=True)

        plotter._num_points = 10000
        plotter.plot(comet_orbit, label="Comet", color="b", trail=False)
        plotter._num_points = 150

        plotter.plot(sc_orbit, label="Spacecraft", color="r", trail=True)
    else:
        plotter._num_points = 10000
        plotter.plot(comet_orbit, label="Comet", color="b", trail=False)
        plotter._num_points = 150

        plotter.plot_body_orbit(Earth, t_arr, color="g", trail=True)
        plotter.plot(sc_orbit, label="Spacecraft", color="r", trail=True)

    bound = (1.25 * u.AU).to_value(u.km)
    plt.xlim([-1.2 * bound, 1.2 * bound])
    plt.ylim([-bound, bound])

    if title:
        plt.title(title)

    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{filename}_orbits.png")


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


def analyze_orbit(
    origin: str, name: str, epoch: Time, elements: dict = None, tof_count: int = 400
):
    p = Porkchop(
        origin,
        name,
        epoch,
        elements=elements,
        node_crossing=epoch + 20.6 * u.day,
        search_days=5,
        search_resolution=5,
        max_tof=500,
        tof_count=tof_count,
    )
    p.process()
    p.graph(f"output/{origin}-{name.replace('/', '')}.png")
    p.print_results()


def jwst_sweep_work_func(longitude: float, epoch: Time, origin: str, elements: dict):
    start = time.monotonic()

    elements["raan"] = longitude * u.deg
    node_crossing = epoch + compute_node_crossing(elements)

    # Intercept is always within 20 days of perihelion
    p = Porkchop(
        origin,
        "REF",
        epoch,
        node_crossing=node_crossing,
        search_resolution=0.1,
        search_days=2,
        min_tof=0.25 * 365,
        max_tof=2 * 365,
        tof_count=200,
        elements=elements,
    )
    p.process()
    end = time.monotonic()

    print(f"Longitude {longitude:0.2f} took {end-start:0.2f}s")

    return p


# Initialize lock in each subprocess
def pool_init(l):
    global lock
    lock = l


def jwst_comet_sweep(
    epoch: Time,
    origin: str = "JWST",
    elements: dict = None,
    points: int = 90,
    base_name: str = "ref",
):
    v_depart = []
    v_arr = []
    tofs = []
    t_arr = []
    t_arr_peri = []
    earth_sc = []
    sun_sc = []
    sun_phase_angle = []

    comet_orbits = []
    sc_orbits = []

    longitudes = np.linspace(0, 360, points, endpoint=False)

    l = multiprocessing.Lock()

    # Only safe to use multiple processes once ephemerides are cached (ie, not
    # on first run with given settings)
    PROCESSES = 4

    with multiprocessing.Pool(PROCESSES, initializer=pool_init, initargs=(l,)) as pool:
        work_func = functools.partial(
            jwst_sweep_work_func, epoch=epoch, origin=origin, elements=elements
        )

        for i, p in enumerate(pool.map(work_func, longitudes)):
            # Find best solution based on dV to leave L2
            best_idx = np.nanargmin(p.v_launch)
            best_idx = np.unravel_index(best_idx, p.v_launch.shape)
            best_time = p.ts_arrive[best_idx[0]]
            best_tof = p.tofs[best_idx[1]]

            # Record all relevent parameters
            v_depart.append(p.v_launch[best_idx].to_value(u.km / u.s))
            v_arr.append(p.v_arrive[best_idx].to_value(u.km / u.s))
            tofs.append(best_tof.to_value(u.day))
            t_arr.append(best_time)
            t_arr_peri.append((best_time - p.t_peri).to_value(u.day))

            # Get position of earth at flyby
            earth_r, _ = earth_states(Time([best_time] * 2))
            earth_r = earth_r[0]

            elements["raan"] = longitudes[i] * u.deg

            # Get position of comet
            comet_orbit = Orbit.from_classical(
                Sun,
                **elements,
                epoch=epoch,
                plane=Planes.EARTH_ECLIPTIC,
            )

            # Get position of spacecraft
            sc_state = p.initial_states[best_idx]
            sc_orbit = Orbit.from_vectors(
                Sun,
                sc_state[:3] * u.km,
                sc_state[3:] * u.km / u.s,
                best_time - best_tof,
                Planes.EARTH_ECLIPTIC,
            )

            # Use position 24h before flyby to compute phase angle
            minus_day = best_time - 1 * u.day
            comet_r = u.Quantity(comet_orbit.propagate(minus_day).r)
            sc_approach = u.Quantity(sc_orbit.propagate(minus_day).r)

            sun_phase_angle.append(
                vector_angle_deg(-comet_r, sc_approach - comet_r).to_value(u.deg)
            )

            # Compute Earth-SC and Sun-SC distances
            earth_sc.append(np.linalg.norm(sc_approach - earth_r).to_value(u.au))
            sun_sc.append(np.linalg.norm(comet_r).to_value(u.au))

            comet_orbits.append(comet_orbit)
            sc_orbits.append(sc_orbit)

    # Graph dV and ToF vs longitude
    plt.figure(dpi=300)
    plt.grid()
    plt.title("Orbit Phase")

    plt.plot(longitudes, v_depart, color="tab:blue")
    plt.xlabel("Longitude of Ascending Node (deg)")
    plt.ylabel("Departure Velocity (km/s)", color="tab:blue")
    plt.tick_params(axis="y", labelcolor="tab:blue")

    plt.twinx()
    plt.plot(longitudes, tofs, color="tab:red")
    plt.ylabel("Time of Flight (days)", color="tab:red")
    plt.tick_params(axis="y", labelcolor="tab:red")

    plt.tight_layout()
    plt.savefig(f"{base_name}_phase.png")
    plt.close()

    # Graph arrival time vs longitude
    plt.figure(dpi=300)
    plt.plot(longitudes, t_arr_peri)
    plt.savefig(f"{base_name}_arrival_peri.png")
    plt.close()

    # Graph distance vs longitude
    plt.figure(dpi=300)
    plt.title("Distance at Comet Flyby")
    plt.plot(longitudes, earth_sc, label="Earth-SC")
    plt.plot(longitudes, sun_sc, label="Sun-SC")
    plt.legend()
    plt.xlabel("Longitude of Ascending Node (deg)")
    plt.ylabel("Distance (AU)")
    plt.grid()

    plt.tight_layout()
    plt.savefig(f"{base_name}_distance.png")
    plt.close()

    # Graph sun phase angle vs longitude
    plt.figure(dpi=300)
    plt.plot(longitudes, sun_phase_angle)
    plt.title("Sun-Comet-S/C Phase Angle, 1 Day Before Flyby")
    plt.xlabel("Longitude of Ascending Node (deg)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{base_name}_phase_angle.png")

    # Graph percent of achievable phasings vs dv

    prop_mass_fraction = 0.6
    isp = 303
    dv = isp * 9.81 * np.log(1 / (1 - prop_mass_fraction)) / 1e3

    dv_sorted = np.sort(v_depart)
    closest_idx = np.argmin(np.abs(dv_sorted - dv))

    plt.figure(dpi=300)
    plt.plot(np.linspace(0, 100, len(dv_sorted)), dv_sorted)
    plt.title("Percent of Phasings Reachable vs Delta V")
    plt.xlabel("Percent Reachable")
    plt.ylabel("Delta V (km/s)")
    plt.axhline(dv, color="r", linestyle="--")
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"{base_name}_percent_reachable.png")

    best_idx = np.nanargmin(v_depart)
    worst_idx = np.nanargmax(v_depart)

    # Graph orbits
    plot_orbit(
        t_arr[best_idx],
        comet_orbits[best_idx],
        sc_orbits[best_idx],
        base_name + "_best",
        "Best Trajectory - Earth View",
        earth_plane=True,
    )
    plot_orbit(
        t_arr[best_idx],
        comet_orbits[best_idx],
        sc_orbits[best_idx],
        base_name + "_best_comet",
        "Best Trajectory - Comet View",
        earth_plane=False,
    )
    plot_orbit(
        t_arr[worst_idx],
        comet_orbits[worst_idx],
        sc_orbits[worst_idx],
        base_name + "_worst",
        "Worst Trajectory",
    )

    # Print everything

    print("\nBest:")
    print(f"\tv_depart = {v_depart[best_idx]:0.2f} km/s")
    print(f"\tv_arr = {v_arr[best_idx]:0.2f} km/s")
    print(f"\tr_earth_sc = {earth_sc[best_idx]:0.3f} AU")
    print(f"\tr_sun_sc = {sun_sc[best_idx]:0.3f} AU")
    print(f"\tphase angle = {sun_phase_angle[best_idx]:0.1f} deg")

    print(f"\nAverage:")
    print(f"\tv_depart = {np.mean(v_depart):0.2f} km/s")
    print(f"\tv_arr = {np.mean(v_arr):0.2f} km/s")
    print(f"\tr_earth_sc = {np.mean(earth_sc):0.3f} AU")
    print(f"\tr_sun_sc = {np.mean(sun_sc):0.3f} AU")
    print(f"\tphase angle = {np.mean(sun_phase_angle):0.1f} deg")

    print("\nWorst:")
    print(f"\tv_depart = {v_depart[worst_idx]:0.2f} km/s")
    print(f"\tv_arr = {v_arr[worst_idx]:0.2f} km/s")
    print(f"\tr_earth_sc = {earth_sc[worst_idx]:0.3f} AU")
    print(f"\tr_sun_sc = {sun_sc[worst_idx]:0.3f} AU")
    print(f"\tphase angle = {sun_phase_angle[worst_idx]:0.1f} deg")

    print(f"\nCan reach {closest_idx/len(v_depart):.1%} with {dv:0.2f} km/s")

    print()


def compute_node_crossing(elements: dict):
    f = 2 * np.pi - elements["argp"].to_value(u.rad)
    e = elements["ecc"].to_value(u.one)

    if f < np.pi:
        E = np.arccos((e + np.cos(f)) / (e * np.cos(f) + 1))
    else:
        E = 2 * np.pi - np.arccos((e + np.cos(f)) / (e * np.cos(f) + 1))

    M = E - e * np.sin(E)

    n = np.sqrt(Sun.k / elements["a"] ** 3)

    return (M / n).to(u.day)


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
    # data that far in the future for JWST
    epoch = Time("2025-06-15T12:00:00", format="isot", scale="tdb")

    jwst_comet_sweep(epoch, elements=elements)

    # analyze_orbit("Earth", "JWST", epoch)
    # analyze_orbit("Earth", "REF", epoch, elements)
    # analyze_orbit("L2", "REF", epoch, elements)
    # analyze_orbit("JWST", "REF", epoch, elements)
