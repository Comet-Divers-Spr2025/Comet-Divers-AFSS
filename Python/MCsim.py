### Impactor Monte Carlo Simulation ###

import astropy.units as u
import dataclasses
import itertools
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import time


@dataclasses.dataclass
class SimConfig:
    ## Mission parameters
    start_dist: u.Quantity["length", 1]
    start_vel: u.Quantity["velocity", 1]
    start_separation: u.Quantity["length", 1]
    tcm_times: list[u.Quantity["time"]]  # Time after start of sim
    comet_radius: float

    ## Covariances

    # Absolute position from ground tracking, used for initial estimate
    cov_pos_abs: u.Quantity
    cov_vel_abs: u.Quantity

    # Maneuver
    cov_dv: u.Quantity
    thrust_scale: float

    # Relative position from optical navigation, used for terminal guidance
    pixel_rads: float
    pixel_error_unresolved: float = 0.1
    pixel_error_resolved: float = 0.5

    ## Simulation parameters
    run_count: int = 1
    target: u.Quantity = [0, 0, 0] * u.m


# region Functions


def linear_propagation(
    r: u.Quantity["length", 3], v: u.Quantity["velocity", 3], dt: u.Quantity["time"]
):
    r = r + v * dt

    return r, v


def normal_sample(cov: u.Quantity):
    return (
        np.random.multivariate_normal(
            mean=[0] * cov.shape[0], cov=cov.to_value(cov.unit)
        )
        * (cov.unit) ** 0.5
    )


def B_plane_targeting(r, v, comet):
    r_comet = comet - r
    r_hat = r_comet / np.linalg.norm(r_comet)
    new_v = r_hat * np.linalg.norm(v)

    dv = new_v - v

    return dv


# Find point along trajectory closest to comet
def time_to_impact(r, v, comet):
    v_hat = v / np.linalg.norm(v)
    d_to_approach = np.linalg.norm(np.dot(comet - r, v_hat))
    dt = d_to_approach / np.linalg.norm(v)

    return dt


def calc_opnav_error(config: SimConfig, r: u.Quantity):
    pixel_resolution = np.linalg.norm(r) * config.pixel_rads

    # Different accuracy for point source vs resolved shape
    if pixel_resolution < 5 * 2 * config.comet_radius:
        pixel_accuracy = config.pixel_error_resolved
    else:
        pixel_accuracy = config.pixel_error_unresolved

    opnav_error = pixel_accuracy * pixel_resolution

    # print(f"Radius in pixels: {(config.comet_radius / pixel_resolution).si}")

    # Cross range error from direct sensor measurement
    error_vec = [
        0,
        np.random.normal(0, opnav_error.to_value(opnav_error.unit)),
        np.random.normal(0, opnav_error.to_value(opnav_error.unit)),
    ] * opnav_error.unit

    # In range error from parallax measurement
    parallax_angle = config.start_separation / config.start_dist
    current_sep = config.start_separation * np.linalg.norm(r) / config.start_dist
    range_error = (config.pixel_rads * pixel_accuracy) * current_sep / parallax_angle**2

    error_vec[0] = (
        np.random.normal(0, range_error.to_value(range_error.unit)) * range_error.unit
    )

    return error_vec


def MonteCarloSample(config: SimConfig):
    # Make sure random numbers are actually random
    np.random.seed(
        int(time.time() * 1e9) * id(multiprocessing.current_process()) % 2**32
    )

    comet = config.target

    trajectory = []
    dv_total = 0 * u.m / u.s
    t = 0 * u.s

    r = u.Quantity([config.start_dist, 0 * u.km, 0 * u.km]).si
    v = u.Quantity([-config.start_vel, 0 * u.km / u.s, 0 * u.km / u.s]).si

    r_est_last = r.copy()

    cov_r_abs = np.diag(config.cov_pos_abs)
    cov_v_abs = np.diag(config.cov_vel_abs)
    cov_dv = np.diag(config.cov_dv)

    # Initial position uncertainty
    r += normal_sample(cov_r_abs)
    v += normal_sample(cov_v_abs)

    trajectory.append([r.copy(), v.copy()])

    # Execute TCMs
    for tcm_time in config.tcm_times:
        # Propagate to next TCM
        dt = tcm_time - t
        t += dt
        r, v = linear_propagation(r, v, dt)

        # Run targeting algorithm and execute maneuver
        r_est = r + calc_opnav_error(config, r)
        v_est = (r_est - r_est_last) / dt
        dv = B_plane_targeting(r_est, v_est, comet)
        dv *= np.random.normal(1, config.thrust_scale)
        dv += normal_sample(cov_dv)
        dv_total += np.linalg.norm(dv)
        v += dv

        r_est_last = r_est

        trajectory.append([r.copy(), v.copy()])

    # Find closest approach distance

    dt = time_to_impact(r, v, comet)
    t_closest = t + dt
    r, v = linear_propagation(r, v, dt)
    d_closest = np.linalg.norm(r)

    trajectory.append([r.copy(), v.copy()])

    # print(f"Closest approach distance: {d_closest.to(u.km)}, {t_closest.to(u.hour)}")

    return trajectory, d_closest, dv_total


def run_batch(config: SimConfig, parallel: bool = True):
    distances = []
    trajectories = []
    dvs = []

    # Execute Monte Carlo Simulation

    if parallel:
        with multiprocessing.Pool(processes=8) as pool:
            results = pool.imap_unordered(
                MonteCarloSample,
                itertools.repeat(config, config.run_count),
                chunksize=50,
            )

            for res in results:
                trajectories.append(res[0])
                distances.append(res[1])
                dvs.append(res[2])
    else:
        for _ in range(config.run_count):
            res = MonteCarloSample(config)
            trajectories.append(res[0])
            distances.append(res[1])
            dvs.append(res[2])

    distances = u.Quantity(distances)
    impact_points = [traj[-1][0] for traj in trajectories]
    dvs = u.Quantity(dvs)

    return distances, impact_points, dvs


def graph_2d(config: SimConfig, impact_points: list, filename: str):
    impact_y = np.array([p[1].to_value(u.km) for p in impact_points])
    impact_z = np.array([p[2].to_value(u.km) for p in impact_points])

    target_yz = config.target[1:].to_value(u.km)
    distances = np.sqrt((impact_y - target_yz[0]) ** 2 + (impact_z - target_yz[1]) ** 2)

    r_mean = np.mean(distances)
    r_3sigma = np.percentile(distances, 99.7)

    plt.figure(figsize=(6, 6), dpi=150)
    plt.gca().add_patch(
        plt.Circle(
            (0, 0), config.comet_radius.to_value(u.km), color="gray", label="Comet"
        )
    )
    if "flyby" in filename:
        plt.annotate("Comet", xy=(0, 0), xytext=(0.3, 0.35), textcoords="figure fraction", arrowprops=dict(facecolor="k", shrink=0.1, width=2))
    plt.scatter(impact_y, impact_z, s=0.5)
    plt.gca().add_patch(
        plt.Circle(target_yz, r_mean, color="red", fill=False, ls="--", label="Mean")
    )
    plt.gca().add_patch(
        plt.Circle(
            target_yz, r_3sigma, color="red", fill=False, ls=":", label="3 Sigma"
        )
    )
    plt.gca().set_aspect("equal")
    plt.gca().set_adjustable("datalim")
    plt.legend()
    plt.xlabel("Y (km)")
    plt.ylabel("Z (km)")
    plt.title(
        f"Intercept Points, N={config.run_count:,}, D={2*config.comet_radius.to(u.m):.0f}"
    )
    plt.tight_layout()
    plt.savefig(filename)


def graph_histogram(config: SimConfig, distances: list, filename: str, title: str):
    min_dist = np.min(distances)

    r_mean = np.mean(distances)
    r_3sigma_upper = np.percentile(distances, 99.7)
    r_3sigma_lower = np.percentile(distances, 0.3)

    if min_dist > 10 * u.km:
        distances = distances.to_value(u.km)
        unit = u.km
    else:
        distances = distances.to_value(u.m)
        unit = u.m

    plt.figure(figsize=(6, 4), dpi=150)
    plt.hist(distances, bins=100)

    if min_dist < config.comet_radius:
        plt.axvline(
            x=config.comet_radius.to_value(unit), c="red", ls="-", label="Comet Radius"
        )
        plt.axvline(
            x=r_3sigma_upper.to_value(unit),
            c="red",
            ls=":",
            label="3 Sigma Impact Distance",
        )
        plt.axvline(
            x=r_mean.to_value(unit), c="red", ls="--", label="Mean Impact Distance"
        )
    else:
        plt.axvline(
            x=r_3sigma_upper.to_value(unit),
            c="red",
            ls=":",
            label="3 Sigma Flyby Distance",
        )
        plt.axvline(x=r_3sigma_lower.to_value(unit), c="red", ls=":")
        plt.axvline(
            x=r_mean.to_value(unit), c="red", ls="--", label="Mean Flyby Distance"
        )

    plt.legend(loc="upper right")

    plt.xlabel(f"Distance from center ({unit})")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)


# endregion

# region Main


def test_no_error():
    start_time = 24 * u.hour
    start_vel = 50 * u.km / u.s
    start_dist = start_vel * start_time

    config = SimConfig(
        start_dist=start_dist,
        start_vel=start_vel,
        tcm_times=[] * u.hour,
        comet_radius=5 * u.km,
        cov_pos_abs=([0, 0, 0] * u.km) ** 2,
        cov_vel_abs=([0, 0, 0] * u.m / u.s) ** 2,
        cov_dv=([0, 0, 0] * u.m / u.s) ** 2,
        thrust_scale=0,
        run_count=1000,
    )

    distances, impact_points, dvs = run_batch(config)
    graph_2d(config, impact_points, "mc_output/test_no_error.png")

    # Should hit center every time with 0 dv
    assert np.max(distances) < 0.1 * u.km
    assert np.sum(dvs) < 0.01 * u.m / u.s


def test_initial_error():
    start_time = 24 * u.hour
    start_vel = 50 * u.km / u.s
    start_dist = start_vel * start_time

    config = SimConfig(
        start_dist=start_dist,
        start_vel=start_vel,
        tcm_times=[] * u.hour,
        comet_radius=5 * u.km,
        cov_pos_abs=([10, 10, 10] * u.km) ** 2,
        cov_vel_abs=([10, 10, 10] * u.m / u.s) ** 2,
        cov_dv=([0, 0, 0] * u.m / u.s) ** 2,
        thrust_scale=0,
        run_count=1000,
    )

    distances, impact_points, dvs = run_batch(config)
    graph_2d(config, impact_points, "mc_output/test_initial_error.png")

    hit_rate = np.sum(distances < config.comet_radius) / len(distances)

    # Should almost never hit
    assert hit_rate < 0.01
    assert np.sum(dvs) < 0.01 * u.m / u.s


def test_single_maneuver():
    start_time = 24 * u.hour
    start_vel = 50 * u.km / u.s
    start_dist = start_vel * start_time

    config = SimConfig(
        start_dist=start_dist,
        start_vel=start_vel,
        tcm_times=[12] * u.hour,
        comet_radius=5 * u.km,
        cov_pos_abs=([10, 10, 10] * u.km) ** 2,
        cov_vel_abs=([10, 10, 10] * u.m / u.s) ** 2,
        cov_dv=([0, 0, 0] * u.m / u.s) ** 2,
        thrust_scale=0,
        run_count=1000,
    )

    distances, impact_points, dvs = run_batch(config)
    graph_2d(config, impact_points, "mc_output/test_single_maneuver.png")

    # Should hit every time
    assert np.max(distances) < 0.1 * u.km
    assert np.max(dvs) < 100 * u.m / u.s


def test_thruster_error():
    start_time = 24 * u.hour
    start_vel = 50 * u.km / u.s
    start_dist = start_vel * start_time

    config = SimConfig(
        start_dist=start_dist,
        start_vel=start_vel,
        tcm_times=[12] * u.hour,
        comet_radius=5 * u.km,
        cov_pos_abs=([10, 10, 10] * u.km) ** 2,
        cov_vel_abs=([10, 10, 10] * u.m / u.s) ** 2,
        cov_dv=([1, 1, 1] * u.mm / u.s) ** 2,
        thrust_scale=0.025,
        run_count=1000,
    )

    distances, impact_points, dvs = run_batch(config)
    graph_2d(config, impact_points, "mc_output/test_thruster_error.png")

    hit_rate = np.sum(distances < config.comet_radius) / len(distances)

    assert hit_rate < 0.3, hit_rate


def common_config(final: bool = True) -> SimConfig:
    start_time = 1 * u.day
    start_vel = 65 * u.km / u.s
    start_dist = start_vel * start_time

    runs = 100_000 if final else 10_000

    # dV error: impulse bit / sc mass
    # MR-111G: 0.076 Ns
    # Mass: 100 kg
    # 7.6e-4 m/s
    # 2.5% thrust scale range based on ISP
    # 2 thruster per direction

    config = SimConfig(
        start_dist=start_dist,
        start_vel=start_vel,
        start_separation=(1 * u.m / u.s) * (4 * u.day),
        tcm_times=start_time - [12, 6, 1, 5 / 60] * u.hour,
        comet_radius=0.69 / 2 * u.km,
        cov_pos_abs=([100e3, 100e3, 100e3] * u.m) ** 2,
        cov_vel_abs=([2, 2, 2] * u.m / u.s) ** 2,
        cov_dv=(2 * 7.6e-4 * ([1.0, 1.0, 1.0] * u.m / u.s)) ** 2,
        thrust_scale=0.025,
        pixel_rads=(6.5 * u.um) / (2628.326 * u.mm),
        run_count=runs,
    )

    return config


def impact(final):
    config = common_config(final=final)

    start = time.time()
    distances, impact_points, dvs = run_batch(config)
    end = time.time()

    print(f"Sim took {end-start:0.3f}s")

    # Plot Results
    # 2D impact point cloud plot
    # 3D trajectory impact plot, with comet shape plotted, and impact point highlighted
    # histogram of successful impacts, including 3-sigma bounds

    # Graphs
    graph_2d(config, impact_points, "mc_output/impact.png")
    graph_histogram(
        config, distances, "mc_output/impact_hist.png", "Impact Distance Histogram"
    )

    # Report Statistics
    print(f"Average delta V: {np.mean(dvs):0.2f}")
    print(f"3 sigma delta V: {np.percentile(dvs, 99.7):0.2f}")

    print(f"Average impact diameter: {np.mean(distances):0.1f}")
    print(f"3 sigma impact diameter: {2 * np.percentile(distances, 99.7):0.1f}")
    print(
        f"Impact percent: {np.sum(distances < config.comet_radius)/len(distances):.2%}"
    )
    print()


def flyby(final):
    FLYBY_DIST = 500  # km

    config = common_config(final=final)
    config.target = [0, FLYBY_DIST, 0] * u.km
    config.tcm_times = 1 * u.day - [12, 6] * u.hour

    start = time.time()
    distances, impact_points, dvs = run_batch(config)
    end = time.time()

    print(f"Sim took {end-start:0.3f}s")

    graph_2d(config, impact_points, "mc_output/flyby.png")
    graph_histogram(
        config, distances, "mc_output/flyby_hist.png", "Flyby Distance Histogram"
    )

    # Statistics
    print(f"Average delta V: {np.mean(dvs):0.2f}")
    print(f"3 sigma delta V: {np.percentile(dvs, 99.7):0.2f}")

    print(f"Mean flyby distance: {np.mean(distances).to(u.km):0.1f}")
    print(
        f"3 sigma flyby distance: {np.percentile(distances, 0.3).to(u.km):0.1f}, {np.percentile(distances, 99.7).to(u.km):0.1f}"
    )


# endregion

if __name__ == "__main__":
    # test_no_error()
    # test_initial_error()
    # test_single_maneuver()
    # test_thruster_error()

    final = True

    impact(final)
    flyby(final)
