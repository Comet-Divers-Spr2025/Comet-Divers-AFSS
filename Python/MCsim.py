### Impactor Monte Carlo Simulation ###

import astropy.units as u
import dataclasses
import numpy as np
import matplotlib.pyplot as plt


@dataclasses.dataclass
class SimConfig:
    ## Mission parameters
    start_dist: u.Quantity["length", 1]
    start_vel: u.Quantity["velocity", 1]
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
    focal_length: u.Quantity["length"] = 1000 * u.mm
    pixel_pitch: u.Quantity["length"] = 10 * u.um
    pixel_accuracy: float = 0.05

    ## Simulation parameters
    run_count: int = 1


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
    pixel_resolution = np.linalg.norm(r) * config.pixel_pitch / config.focal_length
    opnav_error = (config.pixel_accuracy * pixel_resolution).si

    # print(f"Radius in pixels: {(config.comet_radius / pixel_resolution).si}")

    error_vec = [
        0,
        np.random.normal(0, opnav_error.to_value(opnav_error.unit)),
        np.random.normal(0, opnav_error.to_value(opnav_error.unit))
    ] * opnav_error.unit

    return error_vec


def MonteCarloSample(config: SimConfig):
    comet = [0, 0, 0] * u.m

    trajectory = []
    dv_total = 0 * u.m / u.s
    t = 0 * u.s

    r = u.Quantity([config.start_dist, 0 * u.km, 0 * u.km]).si
    v = u.Quantity([-config.start_vel, 0 * u.km / u.s, 0 * u.km / u.s]).si

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

        # Optical navigation
        r += calc_opnav_error(config, r)

        # Run targeting algorithm and execute maneuver
        dv = B_plane_targeting(r, v, comet)
        dv *= np.random.normal(1, config.thrust_scale)
        dv += normal_sample(cov_dv)
        dv_total += np.linalg.norm(dv)
        v += dv

        trajectory.append([r.copy(), v.copy()])

    # Find closest approach distance

    dt = time_to_impact(r, v, comet)
    t_closest = t + dt
    r, v = linear_propagation(r, v, dt)
    d_closest = np.linalg.norm(r - comet)

    trajectory.append([r.copy(), v.copy()])

    # print(f"Closest approach distance: {d_closest.to(u.km)}, {t_closest.to(u.hour)}")

    return trajectory, d_closest, dv_total


def run_batch(config: SimConfig):
    distances = []
    trajectories = []
    dvs = []

    # Execute Monte Carlo Simulation
    for _ in range(config.run_count):
        traj, dist, dv = MonteCarloSample(config)
        distances.append(dist)
        trajectories.append(traj)
        dvs.append(dv)

    distances = u.Quantity(distances)
    impact_points = [traj[-1][0] for traj in trajectories]
    dvs = u.Quantity(dvs)

    return distances, impact_points, dvs


def graph_2d(config: SimConfig, impact_points: list, filename: str):
    impact_y = [p[1].to_value(u.km) for p in impact_points]
    impact_z = [p[2].to_value(u.km) for p in impact_points]

    plt.figure(figsize=(8, 8), dpi=200)
    plt.gca().add_patch(
        plt.Circle((0, 0), config.comet_radius.to_value(u.km), color="gray")
    )
    plt.scatter(impact_y, impact_z, s=1)
    plt.gca().set_aspect("equal")
    plt.gca().set_adjustable("datalim")
    plt.xlabel("Y (km)")
    plt.ylabel("Z (km)")
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


def main():
    start_time = 1 * u.day
    start_vel = 50 * u.km / u.s
    start_dist = start_vel * start_time

    # dV error: impulse bit / sc mass
    # MR-111G: 0.076 Ns
    # Mass: 100 kg
    # 7.6e-4 m/s
    # 2.5% thrust scale range based on ISP

    config = SimConfig(
        start_dist=start_dist,
        start_vel=start_vel,
        tcm_times=start_time - [12, 6, 1] * u.hour,
        comet_radius=5 * u.km,
        cov_pos_abs=([100e3, 100e3, 100e3] * u.m) ** 2,
        cov_vel_abs=([2, 2, 2] * u.m / u.s) ** 2,
        cov_dv=(7.6e-4 * ([1.0, 1.0, 1.0] * u.m / u.s)) ** 2,
        thrust_scale=0.025,
        run_count=1000,
    )

    distances, impact_points, dvs = run_batch(config)

    print(np.mean(dvs), np.max(dvs))

    # Plot Results
    # 2D impact point cloud plot
    # 3D trajectory impact plot, with comet shape plotted, and impact point highlighted
    # histogram of successful impacts, including 3-sigma bounds

    # 2D plot
    graph_2d(config, impact_points, "mc_output/sim_2d.png")

    # Report Statistics
    # mean and standard deviation of impact point
    # information regarding % of success
    # others

    print(f"Mean closest distance: {np.mean(distances):0.1f}")
    print(f"Min distance: {np.min(distances):0.1f}")
    print(
        f"Impact percent: {np.sum(distances < config.comet_radius)/len(distances):.2%}"
    )


# endregion

if __name__ == "__main__":
    # test_no_error()
    # test_initial_error()
    # test_single_maneuver()
    # test_thruster_error()

    main()
