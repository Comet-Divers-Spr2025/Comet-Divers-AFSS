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
    tcm_count: int
    tcm_times: list[u.Quantity["time"]]  # Time after start of sim
    comet_radius: float

    ## Covariances

    # Absolute position from ground tracking, used for initial estimate
    cov_pos_abs: u.Quantity
    cov_vel_abs: u.Quantity

    # Relative position from optical navigation, used for terminal guidance
    # TODO: make these proportional to distance from target
    cov_pos_rel: u.Quantity
    cov_vel_rel: u.Quantity

    # Maneuver
    cov_dv: u.Quantity

    ## Simulation parameters
    run_count: int


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
    dv = np.array([0, 0, 0])  # insert B-plane targeting algorithm here

    return dv


def time_to_impact(r, v, comet):
    v_hat = v / np.linalg.norm(v)
    d_to_approach = np.linalg.norm(np.dot(comet - r, v_hat))
    dt = d_to_approach / np.linalg.norm(v)

    return dt


def MonteCarloSample(config: SimConfig):
    comet = [0, 0, 0] * u.m

    trajectory = []
    t = 0 * u.s

    r = u.Quantity([config.start_dist, 0 * u.km, 0 * u.km]).si
    v = u.Quantity([-config.start_vel, 0 * u.km / u.s, 0 * u.km / u.s]).si

    cov_r_abs = np.diag(config.cov_pos_abs)
    cov_v_abs = np.diag(config.cov_vel_abs)
    cov_r_rel = np.diag(config.cov_pos_rel)
    cov_v_rel = np.diag(config.cov_vel_rel)
    cov_dv = np.diag(config.cov_dv)

    # Initial position uncertainty
    r += normal_sample(cov_r_abs)
    v += normal_sample(cov_v_abs)

    trajectory.append([r.copy(), v.copy()])

    # Execute TCMs
    for i in range(config.tcm_count):
        # Propagate to next TCM
        dt = config.tcm_times[i] - t
        t += dt
        r, v = linear_propagation(r, v, dt)
        r += normal_sample(cov_r_rel)
        v += normal_sample(cov_v_rel)

        # Run targeting algorithm and execute maneuver
        dv = B_plane_targeting(r, v, comet)
        v += dv + normal_sample(cov_dv)

        trajectory.append([r.copy(), v.copy()])

    # Find closest approach distance

    dt = time_to_impact(r, v, comet)
    t_closest = t + dt
    r, v = linear_propagation(r, v, dt)
    d_closest = np.linalg.norm(r - comet)

    trajectory.append([r.copy(), v.copy()])

    # print(f"Closest approach distance: {d_closest.to(u.km)}, {t_closest.to(u.hour)}")

    return trajectory, d_closest


# endregion

# region Main


def main():
    start_time = 24 * u.hour
    start_vel = 50 * u.km / u.s
    start_dist = start_vel * start_time

    config = SimConfig(
        start_dist=start_dist,
        start_vel=start_vel,
        tcm_count=3,
        tcm_times=[12, 18, 23] * u.hour,
        comet_radius=5 * u.km,
        cov_pos_abs=([1e3, 1e3, 1e3] * u.km) ** 2,  # guess
        cov_vel_abs=([10, 10, 10] * u.m / u.s) ** 2,  # guess
        cov_pos_rel=([0, 0, 0] * u.m) ** 2,  # guess
        cov_vel_rel=([0, 0, 0] * u.m / u.s) ** 2,  # guess
        cov_dv=([0.1, 0.1, 0.1] * u.m / u.s) ** 2,  # guess
        run_count=10000,
    )

    distances = []
    trajectories = []

    # Execute Monte Carlo Simulation
    for _ in range(config.run_count):
        traj, dist = MonteCarloSample(config)
        distances.append(dist)
        trajectories.append(traj)

    distances = u.Quantity(distances)

    # Plot Results
    # 2D impact point cloud plot
    # 3D trajectory impact plot, with comet shape plotted, and impact point highlighted
    # histogram of successful impacts, including 3-sigma bounds

    # 2D plot
    impact_points = [traj[-1][0][1:] for traj in trajectories]
    impact_y = [p[0].to_value(u.km) for p in impact_points]
    impact_z = [p[1].to_value(u.km) for p in impact_points]

    plt.figure(figsize=(8, 8), dpi=500)
    plt.scatter(impact_y, impact_z, s=1)
    plt.gca().add_patch(
        plt.Circle((0, 0), config.comet_radius.to_value(u.km), color="r")
    )
    plt.gca().set_aspect("equal")
    plt.gca().set_adjustable("datalim")
    plt.tight_layout()
    plt.savefig("mcsim_2d.png")

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
    main()
