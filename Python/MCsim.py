### Impactor Monte Carlo Simulation ###

import numpy as np
import matplotlib.pyplot as plt
import scipy

#region Function Definitions

def linear_propagation(X, dt):
    r = np.linalg.norm(X[:3])
    v = np.linalg.norm(X[3:])
    new_r = r + v * dt 
    return np.array([new_r, v])

def B_plane_targeting(r, v, B_target):
    required_delta_V = 0 # insert B-plane targeting algorithm here
    return required_delta_V

def MonteCarloSample(X0, num_TCMs, t_impact, covariances):
    trajectory = np.zeros((num_TCMs, 6))
    tsteps = np.linspace(0, t_impact, num_TCMs)
    dt = t_impact / num_TCMs
    trajectory[0, :] = np.random.multivariate_normal(mean=X0, cov=covariances[0])

    for i in range(tsteps-1):
        trajectory[i+1, :] = linear_propagation(trajectory[i, :], dt)
        trajectory[i+1, :] = linear_propagation(trajectory[i, :], dt) + np.random.multivariate_normal(mean=0, cov=covariances[0]) # state estimation error
        #insert B-plane targeting, including B_plane error covariance (np.random.multivariate_normal(mean=0, cov=covariances[1]))
        # return delV necessary
        # v = current velocity + required del_V from B_plane
        # trajectory[i+1, 3:] = v + np.random.multivariate_normal(mean=0, cov=covariances[2]) # maneuver execution error

    return trajectory

#endregion

#region MAIN

# Define Initial Conditions
X0 = [0, 0, 0, 0, 0, 0]     # to change
num_TCMs = 0    # number of TCMs to simulate
t_impact = 1000     # time of impact
comet_state = [0, 0, 0, 0, 0, 0]   # assuming at zero

# Define Uncertainties
placeholder = 0
state_est_covariance = np.diag([placeholder, placeholder, placeholder, placeholder, placeholder, placeholder])
B_plane_covariance = np.diag([placeholder, placeholder, placeholder])
maneuver_exec_covariance = np.diag([placeholder, placeholder, placeholder])
covariances = [state_est_covariance, B_plane_covariance, maneuver_exec_covariance]

# Execute Monte Carlo Simulation
num_samples = 1000
trajectories = np.zeros((num_samples,num_TCMs, 6))
for i in range(num_samples):
    trajectories[i, :, :] = MonteCarloSample(X0, num_TCMs, covariances)

# Plot Results
#2D impact point cloud plot
#3D trajectory impact plot, with comet shape plotted, and impact point highlighted
#histogram of successful impacts, including 3-sigma bounds

# Report Statistics
#mean and standard deviation of impact point
#information regarding % of success
#others

#endregion