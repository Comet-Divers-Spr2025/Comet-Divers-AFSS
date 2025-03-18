### Impactor Monte Carlo Simulation ###

import numpy as np
import matplotlib.pyplot as plt
import scipy

#region Function Definitions

def linear_propagation(r, v, dt):
    new_r = r + v * dt # insert linear propagation algorithm here
    return new_r 

def B_plane_targeting(r, v, B_target):
    required_delta_V = 0 # insert B-plane targeting algorithm here
    return required_delta_V

def MonteCarloSample(X0, num_TCMs, uncertainties):
    trajectory = np.zeros((num_TCMs, 6)) #insert full monte carlo loop algorithm for one trajectory here
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
state_estimation_uncertainty = np.random.normal(placeholder, placeholder, 6)
B_plane_targeting_uncertainty = np.random.normal(placeholder, placeholder, 6)
maneuver_execution_uncertainty = np.random.normal(placeholder, placeholder, 6)
uncertainties = [state_estimation_uncertainty, B_plane_targeting_uncertainty, maneuver_execution_uncertainty]

# Execute Monte Carlo Simulation
num_samples = 1000
trajectories = np.zeros((num_samples,num_TCMs, 6))
for i in range(num_samples):
    trajectories[i, :, :] = MonteCarloSample(X0, num_TCMs, uncertainties)

# Plot Results
#2D impact point cloud plot
#3D trajectory impact plot, with comet shape plotted, and impact point highlighted
#histogram of successful impacts, including 3-sigma bounds

# Report Statistics
#mean and standard deviation of impact point
#information regarding % of success
#others

#endregion