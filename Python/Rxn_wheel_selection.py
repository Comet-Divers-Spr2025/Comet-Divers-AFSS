# Script to plot slew rate requirements for reaction wheels

import numpy as np
import matplotlib.pyplot as plt

# Functions
def angular_offset(t, v, d):
    return np.arctan(v*t/d) # rad

def angular_rate(t, v, d):
    return (v*d) / (v**2 * t**2 + d**2) # rad/s

def angular_acceleration(t, v, d):
    return (-2 * t * v**3) / (d**3 * ((t**2 * v**2 / d**2) + 1)**2) # rad/s^2

def slew_torque(t, v, d, MoI):
    return MoI * angular_acceleration(t, v, d) # N*m

def slew_angular_momentum(t, v, d, MoI):
    return MoI * angular_rate(t, v, d) # N*m


# Mass
m_cricket = 2000 # kg
m_hornet = 50
m_honeybee = 62
m_butterfly = 60

# Flyby velocity
v = 50 # km/s

# time
t = np.linspace(-60, 60, 100) # time in seconds

# MoI
I_cricket = (1/6)*m_cricket*1**2 # kg*m^2, solid cube assumption
I_hornet = (1/6)*m_hornet*1**2 # kg*m^2, solid cube assumption
I_honeybee = (1/6)*m_honeybee*1**2 # kg*m^2, solid cube assumption
I_butterfly = (1/6)*m_butterfly*1**2 # kg*m^2, solid cube assumption

# Flyby distances (not settled, maybe worst case?)
d_cricket = 1000 # km
d_hornet = 1000 # km, unnecessary since impacting    
d_honeybee = 100 # km
d_butterfly = 500 # km

# Plot angular offset (in degrees)
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(t, np.degrees(angular_offset(t, v, d_cricket)), label="Cricket")
plt.title("Angular Offset - Cricket")
plt.xlabel("Time (s)")
plt.ylabel("Angular Offset (degrees)")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, np.degrees(angular_offset(t, v, d_hornet)), label="Hornet")
plt.title("Angular Offset - Hornet")
plt.xlabel("Time (s)")
plt.ylabel("Angular Offset (degrees)")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, np.degrees(angular_offset(t, v, d_honeybee)), label="Honeybee")
plt.title("Angular Offset - Honeybee")
plt.xlabel("Time (s)")
plt.ylabel("Angular Offset (degrees)")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t, np.degrees(angular_offset(t, v, d_butterfly)), label="Butterfly")
plt.title("Angular Offset - Butterfly")
plt.xlabel("Time (s)")
plt.ylabel("Angular Offset (degrees)")
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot angular rate (in degrees per second)
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(t, np.degrees(angular_rate(t, v, d_cricket)), label="Cricket")
plt.title("Angular Rate - Cricket")
plt.xlabel("Time (s)")
plt.ylabel("Angular Rate (deg/s)")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, np.degrees(angular_rate(t, v, d_hornet)), label="Hornet")
plt.title("Angular Rate - Hornet")
plt.xlabel("Time (s)")
plt.ylabel("Angular Rate (deg/s)")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, np.degrees(angular_rate(t, v, d_honeybee)), label="Honeybee")
plt.title("Angular Rate - Honeybee")
plt.xlabel("Time (s)")
plt.ylabel("Angular Rate (deg/s)")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t, np.degrees(angular_rate(t, v, d_butterfly)), label="Butterfly")
plt.title("Angular Rate - Butterfly")
plt.xlabel("Time (s)")
plt.ylabel("Angular Rate (deg/s)")
plt.grid(True)

plt.tight_layout()
# plt.show()

# Plot angular acceleration (in degrees per second squared)
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(t, np.degrees(angular_acceleration(t, v, d_cricket)), label="Cricket")
plt.title("Angular Acceleration - Cricket")
plt.xlabel("Time (s)")
plt.ylabel("Angular Acceleration (deg/s²)")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, np.degrees(angular_acceleration(t, v, d_hornet)), label="Hornet")
plt.title("Angular Acceleration - Hornet")
plt.xlabel("Time (s)")
plt.ylabel("Angular Acceleration (deg/s²)")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, np.degrees(angular_acceleration(t, v, d_honeybee)), label="Honeybee")
plt.title("Angular Acceleration - Honeybee")
plt.xlabel("Time (s)")
plt.ylabel("Angular Acceleration (deg/s²)")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t, np.degrees(angular_acceleration(t, v, d_butterfly)), label="Butterfly")
plt.title("Angular Acceleration - Butterfly")
plt.xlabel("Time (s)")
plt.ylabel("Angular Acceleration (deg/s²)")
plt.grid(True)

plt.tight_layout()
plt.show()

# Plot slew torque
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(t, slew_torque(t, v, d_cricket, I_cricket), label="Cricket")
plt.title("Slew Torque - Cricket")
plt.xlabel("Time (s)")
plt.ylabel("Slew Torque (N*m)")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, slew_torque(t, v, d_hornet, I_hornet), label="Hornet")
plt.title("Slew Torque - Hornet")
plt.xlabel("Time (s)")
plt.ylabel("Slew Torque (N*m)")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, slew_torque(t, v, d_honeybee, I_honeybee), label="Honeybee")
plt.title("Slew Torque - Honeybee")
plt.xlabel("Time (s)")
plt.ylabel("Slew Torque (N*m)")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t, slew_torque(t, v, d_butterfly, I_butterfly), label="Butterfly")
plt.title("Slew Torque - Butterfly")
plt.xlabel("Time (s)")
plt.ylabel("Slew Torque (N*m)")
plt.grid(True)

plt.tight_layout
# plt.show()

# Plot slew momentum
plt.figure(figsize=(10, 8))
plt.subplot(2, 2, 1)
plt.plot(t, slew_angular_momentum(t, v, d_cricket, I_cricket), label="Cricket")
plt.title("Slew Momentum - Cricket")
plt.xlabel("Time (s)")
plt.ylabel("Slew Momentum (N*m-s)")
plt.grid(True)

plt.subplot(2, 2, 2)
plt.plot(t, slew_angular_momentum(t, v, d_hornet, I_hornet), label="Hornet")
plt.title("Slew Momentum - Hornet")
plt.xlabel("Time (s)")
plt.ylabel("Slew Momentum (N*m-s)")
plt.grid(True)

plt.subplot(2, 2, 3)
plt.plot(t, slew_angular_momentum(t, v, d_honeybee, I_honeybee), label="Honeybee")
plt.title("Slew Momentum - Honeybee")
plt.xlabel("Time (s)")
plt.ylabel("Slew Momentum (N*m-s)")
plt.grid(True)

plt.subplot(2, 2, 4)
plt.plot(t, slew_angular_momentum(t, v, d_butterfly, I_butterfly), label="Butterfly")
plt.title("Slew Momentum - Butterfly")
plt.xlabel("Time (s)")
plt.ylabel("Slew Momentum (N*m-s)")
plt.grid(True)

plt.tight_layout
plt.show()


