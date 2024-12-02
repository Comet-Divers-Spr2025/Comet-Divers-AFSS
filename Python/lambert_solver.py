## ENAE484 2024-2025 AFSS Lambert Orbit Problem Solver

import numpy as np

def lambert_solver(R_initial, R_final, TOF, orbit_type, mu):
    """
    Solves Lambert's Problem for IOD

    Parameters:
    R_initial (np.array): Initial position vector of the orbit (km), 3x1 column vector
    R_final (np.array): Final position vector of the orbit (km), 3x1 column vector
    TOF (float): Time of flight for the orbit (seconds)
    orbit_type (int): Orbit trajectory type (short way = 1 or long way = -1)
    mu (float): Gravitational parameter for larger body (km^3/s^2)

    Returns:
    V_initial (np.array): Initial velocity of the orbit (km/s), 3x1 column vector
    V_final (np.array): Final velocity of the orbit (km/s), 3x1 column vector
    rp (float): Radius of periapsis of the orbit (km)
    e (float): Eccentricity of the orbit
    """
    # Calculate the change in true anomaly
    delta_nu = np.arccos(np.dot(R_initial, R_final) / (np.linalg.norm(R_initial) * np.linalg.norm(R_final)))
    A = orbit_type * np.sqrt(np.linalg.norm(R_initial) * np.linalg.norm(R_final) * (1 + np.cos(delta_nu)))

    tol = 1e-6

    if abs(A) < tol and abs(delta_nu) < tol:
        raise ValueError("Error: A and delta_nu are too small.")

    psi = 0
    C2 = 1/2
    C3 = 1/6
    psi_up = 4 * np.pi**2
    psi_low = -4 * np.pi**2
    delta_t = 0

    while abs(TOF - delta_t) > tol:
        y = np.linalg.norm(R_initial) + np.linalg.norm(R_final) + A * (psi * C3 - 1) / np.sqrt(C2)

        if A > 0 and y < 0:
            psi_low = psi_low + 0.1 * (TOF - delta_t) / TOF

        x = np.sqrt(y / C2)
        delta_t = (x**3 * C3 + A * np.sqrt(y)) / np.sqrt(mu)

        if delta_t <= TOF:
            psi_low = psi
        else:
            psi_up = psi

        psi = (psi_up + psi_low) / 2

        if psi > tol:
            C2 = (1 - np.cos(np.sqrt(psi))) / psi
            C3 = (np.sqrt(psi) - np.sin(np.sqrt(psi))) / np.sqrt(psi**3)
        elif psi < -tol:
            C2 = (np.cosh(np.sqrt(-psi)) - 1) / -psi
            C3 = (np.sinh(np.sqrt(-psi)) - np.sqrt(-psi)) / np.sqrt((-psi)**3)
        else:
            C2 = 1/2
            C3 = 1/6

    f = 1 - y / np.linalg.norm(R_initial)
    g = A * np.sqrt(y / mu)
    g_dot = 1 - y / np.linalg.norm(R_final)

    V_initial = (R_final - f * R_initial) / g
    V_final = (g_dot * R_final - R_initial) / g

    h = np.cross(R_initial, V_initial)
    e_vector = np.cross(V_initial, h) / mu - R_initial / np.linalg.norm(R_initial)
    e = np.linalg.norm(e_vector)
    epsilon = np.linalg.norm(V_initial)**2 / 2 - mu / np.linalg.norm(R_initial)
    a = -mu / (2 * epsilon)
    rp = a * (1 - e)

    return V_initial, V_final, rp, e

# Example usage
R_initial = np.array([7000, 0, 0])
R_final = np.array([8000, 1000, 0])
TOF = 3600  # seconds
orbit_type = 1  # short way
mu = 398600  # km^3/s^2

V_initial, V_final, rp, e = lambert_solver(R_initial, R_final, TOF, orbit_type, mu)
print("V_initial:", V_initial)
print("V_final:", V_final)
print("rp:", rp)
print("e:", e)
