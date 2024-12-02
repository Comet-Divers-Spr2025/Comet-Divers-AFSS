% Brendan Bullock
% ENAE404 HW6


% takes column vecs

% Solves Lambert's Problem for IOD
function [V_initial, V_final, rp, e] = lambert_solver(R_initial, R_final, TOF, type, mu)
    % Inputs:
    % R_initial: Initial position vector of the orbit (km), 3x1 column vector
    % R_final: Final position vector of the orbit (km), 3x1 column vector
    % TOF: Time of flight for the orbit (seconds)
    % type: Orbit trajectory type (short way = 1 or long way = -1)
    % mu: Gravitational parameter for larger body (km^3/s^2)

    % Outputs:
    % V_initial: Initial velocity of the orbit (km/s), 3x1 column vector
    % V_final: Final velocity of the orbit (km/s), 3x1 column vector
    % rp: Radius of periapsis of the orbit (km), scalar
    % e: Eccentricity of the orbit, scalar

    % Calculate the change in true anomaly
    delta_nu = acos(dot(R_initial, R_final) / (norm(R_initial) * norm(R_final)));
    A = type * sqrt(norm(R_initial) * norm(R_final) * (1 + cos(delta_nu)));

    tol = 1e-6;

    if abs(A) < tol && abs(delta_nu) < tol
        disp('Error')
    end

    psi = 0;
    C2 = 1/2;
    C3 = 1/6;
    psi_up = 4 * pi^2;
    psi_low = -4 * pi^2;
    delta_t = 0;

    while abs(TOF - delta_t) > tol
        y = norm(R_initial) + norm(R_final) + A * (psi * C3 - 1) / sqrt(C2);

        if A > 0 && y < 0
            psi_low = psi_low + 0.1 * (TOF - delta_t) / TOF;
        end

        x = sqrt(y / C2);
        delta_t = (x^3 * C3 + A * sqrt(y)) / sqrt(mu);

        if delta_t <= TOF
            psi_low = psi;
        else
            psi_up = psi;
        end

        psi = (psi_up + psi_low) / 2;

        if psi > tol
            C2 = (1 - cos(sqrt(psi))) / psi;
            C3 = (sqrt(psi) - sin(sqrt(psi))) / sqrt(psi^3);
        elseif psi < -tol
            C2 = (cosh(sqrt(-psi)) - 1) / -psi;
            C3 = (sinh(sqrt(-psi)) - sqrt(-psi)) / sqrt((-psi)^3);
        else
            C2 = 1/2;
            C3 = 1/6;
        end
    end

    f = 1 - y / norm(R_initial);
    g = A * sqrt(y / mu);
    g_dot = 1 - y / norm(R_final);

    V_initial = (R_final - f * R_initial) / g;
    V_final = (g_dot * R_final - R_initial) / g;

    h = cross(R_initial, V_initial);
    e_vector = cross(V_initial, h) / mu - R_initial / norm(R_initial);
    e = norm(e_vector);
    epsilon = norm(V_initial)^2 / 2 - mu / norm(R_initial);
    a = -mu / (2 * epsilon);
    rp = a * (1 - e);
end







   



              

          

    

    


