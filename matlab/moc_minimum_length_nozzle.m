% MOC_MINIMUM_LENGTH_NOZZLE
% 2D Method of Characteristics (Prandtl-Meyer Function & Mach Angle Net)
% Compatible with MATLAB (R2016b+) and GNU Octave (5.0+).

clear; clc;

fprintf('=== MATLAB/Octave 2D Method of Characteristics (MOC) Solver ===\n');

gamma = 1.21;
M_design = 3.50;
rt = 0.12;

gp1_gm1 = (gamma + 1.0) / (gamma - 1.0);
nu_exit_rad = sqrt(gp1_gm1) * atan(sqrt((M_design^2 - 1.0) / gp1_gm1)) - atan(sqrt(M_design^2 - 1.0));
nu_exit_deg = rad2deg(nu_exit_rad);
theta_max_deg = 0.5 * nu_exit_deg;

fprintf('Design Exit Mach: %.2f | Prandtl-Meyer Angle nu(M): %.2f deg | MLN Max Turn Theta_max: %.2f deg\n', ...
    M_design, nu_exit_deg, theta_max_deg);

n_char = 8;
d_theta = theta_max_deg / n_char;

fprintf('%-6s | %-8s | %-10s | %-10s | %-10s\n', 'Ray #', 'Theta', 'Mach', 'mu [deg]', 'x_CL/Rt');
fprintf('%s\n', repmat('-', 1, 56));

x_cl_prev = 0.0;
for i = 1:n_char
    theta_i = i * d_theta;
    nu_target_rad = deg2rad(2.0 * theta_i);

    % Invert Prandtl-Meyer nu(M) = nu_target via bisection
    lo = 1.0001; hi = 8.0;
    for b = 1:50
        mid = 0.5 * (lo + hi);
        nu_mid = sqrt(gp1_gm1) * atan(sqrt((mid^2 - 1.0) / gp1_gm1)) - atan(sqrt(mid^2 - 1.0));
        if nu_mid < nu_target_rad
            lo = mid;
        else
            hi = mid;
        end
    end
    M_i = 0.5 * (lo + hi);
    mu_deg = rad2deg(asin(1.0 / M_i));
    slope_cm = tan(deg2rad(theta_i - mu_deg));
    x_cl = -rt / min(-0.05, slope_cm);

    fprintf('%-6d | %7.2f  | %8.4f | %8.2f  | %8.3f\n', i, theta_i, M_i, mu_deg, x_cl / rt);
    assert(x_cl > x_cl_prev, 'Characteristic centerline intercepts must march downstream monotonically');
    x_cl_prev = x_cl;
end

assert(abs(nu_exit_deg - 71.0) < 15.0, 'Prandtl-Meyer angle at M=3.5, gamma=1.21 must lie in supersonic range');
fprintf('ALL MATLAB/OCTAVE 2D MOC ASSERTIONS PASSED.\n');
