% SOLVE_NOZZLE_CFD_MACCORMACK
% Conservative Quasi-1D Euler MacCormack CFD Solver for Rocket Nozzles
% Compatible with MATLAB (R2016b+) and GNU Octave (5.0+).
%
% Models:
%   1) Supersonic design/overexpanded expansion through a converging-diverging nozzle
%   2) Capturing a standing normal shock inside the nozzle at elevated back pressure
%
% Author: Gopi Naga Vignesh Thirupati (nozzle-cfd-propulsion)

clear; clc;

fprintf('=== MATLAB/Octave Conservative MacCormack Nozzle CFD Solver ===\n');

% -------------------------------------------------------------------------
% 1. Propellant & Geometry Parameters (LOX/CH4 Methalox Class)
% -------------------------------------------------------------------------
gamma   = 1.21;        % Ratio of specific heats Cp/Cv
Pc_kPa  = 1000.0;      % Chamber total pressure [kPa] (1.0 MPa for clear in-nozzle shock demo)
Pb_kPa  = 380.0;       % Elevated ambient back-pressure forcing a normal shock [kPa]
Tc_K    = 3560.0;      % Chamber stagnation temperature [K]

nx = 101;
x_nd = linspace(0.0, 3.0, nx);          % Nondimensional axial coordinate x/L_ref
dx = x_nd(2) - x_nd(1);

% Smooth parabolic-cosine converging-diverging area ratio A(x)/A*
% Throat at x_nd = 1.0 (minimum A/A* = 1.0), exit A/A* = 8.0
A = zeros(1, nx);
for i = 1:nx
    if x_nd(i) <= 1.0
        A(i) = 1.0 + 2.2 * (1.0 - x_nd(i))^2;
    else
        s = (x_nd(i) - 1.0) / 2.0;
        A(i) = 1.0 + 7.0 * (s^1.45);
    end
end
[~, throat_idx] = min(A);
dA_dx = gradient(A, dx);

% Run two verification cases: Case 1 = Supersonic Isentropic, Case 2 = Normal Shock Inside
case_names = {'Design_Supersonic_Outflow', 'Overexpanded_Normal_Shock_Captured'};
pb_ratios  = [0.008, Pb_kPa / Pc_kPa];
cfl        = 0.45;
max_iter   = 6500;
tol        = 1e-5;
gm1        = gamma - 1.0;

for c = 1:2
    subsonic_out = (c == 2);
    pb_nd = pb_ratios(c);
    cx = 0.08;
    if subsonic_out
        cx = 0.32; % Jameson shock-capturing pressure dissipation coefficient
    end

    % Initial condition along nozzle
    M_init = ones(1, nx);
    for i = 1:nx
        if i <= throat_idx
            M_init(i) = 0.18 + 0.82 * (i / throat_idx);
        elseif ~subsonic_out
            M_init(i) = 1.0 + 2.1 * ((i - throat_idx) / (nx - throat_idx));
        else
            M_init(i) = 1.0 + 0.85 * sin(pi * (i - throat_idx) / (2 * (nx - throat_idx))) ...
                        - 0.65 * ((i - throat_idx) / (nx - throat_idx));
        end
    end

    T_init   = 1.0 ./ (1.0 + 0.5 * gm1 * M_init.^2);
    rho_init = T_init .^ (1.0 / gm1);
    V_init   = M_init .* sqrt(T_init);

    U1 = rho_init .* A;
    U2 = rho_init .* A .* V_init;
    U3 = rho_init .* A .* (T_init ./ gm1 + 0.5 * gamma * V_init.^2);

    converged = false;
    final_diff = 1.0;
    iters = 0;

    for it = 1:max_iter
        rho = max(U1 ./ A, 1e-8);
        V   = U2 ./ max(U1, 1e-8);
        T   = max(gm1 * ((U3 ./ max(U1, 1e-8)) - 0.5 * gamma * V.^2), 1e-4);
        p   = rho .* T;
        a   = sqrt(T);
        dt  = cfl * dx / max(abs(V) + a);

        % Pressure-switch Jameson artificial viscosity
        sensor = zeros(1, nx);
        for i = 2:(nx - 1)
            denom = max(p(i+1) + 2*p(i) + p(i-1), 1e-8);
            sensor(i) = cx * abs(p(i+1) - 2*p(i) + p(i-1)) / denom;
        end

        % Fluxes F(U)
        F1 = U2;
        F2 = (U2.^2) ./ max(U1, 1e-8) + (gm1 / gamma) * (U3 - 0.5 * gamma * (U2.^2) ./ max(U1, 1e-8));
        F3 = gamma * (U2 .* U3) ./ max(U1, 1e-8) - 0.5 * gamma * gm1 * (U2.^3) ./ (max(U1, 1e-8).^2);

        % Predictor step (forward difference)
        U1_p = U1; U2_p = U2; U3_p = U3;
        for i = 2:(nx - 1)
            J2 = (1.0 / gamma) * p(i) * dA_dx(i);
            S1 = sensor(i) * (U1(i+1) - 2*U1(i) + U1(i-1));
            S2 = sensor(i) * (U2(i+1) - 2*U2(i) + U2(i-1));
            S3 = sensor(i) * (U3(i+1) - 2*U3(i) + U3(i-1));

            U1_p(i) = U1(i) - (dt / dx) * (F1(i+1) - F1(i)) + S1;
            U2_p(i) = U2(i) - (dt / dx) * (F2(i+1) - F2(i)) + dt * J2 + S2;
            U3_p(i) = U3(i) - (dt / dx) * (F3(i+1) - F3(i)) + S3;
        end

        % Predictor BCs
        U2_p(1) = max(0.01 * A(1), 2*U2_p(2) - U2_p(3));
        v0 = min(0.85, max(0.01, U2_p(1) / max(U1(1), 1e-6)));
        t0 = max(0.5, 1.0 - 0.5 * gm1 * v0^2);
        rho0 = t0 ^ (1.0 / gm1);
        U1_p(1) = rho0 * A(1);
        U3_p(1) = U1_p(1) * (t0 / gm1 + 0.5 * gamma * v0^2);

        U1_p(nx) = max(1e-6, 2*U1_p(nx-1) - U1_p(nx-2));
        U2_p(nx) = max(1e-6, 2*U2_p(nx-1) - U2_p(nx-2));
        if subsonic_out
            U3_p(nx) = (pb_nd * A(nx)) / gm1 + 0.5 * gamma * (U2_p(nx)^2) / U1_p(nx);
        else
            U3_p(nx) = max(1e-6, 2*U3_p(nx-1) - U3_p(nx-2));
        end

        % Corrector step (backward difference)
        rho_p = max(U1_p ./ A, 1e-8);
        V_p   = U2_p ./ max(U1_p, 1e-8);
        T_p   = max(gm1 * ((U3_p ./ max(U1_p, 1e-8)) - 0.5 * gamma * V_p.^2), 1e-4);
        p_p   = rho_p .* T_p;

        sensor_p = zeros(1, nx);
        for i = 2:(nx - 1)
            denom = max(p_p(i+1) + 2*p_p(i) + p_p(i-1), 1e-8);
            sensor_p(i) = cx * abs(p_p(i+1) - 2*p_p(i) + p_p(i-1)) / denom;
        end

        F1_p = U2_p;
        F2_p = (U2_p.^2) ./ max(U1_p, 1e-8) + (gm1 / gamma) * (U3_p - 0.5 * gamma * (U2_p.^2) ./ max(U1_p, 1e-8));
        F3_p = gamma * (U2_p .* U3_p) ./ max(U1_p, 1e-8) - 0.5 * gamma * gm1 * (U2_p.^3) ./ (max(U1_p, 1e-8).^2);

        U1_new = U1; U2_new = U2; U3_new = U3;
        for i = 2:(nx - 1)
            J2_p = (1.0 / gamma) * p_p(i) * dA_dx(i);
            S1_c = sensor_p(i) * (U1_p(i+1) - 2*U1_p(i) + U1_p(i-1));
            S2_c = sensor_p(i) * (U2_p(i+1) - 2*U2_p(i) + U2_p(i-1));
            S3_c = sensor_p(i) * (U3_p(i+1) - 2*U3_p(i) + U3_p(i-1));

            U1_new(i) = 0.5 * (U1(i) + U1_p(i) - (dt / dx) * (F1_p(i) - F1_p(i-1))) + 0.5 * S1_c;
            U2_new(i) = 0.5 * (U2(i) + U2_p(i) - (dt / dx) * (F2_p(i) - F2_p(i-1)) + dt * J2_p) + 0.5 * S2_c;
            U3_new(i) = 0.5 * (U3(i) + U3_p(i) - (dt / dx) * (F3_p(i) - F3_p(i-1))) + 0.5 * S3_c;
        end

        % Final BCs
        U2_new(1) = max(0.01 * A(1), 2*U2_new(2) - U2_new(3));
        v0 = min(0.85, max(0.01, U2_new(1) / max(U1_new(1), 1e-6)));
        t0 = max(0.5, 1.0 - 0.5 * gm1 * v0^2);
        rho0 = t0 ^ (1.0 / gm1);
        U1_new(1) = rho0 * A(1);
        U3_new(1) = U1_new(1) * (t0 / gm1 + 0.5 * gamma * v0^2);

        U1_new(nx) = max(1e-6, 2*U1_new(nx-1) - U1_new(nx-2));
        U2_new(nx) = max(1e-6, 2*U2_new(nx-1) - U2_new(nx-2));
        if subsonic_out
            U3_new(nx) = (pb_nd * A(nx)) / gm1 + 0.5 * gamma * (U2_new(nx)^2) / U1_new(nx);
        else
            U3_new(nx) = max(1e-6, 2*U3_new(nx-1) - U3_new(nx-2));
        end

        final_diff = max(abs(U2_new - U2) ./ max(abs(U2), 1e-4));
        U1 = U1_new; U2 = U2_new; U3 = U3_new;
        iters = it;
        if it > 200 && final_diff < tol
            converged = true;
            break;
        end
    end

    rho = U1 ./ A;
    V   = U2 ./ U1;
    T   = gm1 * ((U3 ./ U1) - 0.5 * gamma * V.^2);
    p   = rho .* T;
    M   = V ./ sqrt(T);

    fprintf('[Case %d] %-34s | iters=%4d | conv=%d | M_throat=%.3f | M_exit=%.3f | p_exit/Pc=%.4f\n', ...
        c, case_names{c}, iters, converged, M(throat_idx), M(nx), p(nx));

    assert(converged, 'MacCormack solver did not converge within max_iter');
    assert(abs(M(throat_idx) - 1.0) < 0.08, 'Throat must remain near sonic M=1');
    if c == 1
        assert(M(nx) > 2.9, 'Supersonic design nozzle should exceed M > 2.9 at A/A*=8');
    else
        assert(max(M) > 1.8 && M(nx) < 0.7, 'Overexpanded shock case must accelerate > 1.8 then decelerate subsonic < 0.7');
    end
end

fprintf('ALL MATLAB/OCTAVE MACCORMACK ASSERTIONS PASSED.\n');
