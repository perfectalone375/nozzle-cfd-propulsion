% PROPELLANT_TRADE_STUDY
% Multi-Propellant Thermochemical & Bell Nozzle Performance Comparison
% Compatible with MATLAB (R2016b+) and GNU Octave (5.0+).

clear; clc;

fprintf('=== MATLAB/Octave Multi-Propellant Trade Study (Pc=10 MPa, eps=16) ===\n');

g0 = 9.80665;
R_univ = 8314.4626;
Pc_Pa = 10.0e6;
Pa_Pa = 101325.0;
eps_ratio = 16.0;
At_m2 = pi * (0.12^2);
eta_cstar = 0.985;
eta_cf = 0.985;

names   = {'HYDROLOX', 'METHALOX', 'KEROLOX', 'SOLID_APCP'};
Tc_list = [3500.0, 3560.0, 3670.0, 3380.0];
gam_list = [1.20, 1.21, 1.19, 1.17];
Mw_list = [13.4, 20.8, 23.2, 26.8];

fprintf('%-12s | %7s | %5s | %8s | %7s | %8s | %8s | %8s\n', ...
    'Propellant', 'Tc [K]', 'gamma', 'c* [m/s]', 'M_exit', 'Isp_SL', 'Isp_Vac', 'F [kN]');
fprintf('%s\n', repmat('-', 1, 78));

isp_results = zeros(1, length(names));

for k = 1:length(names)
    Tc = Tc_list(k);
    gam = gam_list(k);
    Mw = Mw_list(k);
    R_sp = R_univ / Mw;
    gm1 = gam - 1.0;
    gp1 = gam + 1.0;

    BigGamma = sqrt(gam) * (2.0 / gp1)^((gp1) / (2.0 * gm1));
    c_star = eta_cstar * sqrt(R_sp * Tc) / BigGamma;

    % Solve supersonic exit Mach via damped Newton-Raphson on Area-Mach relation
    M = 3.5;
    for iter = 1:40
        term = (2.0 / gp1) * (1.0 + 0.5 * gm1 * M^2);
        expo = gp1 / (2.0 * gm1);
        f_val = (1.0 / M) * (term^expo) - eps_ratio;
        df_dM = (term^expo) * (-1.0 / M^2) + (1.0 / M) * expo * (term^(expo - 1.0)) * (2.0 / gp1) * gm1 * M;
        step = f_val / df_dM;
        M = max(1.05, M - step);
        if abs(step) < 1e-10
            break;
        end
    end

    pe_pc = (1.0 + 0.5 * gm1 * M^2)^(-gam / gm1);
    Cf_mom = eta_cf * sqrt((2 * gam^2 / gm1) * ((2 / gp1)^(gp1 / gm1)) * (1 - pe_pc^(gm1 / gam)));
    Cf_sl = Cf_mom + eps_ratio * (pe_pc - Pa_Pa / Pc_Pa);
    Cf_vac = Cf_mom + eps_ratio * pe_pc;

    mdot = Pc_Pa * At_m2 / c_star;
    F_sl = Pc_Pa * At_m2 * Cf_sl;
    F_vac = Pc_Pa * At_m2 * Cf_vac;
    Isp_sl = (F_sl / mdot) / g0;
    Isp_vac = (F_vac / mdot) / g0;
    isp_results(k) = Isp_vac;

    fprintf('%-12s | %7.1f | %5.2f | %8.1f | %7.3f | %8.2f | %8.2f | %8.2f\n', ...
        names{k}, Tc, gam, c_star, M, Isp_sl, Isp_vac, F_sl / 1e3);
end

% Physical ranking invariant: Hydrolox > Methalox > Kerolox > Solid APCP
assert(isp_results(1) > isp_results(2) && isp_results(2) > isp_results(3) && isp_results(3) > isp_results(4), ...
    'Specific impulse ranking must follow Hydrolox > Methalox > Kerolox > Solid APCP');
fprintf('ALL MATLAB/OCTAVE PROPELLANT TRADE STUDY ASSERTIONS PASSED.\n');
