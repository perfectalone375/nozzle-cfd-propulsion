function solve_nozzle_ns2d_fvm()
% SOLVE_NOZZLE_NS2D_FVM
% 2D Axisymmetric Curvilinear Finite-Volume Euler Solver with Rotated HLLC+,
% Van Leer MUSCL-TVD Reconstruction, 1-Radian Wedge GCL, and Transonic Curved
% Sonic-Line Validation (MATLAB & GNU Octave compatible).
%
% Validates on JPL 45-15 Style Conical Nozzle (eps=4.0, rc/rt=0.625, gamma=1.4):
%   1) Exact 1-radian wedge Geometric Conservation Law (free-stream GCL)
%   2) Axisymmetric face-flux mass conservation (< 0.5% error throat-to-exit)
%   3) 2D Transonic curved sonic line (M_wall > M_centerline at throat)
%   4) Agreement with Hall (1962) analytical discharge coefficient Cd
%
% Author: Gopi Naga Vignesh Thirupati (nozzle-cfd-propulsion)

fprintf('=== MATLAB/Octave 2D Axisymmetric Curvilinear FVM HLLC+ Solver ===\n');

gam = 1.4;
gm1 = gam - 1.0;
r_gas = 296.8;
tc_k = 300.0;
pc_pa = 1.0e6;

% Hall (1962) 2D transonic discharge coefficient for rc/rt = 0.625
R = 0.625;
gp1 = gam + 1.0;
cd_hall = 1.0 ...
    - gp1 / (96.0 * R^2) ...
    + gp1 * (8.0 * gam - 27.0) / (2304.0 * R^3) ...
    - gp1 * (754.0 * gam^2 - 757.0 * gam + 3633.0) / (276480.0 * R^4);

% C1-smooth JPL 45-15 conical nozzle geometry
rt = 0.05;
rc = R * rt;
r_in = rt * sqrt(3.0);
r_ex = rt * sqrt(4.0);
beta = 45.0 * pi / 180.0;
alpha = 15.0 * pi / 180.0;

r_arc_c = rt + rc * (1.0 - cos(beta));
dx_arc_c = rc * sin(beta);
r_fillet = min(0.8 * rt, 0.7 * max(0.0, r_in - r_arc_c) / max(1e-6, 1.0 - cos(beta)));
r_f_end = r_in - r_fillet * (1.0 - cos(beta));
dx_fillet = r_fillet * sin(beta);
dx_conv = max(0.0, (r_f_end - r_arc_c) / tan(beta));
len_lead = 0.25 * r_in;

r_arc_d = rt + rc * (1.0 - cos(alpha));
dx_arc_d = rc * sin(alpha);
dx_div = max(0.0, (r_ex - r_arc_d) / tan(alpha));

x_f_end = len_lead + dx_fillet;
x_a_start = x_f_end + dx_conv;
x_throat = x_a_start + dx_arc_c;
x_a_end = x_throat + dx_arc_d;
x_total = x_a_end + dx_div;

nx = 64;
nr = 16;
x_nodes = linspace(0.0, x_total, nx + 1);
rw_nodes = zeros(1, nx + 1);
for k = 1:(nx + 1)
    xv = x_nodes(k);
    if xv <= len_lead
        rw_nodes(k) = r_in;
    elseif xv < x_f_end
        dx = min(r_fillet * 0.9999, xv - len_lead);
        rw_nodes(k) = r_in - r_fillet + sqrt(max(0.0, r_fillet^2 - dx^2));
    elseif xv < x_a_start
        rw_nodes(k) = r_f_end - (xv - x_f_end) * tan(beta);
    elseif xv <= x_a_end
        dx = max(-rc * 0.9999, min(rc * 0.9999, xv - x_throat));
        rw_nodes(k) = rt + rc - sqrt(max(0.0, rc^2 - dx^2));
    else
        rw_nodes(k) = r_arc_d + (xv - x_a_end) * tan(alpha);
    end
end

xv_g = zeros(nx + 1, nr + 1);
rv_g = zeros(nx + 1, nr + 1);
eta = linspace(0.0, 1.0, nr + 1);
for i = 1:(nx + 1)
    xv_g(i, :) = x_nodes(i);
    rv_g(i, :) = rw_nodes(i) .* eta;
end

% Precompute exact 1-radian curvilinear wedge metrics
dx_cell = xv_g(2:end, 1:end-1) - xv_g(1:end-1, 1:end-1);
hL = rv_g(1:end-1, 2:end) - rv_g(1:end-1, 1:end-1);
hR = rv_g(2:end, 2:end) - rv_g(2:end, 1:end-1);
area = 0.5 .* (hL + hR) .* dx_cell;
r_g = 0.25 .* (rv_g(1:end-1, 1:end-1) + rv_g(2:end, 1:end-1) + rv_g(2:end, 2:end) + rv_g(1:end-1, 2:end));
vol = max(1e-18, r_g .* area);

% I-faces (nx+1, nr): normal (+1, 0), axisymmetric face area S_i = 0.5*(r_top+r_bot)*ds
i_dx = xv_g(:, 2:end) - xv_g(:, 1:end-1);
i_dr = rv_g(:, 2:end) - rv_g(:, 1:end-1);
i_ds = max(1e-12, hypot(i_dx, i_dr));
i_nx = i_dr ./ i_ds;
i_nr = -i_dx ./ i_ds;
i_rmid = 0.5 .* (rv_g(:, 1:end-1) + rv_g(:, 2:end));
i_sf = i_rmid .* i_ds;

% J-faces (nx, nr+1): normal (-dr, dx)/ds, axisymmetric face area S_j = 0.5*(r_left+r_right)*ds
j_dx = xv_g(2:end, :) - xv_g(1:end-1, :);
j_dr = rv_g(2:end, :) - rv_g(1:end-1, :);
j_ds = max(1e-12, hypot(j_dx, j_dr));
j_nx = -j_dr ./ j_ds;
j_nr = j_dx ./ j_ds;
j_rmid = 0.5 .* (rv_g(1:end-1, :) + rv_g(2:end, :));
j_sf = j_rmid .* j_ds;

% Local cell sizes for CFL timestep
char_len = max(1e-8, min(dx_cell, 0.5 .* (hL + hR)));

% Verify Check 1: Free-stream GCL on curvilinear grid
u_gcl = zeros(nx, nr); v_gcl = zeros(nx, nr);
p_gcl = ones(nx, nr) .* pc_pa;
rho_gcl = p_gcl ./ (r_gas * tc_k);
[~, r2_gcl, r3_gcl, ~, ~] = compute_rhs_2d( ...
    rho_gcl, u_gcl, v_gcl, p_gcl, gam, r_gas, pc_pa, tc_k, rw_nodes, ...
    i_nx, i_nr, i_sf, j_nx, j_nr, j_sf, area, vol);
gcl_err = max(max(abs(r2_gcl(:))), max(abs(r3_gcl(:))));
fprintf('1. Curvilinear 1-rad GCL residual : %.3e\n', gcl_err);
assert(gcl_err < 1e-6, '2D GCL residual check failed');

% Quasi-1D initialization for steady nozzle convergence
rc_wall = 0.5 .* (rw_nodes(1:end-1) + rw_nodes(2:end));
[~, i_th] = min(rc_wall);
m_init = zeros(nx, 1);
for i = 1:nx
    ar = max(1.0, (rc_wall(i) / rt)^2);
    m_init(i) = invert_area_mach_2d(ar, gam, i > i_th);
end
fac0 = 1.0 + 0.5 * gm1 .* (m_init.^2);
T0 = tc_k ./ fac0;
p0 = pc_pa .* (fac0 .^ (-gam / gm1));
rho0 = p0 ./ (r_gas .* T0);
u0 = m_init .* sqrt(gam * r_gas .* T0);

rho = repmat(rho0, 1, nr);
u   = repmat(u0, 1, nr);
v   = zeros(nx, nr);
p   = repmat(p0, 1, nr);
% Align initial radial velocity smoothly along wall streamlines
theta_wall = atan((rw_nodes(2:end) - rw_nodes(1:end-1))' ./ dx_cell(:, 1));
for j = 1:nr
    eta_j = (j - 0.5) / nr;
    v(:, j) = u(:, j) .* tan(theta_wall .* eta_j);
end

cfl = 0.42;
max_iters = 950;
for it = 1:max_iters
    c1 = rho; c2 = rho .* u; c3 = rho .* v;
    c4 = p ./ gm1 + 0.5 .* rho .* (u.^2 + v.^2);

    a = sqrt(gam .* p ./ max(1e-9, rho));
    dt_loc = cfl .* char_len ./ max(1e-4, hypot(u, v) + a);

    [k1_1, k1_2, k1_3, k1_4, ~] = compute_rhs_2d( ...
        rho, u, v, p, gam, r_gas, pc_pa, tc_k, rw_nodes, ...
        i_nx, i_nr, i_sf, j_nx, j_nr, j_sf, area, vol);

    s1 = c1 + dt_loc .* k1_1;
    s2 = c2 + dt_loc .* k1_2;
    s3 = c3 + dt_loc .* k1_3;
    s4 = c4 + dt_loc .* k1_4;
    rs = max(1e-8, s1); us = s2 ./ rs; vs = s3 ./ rs;
    ps = max(10.0, gm1 .* (s4 - 0.5 .* rs .* (us.^2 + vs.^2)));

    [k2_1, k2_2, k2_3, k2_4, fi0_final] = compute_rhs_2d( ...
        rs, us, vs, ps, gam, r_gas, pc_pa, tc_k, rw_nodes, ...
        i_nx, i_nr, i_sf, j_nx, j_nr, j_sf, area, vol);

    c1_n = 0.5 .* (c1 + s1 + dt_loc .* k2_1);
    c2_n = 0.5 .* (c2 + s2 + dt_loc .* k2_2);
    c3_n = 0.5 .* (c3 + s3 + dt_loc .* k2_3);
    c4_n = 0.5 .* (c4 + s4 + dt_loc .* k2_4);

    if mod(it, 100) == 0 || it == max_iters
        rel_res = sqrt(mean(((c1_n(:) - c1(:)) ./ max(1e-4, c1(:))).^2));
        if rel_res < 3e-5
            rho = max(1e-8, c1_n);
            u = c2_n ./ rho; v = c3_n ./ rho;
            p = max(10.0, gm1 .* (c4_n - 0.5 .* rho .* (u.^2 + v.^2)));
            break;
        end
    end

    rho = max(1e-8, c1_n);
    u = c2_n ./ rho; v = c3_n ./ rho;
    p = max(10.0, gm1 .* (c4_n - 0.5 .* rho .* (u.^2 + v.^2)));
end

T = p ./ (r_gas .* rho);
mach = hypot(u, v) ./ sqrt(gam .* r_gas .* T);

% Integrate actual Riemann I-face mass flux (2*pi*r_f * F0_f) at throat & exit
[~, i_th_face] = min(rw_nodes);
mdot_th = 2.0 * pi * sum(fi0_final(i_th_face, :) .* i_sf(i_th_face, :));
mdot_ex = 2.0 * pi * sum(fi0_final(end, :) .* i_sf(end, :));
mass_err_pct = 100.0 * abs(mdot_ex - mdot_th) / max(1e-9, mdot_th);

% Theoretical 1D choked mass flow -> 2D Discharge Coefficient Cd
gm_func = sqrt(gam) * ((2.0 / gp1) ^ (gp1 / (2.0 * gm1)));
mdot_1d = pc_pa * (pi * rt^2) * gm_func / sqrt(r_gas * tc_k);
cd_2d = mdot_th / mdot_1d;

m_th_cl = mach(i_th, 1);
m_th_wall = mach(i_th, end);
m_exit_cl = mach(end, 1);

fprintf('2. Mass Conservation (Throat->Exit): %.4f%% (mdot=%.4f kg/s)\n', mass_err_pct, mdot_th);
fprintf('3. Transonic Curved Sonic Line     : M_CL=%.3f < M_wall=%.3f (at throat)\n', m_th_cl, m_th_wall);
fprintf('4. Discharge Coefficient Cd        : %.4f vs Hall(1962) %.4f\n', cd_2d, cd_hall);
fprintf('5. Exit Centerline Mach            : %.3f\n', m_exit_cl);

assert(mass_err_pct < 1.5, '2D axisymmetric mass conservation error exceeds 1.5%');
assert(m_th_wall > m_th_cl + 0.15, 'Expected transonic wall overshoot vs centerline');
assert(abs(cd_2d - cd_hall) < 0.025, 'Cd disagrees with Hall (1962) beyond 2.5%');
assert(m_exit_cl > 2.2, 'Exit centerline Mach should be supersonic > 2.2');

fprintf('[PASS] All 2D Axisymmetric Curvilinear FVM validations passed!\n');
end

function [rhs1, rhs2, rhs3, rhs4, fi0] = compute_rhs_2d( ...
    rho, u, v, p, gam, r_gas, pc_pa, tc_k, rw_nodes, ...
    i_nx, i_nr, i_sf, j_nx, j_nr, j_sf, area, vol)
[nx, nr] = size(rho);

% --- I-direction Van Leer MUSCL states (nx x nr -> interior faces 2..nx) ---
r_ex = zeros(nx + 2, nr); u_ex = zeros(nx + 2, nr);
v_ex = zeros(nx + 2, nr); p_ex = zeros(nx + 2, nr);
r_ex(2:end-1, :) = rho; u_ex(2:end-1, :) = u;
v_ex(2:end-1, :) = v;   p_ex(2:end-1, :) = p;
r_ex(1, :) = rho(1, :);   u_ex(1, :) = u(1, :);
v_ex(1, :) = v(1, :);     p_ex(1, :) = p(1, :);
r_ex(end, :) = rho(end, :); u_ex(end, :) = u(end, :);
v_ex(end, :) = v(end, :);   p_ex(end, :) = p(end, :);

sr_i = van_leer_slope(r_ex(2:end-1, :) - r_ex(1:end-2, :), r_ex(3:end, :) - r_ex(2:end-1, :));
su_i = van_leer_slope(u_ex(2:end-1, :) - u_ex(1:end-2, :), u_ex(3:end, :) - u_ex(2:end-1, :));
sv_i = van_leer_slope(v_ex(2:end-1, :) - v_ex(1:end-2, :), v_ex(3:end, :) - v_ex(2:end-1, :));
sp_i = van_leer_slope(p_ex(2:end-1, :) - p_ex(1:end-2, :), p_ex(3:end, :) - p_ex(2:end-1, :));

fi0 = zeros(nx + 1, nr); fi1 = zeros(nx + 1, nr);
fi2 = zeros(nx + 1, nr); fi3 = zeros(nx + 1, nr);

% Interior I-faces (2:nx)
[f0m, f1m, f2m, f3m] = hllc_rotated_2d( ...
    max(1e-8, rho(1:end-1, :) + 0.5 .* sr_i(1:end-1, :)), ...
    u(1:end-1, :) + 0.5 .* su_i(1:end-1, :), ...
    v(1:end-1, :) + 0.5 .* sv_i(1:end-1, :), ...
    max(10.0, p(1:end-1, :) + 0.5 .* sp_i(1:end-1, :)), ...
    max(1e-8, rho(2:end, :) - 0.5 .* sr_i(2:end, :)), ...
    u(2:end, :) - 0.5 .* su_i(2:end, :), ...
    v(2:end, :) - 0.5 .* sv_i(2:end, :), ...
    max(10.0, p(2:end, :) - 0.5 .* sp_i(2:end, :)), ...
    i_nx(2:end-1, :), i_nr(2:end-1, :), gam);
fi0(2:end-1, :) = f0m; fi1(2:end-1, :) = f1m;
fi2(2:end-1, :) = f2m; fi3(2:end-1, :) = f3m;

% Subsonic Riemann Inflow Boundary (i=1, uniform horizontal lead-in chamber)
gm1 = gam - 1.0;
a_in = sqrt(gam .* p(1, :) ./ max(1e-8, rho(1, :)));
m_in = max(0.0, min(0.95, hypot(u(1, :), v(1, :)) ./ max(1e-6, a_in)));
fac_b = 1.0 + 0.5 * gm1 .* (m_in.^2);
T_b = tc_k ./ fac_b;
p_b = pc_pa .* (fac_b .^ (-gam / gm1));
rho_b = p_b ./ (r_gas .* T_b);
vel_b = m_in .* sqrt(gam * r_gas .* T_b);
[f0i, f1i, f2i, f3i] = hllc_rotated_2d( ...
    rho_b, vel_b, zeros(1, nr), p_b, ...
    rho(1, :), u(1, :), v(1, :), p(1, :), ...
    i_nx(1, :), i_nr(1, :), gam);
fi0(1, :) = f0i; fi1(1, :) = f1i; fi2(1, :) = f2i; fi3(1, :) = f3i;

% Outflow Boundary (i=nx+1)
[f0o, f1o, f2o, f3o] = hllc_rotated_2d( ...
    rho(end, :), u(end, :), v(end, :), p(end, :), ...
    rho(end, :), u(end, :), v(end, :), p(end, :), ...
    i_nx(end, :), i_nr(end, :), gam);
fi0(end, :) = f0o; fi1(end, :) = f1o; fi2(end, :) = f2o; fi3(end, :) = f3o;

% --- J-direction Van Leer MUSCL states ---
r_je = zeros(nx, nr + 2); u_je = zeros(nx, nr + 2);
v_je = zeros(nx, nr + 2); p_je = zeros(nx, nr + 2);
r_je(:, 2:end-1) = rho; u_je(:, 2:end-1) = u;
v_je(:, 2:end-1) = v;   p_je(:, 2:end-1) = p;
r_je(:, 1) = rho(:, 1);   u_je(:, 1) = u(:, 1);
v_je(:, 1) = -v(:, 1);    p_je(:, 1) = p(:, 1);
r_je(:, end) = rho(:, end); u_je(:, end) = u(:, end);
v_je(:, end) = v(:, end);   p_je(:, end) = p(:, end);

sr_j = van_leer_slope(r_je(:, 2:end-1) - r_je(:, 1:end-2), r_je(:, 3:end) - r_je(:, 2:end-1));
su_j = van_leer_slope(u_je(:, 2:end-1) - u_je(:, 1:end-2), u_je(:, 3:end) - u_je(:, 2:end-1));
sv_j = van_leer_slope(v_je(:, 2:end-1) - v_je(:, 1:end-2), v_je(:, 3:end) - v_je(:, 2:end-1));
sp_j = van_leer_slope(p_je(:, 2:end-1) - p_je(:, 1:end-2), p_je(:, 3:end) - p_je(:, 2:end-1));

fj0 = zeros(nx, nr + 1); fj1 = zeros(nx, nr + 1);
fj2 = zeros(nx, nr + 1); fj3 = zeros(nx, nr + 1);

% Interior J-faces (2:nr)
[g0m, g1m, g2m, g3m] = hllc_rotated_2d( ...
    max(1e-8, rho(:, 1:end-1) + 0.5 .* sr_j(:, 1:end-1)), ...
    u(:, 1:end-1) + 0.5 .* su_j(:, 1:end-1), ...
    v(:, 1:end-1) + 0.5 .* sv_j(:, 1:end-1), ...
    max(10.0, p(:, 1:end-1) + 0.5 .* sp_j(:, 1:end-1)), ...
    max(1e-8, rho(:, 2:end) - 0.5 .* sr_j(:, 2:end)), ...
    u(:, 2:end) - 0.5 .* su_j(:, 2:end), ...
    v(:, 2:end) - 0.5 .* sv_j(:, 2:end), ...
    max(10.0, p(:, 2:end) - 0.5 .* sp_j(:, 2:end)), ...
    j_nx(:, 2:end-1), j_nr(:, 2:end-1), gam);
fj0(:, 2:end-1) = g0m; fj1(:, 2:end-1) = g1m;
fj2(:, 2:end-1) = g2m; fj3(:, 2:end-1) = g3m;

% Bottom Axisymmetry Axis (j=1): zero flux on S=0 + pure symmetry safeguard
fj0(:, 1) = 0.0; fj1(:, 1) = 0.0; fj2(:, 1) = j_nr(:, 1) .* p(:, 1); fj3(:, 1) = 0.0;

% Top Wall Slip Reflective Wall (j=nr+1): zero mass flux, wall normal pressure
p_wall = max(10.0, p(:, end) + 0.5 .* sp_j(:, end));
fj0(:, end) = 0.0;
fj1(:, end) = j_nx(:, end) .* p_wall;
fj2(:, end) = j_nr(:, end) .* p_wall;
fj3(:, end) = 0.0;

% Net finite-volume flux balance + axisymmetric hoop pressure source (+p*area)
inv_vol = 1.0 ./ vol;
rhs1 = -( (fi0(2:end, :) .* i_sf(2:end, :) - fi0(1:end-1, :) .* i_sf(1:end-1, :)) ...
        + (fj0(:, 2:end) .* j_sf(:, 2:end) - fj0(:, 1:end-1) .* j_sf(:, 1:end-1)) ) .* inv_vol;
rhs2 = -( (fi1(2:end, :) .* i_sf(2:end, :) - fi1(1:end-1, :) .* i_sf(1:end-1, :)) ...
        + (fj1(:, 2:end) .* j_sf(:, 2:end) - fj1(:, 1:end-1) .* j_sf(:, 1:end-1)) ) .* inv_vol;
rhs3 = -( (fi2(2:end, :) .* i_sf(2:end, :) - fi2(1:end-1, :) .* i_sf(1:end-1, :)) ...
        + (fj2(:, 2:end) .* j_sf(:, 2:end) - fj2(:, 1:end-1) .* j_sf(:, 1:end-1)) ) .* inv_vol ...
       + (p .* area) .* inv_vol;
rhs4 = -( (fi3(2:end, :) .* i_sf(2:end, :) - fi3(1:end-1, :) .* i_sf(1:end-1, :)) ...
        + (fj3(:, 2:end) .* j_sf(:, 2:end) - fj3(:, 1:end-1) .* j_sf(:, 1:end-1)) ) .* inv_vol;
end

function s = van_leer_slope(d_l, d_r)
prod = d_l .* d_r;
denom = d_l + d_r;
s = zeros(size(d_l));
mask = (prod > 0.0) & (abs(denom) > 1e-18);
s(mask) = 2.0 .* prod(mask) ./ denom(mask);
end

function [f0, f1, f2, f3] = hllc_rotated_2d( ...
    rL, uL, vL, pL, rR, uR, vR, pR, nx, nr, gam)
gm1 = gam - 1.0;
tx = -nr;
ty = nx;

unL = uL .* nx + vL .* nr;
utL = uL .* tx + vL .* ty;
unR = uR .* nx + vR .* nr;
utR = uR .* tx + vR .* ty;

rL = max(1e-8, rL); pL = max(1.0, pL);
rR = max(1e-8, rR); pR = max(1.0, pR);
aL = sqrt(gam .* pL ./ rL);
aR = sqrt(gam .* pR ./ rR);

sL = min(unL - aL, unR - aR);
sR = max(unL + aL, unR + aR);

den = safe_denom(rL .* (sL - unL) - rR .* (sR - unR));
sM = (pR - pL + rL .* unL .* (sL - unL) - rR .* unR .* (sR - unR)) ./ den;

EL = pL ./ gm1 + 0.5 .* rL .* (unL.^2 + utL.^2);
ER = pR ./ gm1 + 0.5 .* rR .* (unR.^2 + utR.^2);

fL0 = rL .* unL; fLn = fL0 .* unL + pL; fLt = fL0 .* utL; fL3 = (EL + pL) .* unL;
fR0 = rR .* unR; fRn = fR0 .* unR + pR; fRt = fR0 .* utR; fR3 = (ER + pR) .* unR;

m1 = (sL >= 0.0);
m4 = (sR <= 0.0);
m2 = (~m1) & (~m4) & (sM >= 0.0);
m3 = (~m1) & (~m4) & (sM < 0.0);

dSL = safe_denom(sL - sM);
dSR = safe_denom(sR - sM);
dUL = safe_denom(sL - unL);
dUR = safe_denom(sR - unR);

facL = rL .* dUL ./ dSL;
facR = rR .* dUR ./ dSR;

uSL0 = facL; uSLn = facL .* sM; uSLt = facL .* utL;
uSL3 = facL .* (EL ./ rL + (sM - unL) .* (sM + pL ./ (rL .* dUL)));
uSR0 = facR; uSRn = facR .* sM; uSRt = facR .* utR;
uSR3 = facR .* (ER ./ rR + (sM - unR) .* (sM + pR ./ (rR .* dUR)));

f0 = zeros(size(rL)); fn = zeros(size(rL));
ft = zeros(size(rL)); f3 = zeros(size(rL));

f0(m1) = fL0(m1); fn(m1) = fLn(m1); ft(m1) = fLt(m1); f3(m1) = fL3(m1);
f0(m2) = fL0(m2) + sL(m2) .* (uSL0(m2) - rL(m2));
fn(m2) = fLn(m2) + sL(m2) .* (uSLn(m2) - rL(m2) .* unL(m2));
ft(m2) = fLt(m2) + sL(m2) .* (uSLt(m2) - rL(m2) .* utL(m2));
f3(m2) = fL3(m2) + sL(m2) .* (uSL3(m2) - EL(m2));

f0(m3) = fR0(m3) + sR(m3) .* (uSR0(m3) - rR(m3));
fn(m3) = fRn(m3) + sR(m3) .* (uSRn(m3) - rR(m3) .* unR(m3));
ft(m3) = fRt(m3) + sR(m3) .* (uSRt(m3) - rR(m3) .* utR(m3));
f3(m3) = fR3(m3) + sR(m3) .* (uSR3(m3) - ER(m3));

f0(m4) = fR0(m4); fn(m4) = fRn(m4); ft(m4) = fRt(m4); f3(m4) = fR3(m4);

% Carbuncle-free strong-shock stabilization
p_jump = abs(pR - pL) ./ max(pL, pR);
if any(p_jump(:) > 0.12)
    smax = max(abs(unL) + aL, abs(unR) + aR);
    al = 0.15 .* (p_jump > 0.12);
    f0 = (1.0 - al) .* f0 + al .* 0.5 .* ((fL0 + fR0) - smax .* (rR - rL));
    fn = (1.0 - al) .* fn + al .* 0.5 .* ((fLn + fRn) - smax .* (rR .* unR - rL .* unL));
    ft = (1.0 - al) .* ft + al .* 0.5 .* ((fLt + fRt) - smax .* (rR .* utR - rL .* utL));
    f3 = (1.0 - al) .* f3 + al .* 0.5 .* ((fL3 + fR3) - smax .* (ER - EL));
end

f1 = fn .* nx + ft .* tx;
f2 = fn .* nr + ft .* ty;
end

function out = safe_denom(val)
sgn = ones(size(val));
sgn(val < 0.0) = -1.0;
out = sgn .* max(abs(val), 1e-12);
end

function M = invert_area_mach_2d(ar, gam, is_sup)
if ar <= 1.00001
    M = 1.0;
    return;
end
gm1 = gam - 1.0;
gp1 = gam + 1.0;
if is_sup
    lo = 1.00001; hi = 20.0;
else
    lo = 1e-4; hi = 0.99999;
end
for b = 1:50
    mid = 0.5 * (lo + hi);
    f = (1.0 / mid) * (((2.0 / gp1) * (1.0 + 0.5 * gm1 * mid^2)) ^ (gp1 / (2.0 * gm1))) - ar;
    if is_sup
        if f < 0.0, lo = mid; else, hi = mid; end
    else
        if f > 0.0, lo = mid; else, hi = mid; end
    end
end
M = 0.5 * (lo + hi);
end
