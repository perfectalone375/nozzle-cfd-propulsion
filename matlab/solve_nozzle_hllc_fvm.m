function solve_nozzle_hllc_fvm()
% SOLVE_NOZZLE_HLLC_FVM
% Finite-Volume HLLC+ Approximate Riemann Solver with Van Leer MUSCL-TVD & SSP-RK2
% Compatible with MATLAB (R2016b+) and GNU Octave (5.0+).
%
% Verifies:
%   1) Sod Shock Tube rarefaction + contact discontinuity + shock capture
%   2) Overexpanded Rocket Nozzle capturing standing normal shock without MacCormack ringing
%
% Author: Gopi Naga Vignesh Thirupati (nozzle-cfd-propulsion)

fprintf('=== MATLAB/Octave Finite-Volume HLLC+ MUSCL-TVD CFD Solver ===\n');

%% 1. Benchmark A: Classic NASA/Toro 1D Sod Shock Tube
nx_sod = 200;
dx_s = 1.0 / nx_sod;
x_s = ((1:nx_sod) - 0.5) * dx_s;
gam_s = 1.4;
gm1_s = gam_s - 1.0;

rho = ones(1, nx_sod); rho(x_s >= 0.5) = 0.125;
u   = zeros(1, nx_sod);
p   = ones(1, nx_sod); p(x_s >= 0.5) = 0.1;

u1 = rho;
u2 = rho .* u;
u3 = p ./ gm1_s + 0.5 .* rho .* (u.^2);

t = 0.0; t_end = 0.20; cfl = 0.45;
while t < t_end - 1e-12
    r_c = max(1e-9, u1);
    u_c = u2 ./ r_c;
    p_c = max(1e-9, gm1_s .* (u3 - 0.5 .* r_c .* (u_c.^2)));
    a_c = sqrt(gam_s .* p_c ./ r_c);
    dt = min(t_end - t, cfl * dx_s / max(abs(u_c) + a_c));

    [k1_1, k1_2, k1_3] = sod_rhs(u1, u2, u3, dx_s, gam_s);
    u1_s = u1 + dt .* k1_1; u2_s = u2 + dt .* k1_2; u3_s = u3 + dt .* k1_3;
    [k2_1, k2_2, k2_3] = sod_rhs(u1_s, u2_s, u3_s, dx_s, gam_s);

    u1 = 0.5 .* (u1 + u1_s + dt .* k2_1);
    u2 = 0.5 .* (u2 + u2_s + dt .* k2_2);
    u3 = 0.5 .* (u3 + u3_s + dt .* k2_3);
    t = t + dt;
end

rho_fin = max(1e-9, u1);
u_fin = u2 ./ rho_fin;
p_fin = max(1e-9, gm1_s .* (u3 - 0.5 .* rho_fin .* (u_fin.^2)));
fprintf('Sod Tube t=0.20 | rho(0.1)=%.3f rho(0.6)=%.3f p(0.6)=%.3f u(0.6)=%.3f\n', ...
    rho_fin(20), rho_fin(120), p_fin(120), u_fin(120));
assert(abs(p_fin(120) - 0.3031) < 0.02, 'Sod star pressure mismatch');
assert(abs(u_fin(120) - 0.9275) < 0.03, 'Sod star velocity mismatch');

%% 2. Benchmark B: Overexpanded Rocket Nozzle Standing Normal Shock
gam = 1.21;
gm1 = gam - 1.0;
nx = 121;
x_nd = linspace(0.0, 3.0, nx);
dx = x_nd(2) - x_nd(1);

A = zeros(1, nx);
for i = 1:nx
    if x_nd(i) <= 1.0
        A(i) = 1.0 + 2.2 * (1.0 - x_nd(i))^2;
    else
        A(i) = 1.0 + 7.0 * (((x_nd(i) - 1.0) / 2.0)^1.45);
    end
end
[~, throat_idx] = min(A);

a_int = zeros(1, nx + 1);
a_int(1) = A(1);
a_int(2:nx) = 0.5 .* (A(1:end-1) + A(2:end));
a_int(nx + 1) = max(1.0, A(end) + 0.5 * (A(end) - A(end - 1)));
da_cell = a_int(2:end) - a_int(1:end-1);

pb_nd = 0.38;  % Back pressure Pb/Pc = 380 kPa / 1.0 MPa forcing shock in bell
m_init = zeros(1, nx);
for i = 1:nx
    m_init(i) = invert_area_mach(A(i), gam, i > throat_idx);
    if i > throat_idx
        frac = (i - throat_idx) / (nx - throat_idx);
        m_init(i) = max(0.25, m_init(i) * (1.0 - 0.65 * frac));
    end
end
t_0 = 1.0 ./ (1.0 + 0.5 * gm1 .* (m_init.^2));
p_0 = t_0 .^ (gam / gm1);
r_0 = p_0 ./ t_0;
v_0 = m_init .* sqrt(gam .* t_0);

nu1 = r_0; nu2 = r_0 .* v_0; nu3 = p_0 ./ gm1 + 0.5 .* r_0 .* (v_0.^2);
res = 1.0;
for it = 1:9000
    ri = max(1e-9, nu1);
    ui = nu2 ./ ri;
    pi = max(1e-9, gm1 .* (nu3 - 0.5 .* ri .* (ui.^2)));
    ai = sqrt(gam .* pi ./ ri);
    dt_n = 0.50 * dx / max(abs(ui) + ai);

    [k1_1, k1_2, k1_3] = nozzle_rhs(nu1, nu2, nu3, A, a_int, da_cell, dx, gam, pb_nd);
    s1 = nu1 + dt_n .* k1_1; s2 = nu2 + dt_n .* k1_2; s3 = nu3 + dt_n .* k1_3;
    [k2_1, k2_2, k2_3] = nozzle_rhs(s1, s2, s3, A, a_int, da_cell, dx, gam, pb_nd);

    n1 = 0.5 .* (nu1 + s1 + dt_n .* k2_1);
    n2 = 0.5 .* (nu2 + s2 + dt_n .* k2_2);
    n3 = 0.5 .* (nu3 + s3 + dt_n .* k2_3);
    res = max(abs(n2 - nu2) ./ max(abs(nu2), 1e-4));
    nu1 = n1; nu2 = n2; nu3 = n3;
    if it > 200 && res < 1e-5
        break;
    end
end

r_f = max(1e-9, nu1);
u_f = nu2 ./ r_f;
p_f = max(1e-9, gm1 .* (nu3 - 0.5 .* r_f .* (u_f.^2)));
m_f = u_f ./ sqrt(gam .* p_f ./ r_f);
max_m_div = max(m_f(throat_idx:end));
exit_m = m_f(end);
exit_p_nd = p_f(end);

fprintf('Nozzle FVM | max_M=%.3f -> exit_M=%.3f | exit P/Pc=%.4f (target %.4f) | res=%.2e\n', ...
    max_m_div, exit_m, exit_p_nd, pb_nd, res);
assert(max_m_div > 1.8 && exit_m < 0.7, 'Must capture supersonic -> subsonic normal shock');
assert(abs(exit_p_nd - pb_nd) < 0.015, 'Exit pressure must match ambient back pressure');
fprintf('ALL MATLAB/OCTAVE HLLC+ FVM ASSERTIONS PASSED.\n');
end

function slope = vanleer_slopes(q)
n_tot = length(q);
slope = zeros(1, n_tot - 2);
for i = 2:(n_tot - 1)
    dm = q(i) - q(i - 1);
    dp = q(i + 1) - q(i);
    if dm * dp > 0.0
        den = dp;
        if abs(den) < 1e-14
            den = 1e-14;
        end
        r = dm / den;
        slope(i - 1) = (2.0 * r / (1.0 + r)) * dp;
    end
end
end

function [f1, f2, f3] = hllc_flux(rl, ul, pl, rr, ur, pr, gam)
gm1 = gam - 1.0;
el = pl / gm1 + 0.5 * rl * ul^2;
er = pr / gm1 + 0.5 * rr * ur^2;
al = sqrt(gam * pl / rl);
ar = sqrt(gam * pr / rr);

f1l = rl * ul; f2l = rl * ul^2 + pl; f3l = ul * (el + pl);
f1r = rr * ur; f2r = rr * ur^2 + pr; f3r = ur * (er + pr);

sql = sqrt(rl); sqr = sqrt(rr);
uroe = (sql * ul + sqr * ur) / (sql + sqr);
hroe = (sql * ((el + pl) / rl) + sqr * ((er + pr) / rr)) / (sql + sqr);
aroe = sqrt(max(1e-9, gm1 * (hroe - 0.5 * uroe^2)));

sl = min(ul - al, uroe - aroe);
sr = max(ur + ar, uroe + aroe);
den = rl * (sl - ul) - rr * (sr - ur);
if abs(den) < 1e-12
    den = 1e-12;
end
sstar = (pr - pl + rl * ul * (sl - ul) - rr * ur * (sr - ur)) / den;

if sl >= 0.0
    f1 = f1l; f2 = f2l; f3 = f3l;
elseif sstar >= 0.0
    fac = rl * (sl - ul) / min(-1e-12, sl - sstar);
    u3s = fac * (el / rl + (sstar - ul) * (sstar + pl / (rl * (sl - ul) - 1e-15)));
    f1 = f1l + sl * (fac - rl);
    f2 = f2l + sl * (fac * sstar - rl * ul);
    f3 = f3l + sl * (u3s - el);
elseif sr >= 0.0
    fac = rr * (sr - ur) / max(1e-12, sr - sstar);
    u3s = fac * (er / rr + (sstar - ur) * (sstar + pr / (rr * (sr - ur) + 1e-15)));
    f1 = f1r + sr * (fac - rr);
    f2 = f2r + sr * (fac * sstar - rr * ur);
    f3 = f3r + sr * (u3s - er);
else
    f1 = f1r; f2 = f2r; f3 = f3r;
end

p_jump = abs(pr - pl) / (pr + pl + 1e-12);
if sl < 0.0 && sr > 0.0 && p_jump > 0.08
    alp = min(0.35, 0.5 * p_jump^2);
    sr_sl = max(1e-9, sr - sl);
    f1h = (sr * f1l - sl * f1r + sl * sr * (rr - rl)) / sr_sl;
    f2h = (sr * f2l - sl * f2r + sl * sr * (rr * ur - rl * ul)) / sr_sl;
    f3h = (sr * f3l - sl * f3r + sl * sr * (er - el)) / sr_sl;
    f1 = (1.0 - alp) * f1 + alp * f1h;
    f2 = (1.0 - alp) * f2 + alp * f2h;
    f3 = (1.0 - alp) * f3 + alp * f3h;
end
end

function [d1, d2, d3] = sod_rhs(c1, c2, c3, dx, gam)
n = length(c1); gm1 = gam - 1.0;
r = max(1e-9, c1); u = c2 ./ r; p = max(1e-9, gm1 .* (c3 - 0.5 .* r .* (u.^2)));
rp = [r(1), r(1), r, r(end), r(end)];
up = [u(1), u(1), u, u(end), u(end)];
pp = [p(1), p(1), p, p(end), p(end)];
sr = vanleer_slopes(rp); su = vanleer_slopes(up); sp = vanleer_slopes(pp);
f1 = zeros(1, n + 1); f2 = zeros(1, n + 1); f3 = zeros(1, n + 1);
for j = 1:(n + 1)
    [f1(j), f2(j), f3(j)] = hllc_flux( ...
        max(1e-9, rp(j+1) + 0.5*sr(j)), up(j+1) + 0.5*su(j), max(1e-9, pp(j+1) + 0.5*sp(j)), ...
        max(1e-9, rp(j+2) - 0.5*sr(j+1)), up(j+2) - 0.5*su(j+1), max(1e-9, pp(j+2) - 0.5*sp(j+1)), gam);
end
d1 = -(f1(2:end) - f1(1:end-1)) ./ dx;
d2 = -(f2(2:end) - f2(1:end-1)) ./ dx;
d3 = -(f3(2:end) - f3(1:end-1)) ./ dx;
end

function [d1, d2, d3] = nozzle_rhs(c1, c2, c3, A, a_int, da_cell, dx, gam, pb_nd)
n = length(c1); gm1 = gam - 1.0;
r = max(1e-9, c1); u = c2 ./ r; p = max(1e-9, gm1 .* (c3 - 0.5 .* r .* (u.^2)));

a0 = sqrt(gam * p(1) / r(1));
rmin = u(1) - 2.0 * a0 / gm1;
aq = 1.0 / gm1 + 2.0 / (gm1^2);
bq = 2.0 * rmin / gm1;
cq = 0.5 * rmin^2 - gam / gm1;
ain = max(0.2, (-bq + sqrt(max(0.0, bq^2 - 4.0 * aq * cq))) / (2.0 * aq));
uin = max(0.005 * ain, min(0.92 * ain, rmin + 2.0 * ain / gm1));
tin = min(1.0, max(0.4, ain^2 / gam));
pin = tin ^ (gam / gm1); rin = pin / tin;

ae = sqrt(gam * p(end) / r(end));
pout = pb_nd; se = p(end) / (r(end)^gam);
rout = max(1e-6, (pout / se)^(1.0 / gam));
uout = max(0.02, u(end) + (p(end) - pout) / (r(end) * ae));

rp = [rin, rin, r, rout, rout];
up = [uin, uin, u, uout, uout];
pp = [pin, pin, p, pout, pout];
sr = vanleer_slopes(rp); su = vanleer_slopes(up); sp = vanleer_slopes(pp);
for c = 2:(n + 3)
    sw = abs(pp(c+1) - 2*pp(c) + pp(c-1)) / max(pp(c+1) + 2*pp(c) + pp(c-1), 1e-12);
    damp = min(1.0, max(0.0, 1.0 - 8.0 * max(0.0, sw - 0.03)));
    sr(c-1) = damp * sr(c-1); su(c-1) = damp * su(c-1); sp(c-1) = damp * sp(c-1);
end

f1 = zeros(1, n + 1); f2 = zeros(1, n + 1); f3 = zeros(1, n + 1);
for j = 1:(n + 1)
    [f1(j), f2(j), f3(j)] = hllc_flux( ...
        max(1e-9, rp(j+1) + 0.5*sr(j)), up(j+1) + 0.5*su(j), max(1e-9, pp(j+1) + 0.5*sp(j)), ...
        max(1e-9, rp(j+2) - 0.5*sr(j+1)), up(j+2) - 0.5*su(j+1), max(1e-9, pp(j+2) - 0.5*sp(j+1)), gam);
end
inv_v = 1.0 ./ (A .* dx);
d1 = -(f1(2:end) .* a_int(2:end) - f1(1:end-1) .* a_int(1:end-1)) .* inv_v;
d2 = -(f2(2:end) .* a_int(2:end) - f2(1:end-1) .* a_int(1:end-1)) .* inv_v + p .* da_cell .* inv_v;
d3 = -(f3(2:end) .* a_int(2:end) - f3(1:end-1) .* a_int(1:end-1)) .* inv_v;
end

function M = invert_area_mach(ar, gam, is_sup)
if ar <= 1.00001
    M = 1.0;
    return;
end
gm1 = gam - 1.0;
gp1 = gam + 1.0;
if is_sup
    lo = 1.00001;
    hi = 20.0;
else
    lo = 1e-4;
    hi = 0.99999;
end
for b = 1:55
    mid = 0.5 * (lo + hi);
    f = (1.0 / mid) * (((2.0 / gp1) * (1.0 + 0.5 * gm1 * mid^2)) ^ (gp1 / (2.0 * gm1))) - ar;
    if is_sup
        if f < 0.0
            lo = mid;
        else
            hi = mid;
        end
    else
        if f > 0.0
            lo = mid;
        else
            hi = mid;
        end
    end
end
M = 0.5 * (lo + hi);
end
