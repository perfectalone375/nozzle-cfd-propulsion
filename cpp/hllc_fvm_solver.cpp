// High-Speed C++20 Finite-Volume Riemann Solver (HLLC+, AUSM+, Rusanov) + MUSCL-TVD + SSP-RK2
// Standalone binary: g++ -O3 -std=c++20 -Wall -Wextra -o hllc_fvm_cfd hllc_fvm_solver.cpp
// Shared DLL (Win):  g++ -O3 -std=c++20 -shared -DNOZZLE_FVM_BUILD_DLL -o nozzle_fvm_core.dll hllc_fvm_solver.cpp
// Shared SO (Linux): g++ -O3 -std=c++20 -shared -fPIC -DNOZZLE_FVM_BUILD_DLL -o libnozzle_fvm_core.so hllc_fvm_solver.cpp

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#ifdef _WIN32
#define FVM_API __declspec(dllexport)
#else
#define FVM_API
#endif

namespace fvm_core {

inline double tvd_limiter(double dq_minus, double dq_plus, int limiter_id) {
    if (limiter_id == 0) return 0.0;  // first_order
    if (dq_minus * dq_plus <= 0.0) return 0.0;
    double denom = (std::abs(dq_plus) > 1e-14) ? dq_plus : std::copysign(1e-14, dq_plus + 1e-30);
    double r = dq_minus / denom;
    double phi = 0.0;
    if (limiter_id == 1) {
        phi = std::max(0.0, std::min(1.0, r));  // minmod
    } else if (limiter_id == 2) {
        phi = (r > 0.0) ? (2.0 * r) / (1.0 + r) : 0.0;  // van_leer
    } else if (limiter_id == 3) {
        phi = std::max(0.0, std::max(std::min(2.0 * r, 1.0), std::min(r, 2.0)));  // superbee
    } else {
        phi = std::max(0.0, std::min(2.0 * r, std::min(0.5 * (1.0 + r), 2.0)));  // mc
    }
    return phi * dq_plus;
}

inline void riemann_flux(
    double rho_l, double u_l, double p_l,
    double rho_r, double u_r, double p_r,
    double gamma, int flux_id,
    double& f1, double& f2, double& f3
) {
    const double gm1 = gamma - 1.0;
    const double e_l = p_l / gm1 + 0.5 * rho_l * u_l * u_l;
    const double e_r = p_r / gm1 + 0.5 * rho_r * u_r * u_r;
    const double a_l = std::sqrt(gamma * p_l / rho_l);
    const double a_r = std::sqrt(gamma * p_r / rho_r);

    const double f1_l = rho_l * u_l;
    const double f2_l = rho_l * u_l * u_l + p_l;
    const double f3_l = u_l * (e_l + p_l);

    const double f1_r = rho_r * u_r;
    const double f2_r = rho_r * u_r * u_r + p_r;
    const double f3_r = u_r * (e_r + p_r);

    if (flux_id == 2) {  // Rusanov
        double smax = std::max(std::abs(u_l) + a_l, std::abs(u_r) + a_r);
        f1 = 0.5 * (f1_l + f1_r) - 0.5 * smax * (rho_r - rho_l);
        f2 = 0.5 * (f2_l + f2_r) - 0.5 * smax * (rho_r * u_r - rho_l * u_l);
        f3 = 0.5 * (f3_l + f3_r) - 0.5 * smax * (e_r - e_l);
        return;
    }

    if (flux_id == 1) {  // AUSM+
        double a_half = 0.5 * (a_l + a_r);
        double ml = u_l / a_half;
        double mr = u_r / a_half;
        double mp = (std::abs(ml) >= 1.0)
            ? 0.5 * (ml + std::abs(ml))
            : 0.25 * (ml + 1.0) * (ml + 1.0) + 0.125 * (ml * ml - 1.0) * (ml * ml - 1.0);
        double mm = (std::abs(mr) >= 1.0)
            ? 0.5 * (mr - std::abs(mr))
            : -0.25 * (mr - 1.0) * (mr - 1.0) - 0.125 * (mr * mr - 1.0) * (mr * mr - 1.0);
        double pp = (std::abs(ml) >= 1.0)
            ? (ml > 0.0 ? 1.0 : 0.0)
            : 0.25 * (ml + 1.0) * (ml + 1.0) * (2.0 - ml) + 0.1875 * ml * (ml * ml - 1.0) * (ml * ml - 1.0);
        double pm = (std::abs(mr) >= 1.0)
            ? (mr < 0.0 ? 1.0 : 0.0)
            : 0.25 * (mr - 1.0) * (mr - 1.0) * (2.0 + mr) - 0.1875 * mr * (mr * mr - 1.0) * (mr * mr - 1.0);

        double m_half = mp + mm;
        double p_half = pp * p_l + pm * p_r;
        double mdot = a_half * ((m_half >= 0.0) ? m_half * rho_l : m_half * rho_r);
        double psi_u = (m_half >= 0.0) ? u_l : u_r;
        double psi_h = (m_half >= 0.0) ? ((e_l + p_l) / rho_l) : ((e_r + p_r) / rho_r);
        f1 = mdot;
        f2 = mdot * psi_u + p_half;
        f3 = mdot * psi_h;
        return;
    }

    // HLLC+ (Roe-Einfeldt wave speeds + pressure-shock HLL stabilizer, exact on contact waves)
    double sq_l = std::sqrt(rho_l);
    double sq_r = std::sqrt(rho_r);
    double h_l = (e_l + p_l) / rho_l;
    double h_r = (e_r + p_r) / rho_r;
    double u_roe = (sq_l * u_l + sq_r * u_r) / (sq_l + sq_r);
    double h_roe = (sq_l * h_l + sq_r * h_r) / (sq_l + sq_r);
    double a_roe = std::sqrt(std::max(1e-9, gm1 * (h_roe - 0.5 * u_roe * u_roe)));

    double s_l = std::min(u_l - a_l, u_roe - a_roe);
    double s_r = std::max(u_r + a_r, u_roe + a_roe);
    double denom = rho_l * (s_l - u_l) - rho_r * (s_r - u_r);
    double denom_safe = (std::abs(denom) > 1e-12) ? denom : std::copysign(1e-12, denom + 1e-30);
    double s_star = (p_r - p_l + rho_l * u_l * (s_l - u_l) - rho_r * u_r * (s_r - u_r)) / denom_safe;

    if (s_l >= 0.0) {
        f1 = f1_l; f2 = f2_l; f3 = f3_l;
    } else if (s_star >= 0.0) {
        double d_l = s_l - s_star;
        double fac_l = rho_l * (s_l - u_l) / ((std::abs(d_l) > 1e-12) ? d_l : -1e-12);
        double u3s_l = fac_l * (e_l / rho_l + (s_star - u_l) * (s_star + p_l / (rho_l * (s_l - u_l) - 1e-15)));
        f1 = f1_l + s_l * (fac_l - rho_l);
        f2 = f2_l + s_l * (fac_l * s_star - rho_l * u_l);
        f3 = f3_l + s_l * (u3s_l - e_l);
    } else if (s_r >= 0.0) {
        double d_r = s_r - s_star;
        double fac_r = rho_r * (s_r - u_r) / ((std::abs(d_r) > 1e-12) ? d_r : 1e-12);
        double u3s_r = fac_r * (e_r / rho_r + (s_star - u_r) * (s_star + p_r / (rho_r * (s_r - u_r) + 1e-15)));
        f1 = f1_r + s_r * (fac_r - rho_r);
        f2 = f2_r + s_r * (fac_r * s_star - rho_r * u_r);
        f3 = f3_r + s_r * (u3s_r - e_r);
    } else {
        f1 = f1_r; f2 = f2_r; f3 = f3_r;
    }

    double p_jump = std::abs(p_r - p_l) / (p_r + p_l + 1e-12);
    if (s_l < 0.0 && s_r > 0.0 && p_jump > 0.08) {
        double alpha = std::min(0.35, 0.5 * p_jump * p_jump);
        double sr_sl = std::max(1e-9, s_r - s_l);
        double f1_hll = (s_r * f1_l - s_l * f1_r + s_l * s_r * (rho_r - rho_l)) / sr_sl;
        double f2_hll = (s_r * f2_l - s_l * f2_r + s_l * s_r * (rho_r * u_r - rho_l * u_l)) / sr_sl;
        double f3_hll = (s_r * f3_l - s_l * f3_r + s_l * s_r * (e_r - e_l)) / sr_sl;
        f1 = (1.0 - alpha) * f1 + alpha * f1_hll;
        f2 = (1.0 - alpha) * f2 + alpha * f2_hll;
        f3 = (1.0 - alpha) * f3 + alpha * f3_hll;
    }
}

static double area_mach(double M, double gam) {
    double gm1 = gam - 1.0;
    double gp1 = gam + 1.0;
    double term = (2.0 / gp1) * (1.0 + 0.5 * gm1 * M * M);
    return (1.0 / M) * std::pow(term, gp1 / (2.0 * gm1));
}

static double solve_mach(double ar, double gam, bool supersonic) {
    if (ar <= 1.000001) return 1.0;
    double lo = supersonic ? 1.00001 : 1e-4;
    double hi = supersonic ? 25.0 : 0.99999;
    for (int i = 0; i < 65; ++i) {
        double mid = 0.5 * (lo + hi);
        double f = area_mach(mid, gam) - ar;
        if (supersonic) {
            if (f < 0.0) lo = mid; else hi = mid;
        } else {
            if (f > 0.0) lo = mid; else hi = mid;
        }
    }
    return 0.5 * (lo + hi);
}

static int run_nozzle_solver(
    int n, double dx_nd, double gamma, double pb_nd,
    int flux_id, int limiter_id, double cfl, int max_iter, double tol,
    const double* area, double* out_rho, double* out_u, double* out_p,
    int& out_iters, int& out_converged, double& out_residual,
    double* out_res_hist = nullptr, int max_hist_pts = 0, int* out_hist_len = nullptr
) {
    if (n < 15 || !area || !out_rho || !out_u || !out_p) return -1;

    const double gm1 = gamma - 1.0;
    const bool subsonic_outflow = (pb_nd > 0.0);

    int throat_idx = 0;
    double min_a = 1e9;
    for (int i = 0; i < n; ++i) {
        if (area[i] < min_a) { min_a = area[i]; throat_idx = i; }
    }

    std::vector<double> a_int(n + 1), da_cell(n);
    a_int[0] = area[0];
    for (int i = 1; i < n; ++i) a_int[i] = 0.5 * (area[i - 1] + area[i]);
    a_int[n] = std::max(1.0, area[n - 1] + 0.5 * (area[n - 1] - area[n - 2]));
    for (int i = 0; i < n; ++i) da_cell[i] = a_int[i + 1] - a_int[i];

    std::vector<double> u1(n), u2(n), u3(n), u1s(n), u2s(n), u3s(n);
    std::vector<double> u1n(n), u2n(n), u3n(n);
    std::vector<double> k1_1(n), k1_2(n), k1_3(n), k2_1(n), k2_2(n), k2_3(n);
    std::vector<double> r_pad(n + 4), u_pad(n + 4), p_pad(n + 4);
    std::vector<double> sr(n + 2), su(n + 2), sp(n + 2);
    std::vector<double> f1(n + 1), f2(n + 1), f3(n + 1);

    for (int i = 0; i < n; ++i) {
        double m0 = solve_mach(area[i], gamma, i > throat_idx);
        if (subsonic_outflow && i > throat_idx) {
            double frac = static_cast<double>(i - throat_idx) / static_cast<double>(n - throat_idx);
            m0 = std::max(0.25, m0 * (1.0 - 0.65 * frac));
        }
        double t0 = 1.0 / (1.0 + 0.5 * gm1 * m0 * m0);
        double p0 = std::pow(t0, gamma / gm1);
        double rho0 = p0 / t0;
        double vel0 = m0 * std::sqrt(gamma * t0);
        u1[i] = rho0;
        u2[i] = rho0 * vel0;
        u3[i] = p0 / gm1 + 0.5 * rho0 * vel0 * vel0;
    }

    const double h0_nd = gamma / gm1;
    const double a_quad = 1.0 / gm1 + 2.0 / (gm1 * gm1);

    auto eval_rhs = [&](const std::vector<double>& c1, const std::vector<double>& c2,
                        const std::vector<double>& c3, std::vector<double>& d1,
                        std::vector<double>& d2, std::vector<double>& d3) {
        for (int i = 0; i < n; ++i) {
            double ri = std::max(1e-9, c1[i]);
            double ui = c2[i] / ri;
            double pi = std::max(1e-9, gm1 * (c3[i] - 0.5 * ri * ui * ui));
            r_pad[i + 2] = ri;
            u_pad[i + 2] = ui;
            p_pad[i + 2] = pi;
        }

        double a0 = std::sqrt(gamma * p_pad[2] / r_pad[2]);
        double r_minus = u_pad[2] - (2.0 * a0 / gm1);
        double b_quad = (2.0 * r_minus) / gm1;
        double c_quad = 0.5 * r_minus * r_minus - h0_nd;
        double disc = std::max(0.0, b_quad * b_quad - 4.0 * a_quad * c_quad);
        double a_in = std::max(0.2, (-b_quad + std::sqrt(disc)) / (2.0 * a_quad));
        double u_in = std::max(0.005 * a_in, std::min(0.92 * a_in, r_minus + (2.0 * a_in / gm1)));
        double t_in = std::min(1.0, std::max(0.4, (a_in * a_in) / gamma));
        double p_in = std::pow(t_in, gamma / gm1);
        double r_in = p_in / t_in;
        r_pad[0] = r_pad[1] = r_in;
        u_pad[0] = u_pad[1] = u_in;
        p_pad[0] = p_pad[1] = p_in;

        if (subsonic_outflow) {
            double ae = std::sqrt(gamma * p_pad[n + 1] / r_pad[n + 1]);
            double pout = std::max(1e-5, pb_nd);
            double se = p_pad[n + 1] / std::pow(r_pad[n + 1], gamma);
            double rout = std::max(1e-6, std::pow(pout / se, 1.0 / gamma));
            double uout = std::max(0.02, u_pad[n + 1] + (p_pad[n + 1] - pout) / (r_pad[n + 1] * ae));
            r_pad[n + 2] = r_pad[n + 3] = rout;
            u_pad[n + 2] = u_pad[n + 3] = uout;
            p_pad[n + 2] = p_pad[n + 3] = pout;
        } else {
            r_pad[n + 2] = r_pad[n + 3] = r_pad[n + 1];
            u_pad[n + 2] = u_pad[n + 3] = u_pad[n + 1];
            p_pad[n + 2] = p_pad[n + 3] = p_pad[n + 1];
        }

        for (int c = 1; c <= n + 2; ++c) {
            double dp_sw = std::abs(p_pad[c + 1] - 2.0 * p_pad[c] + p_pad[c - 1]) /
                           std::max(p_pad[c + 1] + 2.0 * p_pad[c] + p_pad[c - 1], 1e-12);
            double damp = std::clamp(1.0 - 8.0 * std::max(0.0, dp_sw - 0.03), 0.0, 1.0);
            sr[c - 1] = damp * tvd_limiter(r_pad[c] - r_pad[c - 1], r_pad[c + 1] - r_pad[c], limiter_id);
            su[c - 1] = damp * tvd_limiter(u_pad[c] - u_pad[c - 1], u_pad[c + 1] - u_pad[c], limiter_id);
            sp[c - 1] = damp * tvd_limiter(p_pad[c] - p_pad[c - 1], p_pad[c + 1] - p_pad[c], limiter_id);
        }

        for (int j = 0; j <= n; ++j) {
            double rl = std::max(1e-9, r_pad[j + 1] + 0.5 * sr[j]);
            double ul = u_pad[j + 1] + 0.5 * su[j];
            double pl = std::max(1e-9, p_pad[j + 1] + 0.5 * sp[j]);
            double rr = std::max(1e-9, r_pad[j + 2] - 0.5 * sr[j + 1]);
            double ur = u_pad[j + 2] - 0.5 * su[j + 1];
            double pr = std::max(1e-9, p_pad[j + 2] - 0.5 * sp[j + 1]);
            riemann_flux(rl, ul, pl, rr, ur, pr, gamma, flux_id, f1[j], f2[j], f3[j]);
        }

        for (int i = 0; i < n; ++i) {
            double inv_vol = 1.0 / (area[i] * dx_nd);
            double pi = p_pad[i + 2];
            d1[i] = -(f1[i + 1] * a_int[i + 1] - f1[i] * a_int[i]) * inv_vol;
            d2[i] = -(f2[i + 1] * a_int[i + 1] - f2[i] * a_int[i]) * inv_vol + pi * da_cell[i] * inv_vol;
            d3[i] = -(f3[i + 1] * a_int[i + 1] - f3[i] * a_int[i]) * inv_vol;
        }
    };

    out_converged = 0;
    out_iters = 0;
    out_residual = 1.0;

    for (int it = 1; it <= max_iter; ++it) {
        double max_spd = 1e-6;
        for (int i = 0; i < n; ++i) {
            double ri = std::max(1e-9, u1[i]);
            double ui = u2[i] / ri;
            double pi = std::max(1e-9, gm1 * (u3[i] - 0.5 * ri * ui * ui));
            double ai = std::sqrt(gamma * pi / ri);
            max_spd = std::max(max_spd, std::abs(ui) + ai);
        }
        double dt = cfl * dx_nd / max_spd;

        eval_rhs(u1, u2, u3, k1_1, k1_2, k1_3);
        for (int i = 0; i < n; ++i) {
            u1s[i] = u1[i] + dt * k1_1[i];
            u2s[i] = u2[i] + dt * k1_2[i];
            u3s[i] = u3[i] + dt * k1_3[i];
        }
        eval_rhs(u1s, u2s, u3s, k2_1, k2_2, k2_3);

        double max_diff = 0.0;
        for (int i = 0; i < n; ++i) {
            u1n[i] = 0.5 * (u1[i] + u1s[i] + dt * k2_1[i]);
            u2n[i] = 0.5 * (u2[i] + u2s[i] + dt * k2_2[i]);
            u3n[i] = 0.5 * (u3[i] + u3s[i] + dt * k2_3[i]);
            max_diff = std::max(max_diff, std::abs(u2n[i] - u2[i]) / std::max(std::abs(u2[i]), 1e-4));
        }
        u1.swap(u1n);
        u2.swap(u2n);
        u3.swap(u3n);

        out_iters = it;
        out_residual = max_diff;
        if (out_res_hist && out_hist_len && (*out_hist_len < max_hist_pts)) {
            if (it == 1 || (it % 50 == 0) || (it > 150 && max_diff < tol)) {
                out_res_hist[(*out_hist_len)++] = max_diff;
            }
        }
        if (it > 150 && max_diff < tol) {
            out_converged = 1;
            break;
        }
    }

    for (int i = 0; i < n; ++i) {
        double ri = std::max(1e-9, u1[i]);
        double ui = u2[i] / ri;
        double pi = std::max(1e-9, gm1 * (u3[i] - 0.5 * ri * ui * ui));
        out_rho[i] = ri;
        out_u[i] = ui;
        out_p[i] = pi;
    }
    return 0;
}

}  // namespace fvm_core

extern "C" FVM_API int solve_quasi1d_fvm_c(
    int n, double dx_nd, double gamma, double pb_nd,
    int flux_id, int limiter_id, double cfl, int max_iter, double tol,
    const double* area_ratio, double* out_rho, double* out_u, double* out_p,
    int* out_iters, int* out_converged, double* out_residual,
    double* out_res_hist, int max_hist_pts, int* out_hist_len
) {
    int iters = 0, conv = 0, hist_len = 0;
    double res = 1.0;
    int rc = fvm_core::run_nozzle_solver(
        n, dx_nd, gamma, pb_nd, flux_id, limiter_id, cfl, max_iter, tol,
        area_ratio, out_rho, out_u, out_p, iters, conv, res,
        out_res_hist, max_hist_pts, &hist_len
    );
    if (out_iters) *out_iters = iters;
    if (out_converged) *out_converged = conv;
    if (out_residual) *out_residual = res;
    if (out_hist_len) *out_hist_len = hist_len;
    return rc;
}

#ifndef NOZZLE_FVM_BUILD_DLL
int main(int argc, char* argv[]) {
    double gamma = 1.21, pc_mpa = 10.0, pa_kpa = 101.325, eps = 12.0, cfl = 0.50, tol = 1e-6;
    int nx = 161, max_iter = 15000, flux_id = 0, limiter_id = 2;
    bool json_out = false;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--gamma" && i + 1 < argc) gamma = std::stod(argv[++i]);
        else if (a == "--pc-mpa" && i + 1 < argc) pc_mpa = std::stod(argv[++i]);
        else if (a == "--pa-kpa" && i + 1 < argc) pa_kpa = std::stod(argv[++i]);
        else if (a == "--eps" && i + 1 < argc) eps = std::stod(argv[++i]);
        else if (a == "--nx" && i + 1 < argc) nx = std::stoi(argv[++i]);
        else if (a == "--max-iter" && i + 1 < argc) max_iter = std::stoi(argv[++i]);
        else if (a == "--flux" && i + 1 < argc) {
            std::string f = argv[++i];
            flux_id = (f == "ausm_plus") ? 1 : ((f == "rusanov") ? 2 : 0);
        } else if (a == "--limiter" && i + 1 < argc) {
            std::string l = argv[++i];
            limiter_id = (l == "first_order") ? 0 : ((l == "minmod") ? 1 : ((l == "superbee") ? 3 : 2));
        } else if (a == "--json") {
            json_out = true;
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    std::vector<double> area(nx), rho(nx), u(nx), p(nx);
    const double dx_nd = 3.5 / static_cast<double>(nx - 1);
    for (int i = 0; i < nx; ++i) {
        double x = i * dx_nd;
        if (x <= 1.0) area[i] = 1.0 + 2.2 * (1.0 - x) * (1.0 - x);
        else area[i] = 1.0 + (eps - 1.0) * std::pow((x - 1.0) / 2.5, 1.4);
    }

    double m_sup = fvm_core::solve_mach(area.back(), gamma, true);
    double gm1 = gamma - 1.0;
    double pc_pa = pc_mpa * 1e6;
    double pa_pa = std::max(1.0, pa_kpa * 1e3);
    double pe_sup = pc_pa * std::pow(1.0 + 0.5 * gm1 * m_sup * m_sup, -gamma / gm1);
    double pe_shk = pe_sup * (1.0 + (2.0 * gamma / (gamma + 1.0)) * (m_sup * m_sup - 1.0));
    double pb_nd = (pa_pa > pe_shk) ? (pa_pa / pc_pa) : -1.0;

    int iters = 0, conv = 0;
    double res = 1.0;
    fvm_core::run_nozzle_solver(
        nx, dx_nd, gamma, pb_nd, flux_id, limiter_id, cfl, max_iter, tol,
        area.data(), rho.data(), u.data(), p.data(), iters, conv, res
    );
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    double exit_mach = u.back() / std::sqrt(gamma * p.back() / rho.back());
    double exit_p_kpa = p.back() * pc_mpa * 1000.0;

    if (json_out) {
        std::cout << std::fixed << std::setprecision(4)
                  << "{\"solver\":\"fvm_cpp20\",\"nx\":" << nx
                  << ",\"iterations\":" << iters
                  << ",\"converged\":" << (conv ? "true" : "false")
                  << ",\"exit_mach\":" << exit_mach
                  << ",\"exit_pressure_kpa\":" << exit_p_kpa
                  << ",\"solve_ms\":" << ms << "}\n";
    } else {
        std::cout << "[C++20 FVM] nx=" << nx << " iters=" << iters
                  << " conv=" << conv << " Mex=" << exit_mach
                  << " Pex=" << exit_p_kpa << " kPa in " << ms << " ms\n";
    }
    return 0;
}
#endif
