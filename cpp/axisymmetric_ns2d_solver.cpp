#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

#if defined(_WIN32) && defined(NOZZLE_FVM2D_BUILD_DLL)
#define FVM2D_EXPORT extern "C" __declspec(dllexport)
#else
#define FVM2D_EXPORT extern "C"
#endif

namespace ns2d_core {

constexpr double kPi = 3.14159265358979323846;

struct Prim {
    double rho;
    double u;
    double v;
    double p;
};

struct Cons {
    double r;
    double ru;
    double rv;
    double rE;
};

struct Flux4 {
    double f0;
    double f1;
    double f2;
    double f3;
};

inline Cons prim_to_cons(const Prim& w, double gm1) {
    const double ke = 0.5 * (w.u * w.u + w.v * w.v);
    return {w.rho, w.rho * w.u, w.rho * w.v, w.p / gm1 + w.rho * ke};
}

inline Prim cons_to_prim(const Cons& u, double gm1, double r_floor = 1e-8, double p_floor = 1.0) {
    const double rho = std::max(r_floor, u.r);
    const double vx = u.ru / rho;
    const double vr = u.rv / rho;
    const double ke = 0.5 * (vx * vx + vr * vr);
    const double eint = std::max(p_floor / gm1, u.rE - rho * ke);
    const double p = std::max(p_floor, gm1 * eint);
    return {rho, vx, vr, p};
}

inline double limiter_slope(double dl, double dr, int limiter_id) {
    if (limiter_id == 0 || dl * dr <= 0.0) {
        return 0.0;
    }
    const double adl = std::abs(dl);
    const double adr = std::abs(dr);
    const double sgn = (dl > 0.0) ? 1.0 : -1.0;
    if (limiter_id == 1) {  // minmod
        return sgn * std::min(adl, adr);
    }
    if (limiter_id == 3) {  // mc
        return sgn * std::min({2.0 * adl, 2.0 * adr, 0.5 * (adl + adr)});
    }
    // van_leer default (id == 2)
    return (2.0 * dl * dr) / (dl + dr + std::copysign(1e-30, dl + dr));
}

inline double sutherland_mu(double T, double mu_ref, double T_ref = 273.15, double S = 110.4) {
    const double Tc = std::max(50.0, T);
    const double tr = Tc / T_ref;
    return mu_ref * tr * std::sqrt(tr) * ((T_ref + S) / (Tc + S));
}

// Rotated 2D HLLC+ Riemann flux with carbuncle-free shock blend (0 at shear/contact waves)
inline Flux4 hllc_rotated_flux(
    const Prim& wl,
    const Prim& wr,
    double nx,
    double nr,
    double gamma
) {
    const double gm1 = gamma - 1.0;

    // Face normal and tangential components
    const double un_l = wl.u * nx + wl.v * nr;
    const double ut_l = -wl.u * nr + wl.v * nx;
    const double un_r = wr.u * nx + wr.v * nr;
    const double ut_r = -wr.u * nr + wr.v * nx;

    const double rho_l = std::max(1e-8, wl.rho);
    const double rho_r = std::max(1e-8, wr.rho);
    const double p_l = std::max(1.0, wl.p);
    const double p_r = std::max(1.0, wr.p);

    const double a_l = std::sqrt(gamma * p_l / rho_l);
    const double a_r = std::sqrt(gamma * p_r / rho_r);

    const double ke_l = 0.5 * (wl.u * wl.u + wl.v * wl.v);
    const double ke_r = 0.5 * (wr.u * wr.u + wr.v * wr.v);
    const double E_l = p_l / (gm1 * rho_l) + ke_l;
    const double E_r = p_r / (gm1 * rho_r) + ke_r;
    const double H_l = E_l + p_l / rho_l;
    const double H_r = E_r + p_r / rho_r;

    const double sq_l = std::sqrt(rho_l);
    const double sq_r = std::sqrt(rho_r);
    const double inv_sq = 1.0 / (sq_l + sq_r);
    const double un_roe = (sq_l * un_l + sq_r * un_r) * inv_sq;
    const double ut_roe = (sq_l * ut_l + sq_r * ut_r) * inv_sq;
    const double H_roe = (sq_l * H_l + sq_r * H_r) * inv_sq;
    const double a_roe = std::sqrt(
        std::max(1e-9, gm1 * (H_roe - 0.5 * (un_roe * un_roe + ut_roe * ut_roe)))
    );

    const double s_l = std::min(un_l - a_l, un_roe - a_roe);
    const double s_r = std::max(un_r + a_r, un_roe + a_roe);

    // Pure normal-frame fluxes
    const double fl0 = rho_l * un_l;
    const double fln = rho_l * un_l * un_l + p_l;
    const double flt = rho_l * un_l * ut_l;
    const double fl3 = (rho_l * E_l + p_l) * un_l;

    const double fr0 = rho_r * un_r;
    const double frn = rho_r * un_r * un_r + p_r;
    const double frt = rho_r * un_r * ut_r;
    const double fr3 = (rho_r * E_r + p_r) * un_r;

    double f0 = 0.0, fn = 0.0, ft = 0.0, f3 = 0.0;

    if (s_l >= 0.0) {
        f0 = fl0;
        fn = fln;
        ft = flt;
        f3 = fl3;
    } else if (s_r <= 0.0) {
        f0 = fr0;
        fn = frn;
        ft = frt;
        f3 = fr3;
    } else {
        double denom = rho_l * (s_l - un_l) - rho_r * (s_r - un_r);
        if (std::abs(denom) < 1e-12) {
            denom = std::copysign(1e-12, denom + 1e-30);
        }
        const double s_star =
            (p_r - p_l + rho_l * un_l * (s_l - un_l) - rho_r * un_r * (s_r - un_r)) / denom;

        if (s_star >= 0.0) {
            double dsl = s_l - s_star;
            if (std::abs(dsl) < 1e-12) dsl = -1e-12;
            const double fac = rho_l * (s_l - un_l) / dsl;
            const double u_star_e = fac * (E_l + (s_star - un_l) * (s_star + p_l / (rho_l * (s_l - un_l) - 1e-15)));
            f0 = fl0 + s_l * (fac - rho_l);
            fn = fln + s_l * (fac * s_star - rho_l * un_l);
            ft = flt + s_l * (fac * ut_l - rho_l * ut_l);
            f3 = fl3 + s_l * (u_star_e - rho_l * E_l);
        } else {
            double dsr = s_r - s_star;
            if (std::abs(dsr) < 1e-12) dsr = 1e-12;
            const double fac = rho_r * (s_r - un_r) / dsr;
            const double u_star_e = fac * (E_r + (s_star - un_r) * (s_star + p_r / (rho_r * (s_r - un_r) + 1e-15)));
            f0 = fr0 + s_r * (fac - rho_r);
            fn = frn + s_r * (fac * s_star - rho_r * un_r);
            ft = frt + s_r * (fac * ut_r - rho_r * ut_r);
            f3 = fr3 + s_r * (u_star_e - rho_r * E_r);
        }

        // 2D carbuncle & stationary shock HLL stabilizer active strictly across strong pressure jumps
        const double p_jump = std::abs(p_r - p_l) / (p_r + p_l + 1e-12);
        if (p_jump > 0.12) {
            const double alpha = std::min(0.40, 0.55 * p_jump * p_jump);
            const double sr_sl = std::max(1e-9, s_r - s_l);
            const double fhll0 = (s_r * fl0 - s_l * fr0 + s_l * s_r * (rho_r - rho_l)) / sr_sl;
            const double fhlln =
                (s_r * fln - s_l * frn + s_l * s_r * (rho_r * un_r - rho_l * un_l)) / sr_sl;
            const double fhllt =
                (s_r * flt - s_l * frt + s_l * s_r * (rho_r * ut_r - rho_l * ut_l)) / sr_sl;
            const double fhll3 =
                (s_r * fl3 - s_l * fr3 + s_l * s_r * (rho_r * E_r - rho_l * E_l)) / sr_sl;
            f0 = (1.0 - alpha) * f0 + alpha * fhll0;
            fn = (1.0 - alpha) * fn + alpha * fhlln;
            ft = (1.0 - alpha) * ft + alpha * fhllt;
            f3 = (1.0 - alpha) * f3 + alpha * fhll3;
        }
    }

    // Rotate momentum flux back from (normal, tangent) to (x, r)
    const double fx = fn * nx - ft * nr;
    const double fr = fn * nr + ft * nx;
    return {f0, fx, fr, f3};
}

// Curvilinear structured block geometry & precomputed Green-Gauss axisymmetric wedge metrics
struct GridBlock {
    int nx = 0;
    int nr = 0;
    std::vector<double> xv;       // (nx+1)*(nr+1)
    std::vector<double> rv;       // (nx+1)*(nr+1)
    std::vector<double> xc;       // nx*nr centroid x
    std::vector<double> rc;       // nx*nr centroid r
    std::vector<double> area;     // nx*nr planar quad area
    std::vector<double> vol;      // nx*nr 1-radian wedge volume = rc * area
    std::vector<double> h_min;    // nx*nr minimum characteristic cell width
    std::vector<double> dist_w;   // nx*nr distance to nearest wall/slip-layer
    std::vector<double> local_rw; // nx*nr local nozzle outer wall radius r(x)
    // I-faces: (nx+1) * nr, index i * nr + j, normal points i-1 -> i
    std::vector<double> i_nx, i_nr, i_sf, i_ds;
    // J-faces: nx * (nr+1), index i * (nr+1) + j, normal points j-1 -> j
    std::vector<double> j_nx, j_nr, j_sf, j_ds;

    void build(int nx_, int nr_, const double* xv_in, const double* rv_in, double wall_r_ref) {
        nx = nx_;
        nr = nr_;
        const int nv = (nx + 1) * (nr + 1);
        xv.assign(xv_in, xv_in + nv);
        rv.assign(rv_in, rv_in + nv);

        const int nc = nx * nr;
        xc.resize(nc);
        rc.resize(nc);
        area.resize(nc);
        vol.resize(nc);
        h_min.resize(nc);
        dist_w.resize(nc);
        local_rw.resize(nc);

        auto vidx = [&](int i, int j) { return i * (nr + 1) + j; };

        for (int i = 0; i < nx; ++i) {
            const double rw_local = 0.5 * (rv[vidx(i, nr)] + rv[vidx(i + 1, nr)]);
            for (int j = 0; j < nr; ++j) {
                const double x0 = xv[vidx(i, j)];
                const double r0 = rv[vidx(i, j)];
                const double x1 = xv[vidx(i + 1, j)];
                const double r1 = rv[vidx(i + 1, j)];
                const double x2 = xv[vidx(i + 1, j + 1)];
                const double r2 = rv[vidx(i + 1, j + 1)];
                const double x3 = xv[vidx(i, j + 1)];
                const double r3 = rv[vidx(i, j + 1)];

                const double a1 = 0.5 * ((x1 - x0) * (r2 - r0) - (r1 - r0) * (x2 - x0));
                const double a2 = 0.5 * ((x2 - x0) * (r3 - r0) - (r2 - r0) * (x3 - x0));
                const double a = std::max(1e-18, a1 + a2);
                const double xg = (a1 * (x0 + x1 + x2) / 3.0 + a2 * (x0 + x2 + x3) / 3.0) / a;
                const double rg =
                    std::max(1e-9, (a1 * (r0 + r1 + r2) / 3.0 + a2 * (r0 + r2 + r3) / 3.0) / a);

                const int c = i * nr + j;
                xc[c] = xg;
                rc[c] = rg;
                area[c] = a;
                vol[c] = rg * a;
                local_rw[c] = std::max(1e-6, rw_local);

                const double d_i = std::hypot(0.5 * (x1 + x2 - x0 - x3), 0.5 * (r1 + r2 - r0 - r3));
                const double d_j = std::hypot(0.5 * (x3 + x2 - x0 - x1), 0.5 * (r3 + r2 - r0 - r1));
                h_min[c] = std::max(1e-7, std::min(d_i, d_j));

                const double xw = 0.5 * (xv[vidx(i, nr)] + xv[vidx(i + 1, nr)]);
                const double rw = rw_local;
                dist_w[c] = std::min(std::hypot(xg - xw, rg - rw), 0.5 * wall_r_ref);
            }
        }

        const int nif = (nx + 1) * nr;
        i_nx.resize(nif);
        i_nr.resize(nif);
        i_sf.resize(nif);
        i_ds.resize(nif);
        for (int i = 0; i <= nx; ++i) {
            for (int j = 0; j < nr; ++j) {
                const double xa = xv[vidx(i, j)];
                const double ra = rv[vidx(i, j)];
                const double xb = xv[vidx(i, j + 1)];
                const double rb = rv[vidx(i, j + 1)];
                const double dx = xb - xa;
                const double dr = rb - ra;
                const double ds = std::max(1e-12, std::hypot(dx, dr));
                const double rf = 0.5 * (ra + rb);
                const int f = i * nr + j;
                i_nx[f] = dr / ds;
                i_nr[f] = -dx / ds;
                i_ds[f] = ds;
                i_sf[f] = rf * ds;
            }
        }

        const int njf = nx * (nr + 1);
        j_nx.resize(njf);
        j_nr.resize(njf);
        j_sf.resize(njf);
        j_ds.resize(njf);
        for (int i = 0; i < nx; ++i) {
            for (int j = 0; j <= nr; ++j) {
                const double xa = xv[vidx(i, j)];
                const double ra = rv[vidx(i, j)];
                const double xb = xv[vidx(i + 1, j)];
                const double rb = rv[vidx(i + 1, j)];
                const double dx = xb - xa;
                const double dr = rb - ra;
                const double ds = std::max(1e-12, std::hypot(dx, dr));
                const double rf = 0.5 * (ra + rb);
                const int f = i * (nr + 1) + j;
                j_nx[f] = -dr / ds;
                j_nr[f] = dx / ds;
                j_ds[f] = ds;
                j_sf[f] = rf * ds;
            }
        }
    }
};

struct BlockState {
    const GridBlock* g = nullptr;
    std::vector<Cons> u;
    std::vector<Cons> u0;
    std::vector<Prim> w;
    std::vector<Cons> res;
    std::vector<double> gradx_u, gradr_u;
    std::vector<double> gradx_v, gradr_v;
    std::vector<double> gradx_T, gradr_T;
    std::vector<double> mu_eff;
    std::vector<Flux4> fi;
    std::vector<Flux4> fj;

    void init(const GridBlock& grid, const Prim* w_init, double gm1) {
        g = &grid;
        const int nc = grid.nx * grid.nr;
        u.resize(nc);
        u0.resize(nc);
        w.assign(w_init, w_init + nc);
        res.resize(nc);
        gradx_u.assign(nc, 0.0);
        gradr_u.assign(nc, 0.0);
        gradx_v.assign(nc, 0.0);
        gradr_v.assign(nc, 0.0);
        gradx_T.assign(nc, 0.0);
        gradr_T.assign(nc, 0.0);
        mu_eff.assign(nc, 0.0);
        fi.resize((grid.nx + 1) * grid.nr);
        fj.resize(grid.nx * (grid.nr + 1));
        for (int c = 0; c < nc; ++c) {
            u[c] = prim_to_cons(w[c], gm1);
            u0[c] = u[c];
        }
    }
};

// Riemann inflow ghost at subsonic nozzle entrance preserving total enthalpy + entropy
inline Prim inflow_riemann_state(
    const Prim& win,
    double nx_wall,
    double nr_wall,
    double pc_pa,
    double tc_k,
    double gamma,
    double r_gas,
    double inflow_slope
) {
    const double gm1 = gamma - 1.0;
    // Mirror outward normal (-nx_wall, -nr_wall) points INTO interior
    const double rho_in = std::max(1e-8, win.rho);
    const double p_in = std::max(1.0, win.p);
    const double a_in = std::sqrt(gamma * p_in / rho_in);
    const double u_mag_in = std::hypot(win.u, win.v);
    const double m_in = u_mag_in / std::max(1e-6, a_in);

    // Subsonic Riemann invariant R- extrapolated from interior; total pressure & entropy clamped
    double m_b = std::clamp(m_in, 0.0, 0.95);
    const double fac = 1.0 + 0.5 * gm1 * m_b * m_b;
    const double T_b = tc_k / fac;
    const double p_b = pc_pa * std::pow(fac, -gamma / gm1);
    const double rho_b = p_b / (r_gas * T_b);
    const double a_b = std::sqrt(gamma * r_gas * T_b);
    const double vel_b = m_b * a_b;
    const double inv_s = 1.0 / std::sqrt(1.0 + inflow_slope * inflow_slope);
    (void)nx_wall;
    (void)nr_wall;
    return {rho_b, vel_b * inv_s, vel_b * inflow_slope * inv_s, p_b};
}

// Characteristic outflow state (supersonic extrapolate; subsonic impose p_back)
inline Prim outflow_state(const Prim& win, double nx, double nr, double p_back, double gamma) {
    const double rho = std::max(1e-8, win.rho);
    const double p = std::max(1.0, win.p);
    const double a = std::sqrt(gamma * p / rho);
    const double un = win.u * nx + win.v * nr;
    if (un >= a) {
        return win;
    }
    if (un >= 0.0) {
        const double p_b = std::max(1.0, p_back);
        const double rho_b = std::max(1e-8, rho * std::pow(p_b / p, 1.0 / gamma));
        const double a_b = std::sqrt(gamma * p_b / rho_b);
        const double un_b = std::max(0.0, un + 2.0 * (a - a_b) / (gamma - 1.0));
        const double ut = -win.u * nr + win.v * nx;
        return {rho_b, un_b * nx - ut * nr, un_b * nr + ut * nx, p_b};
    }
    // Reverse entrainment from ambient at subsonic exit
    const double T_amb = 300.0;
    const double r_spec_approx = p / (rho * std::max(50.0, p / (rho * 400.0)));
    (void)r_spec_approx;
    const double rho_b = std::max(1e-8, rho * (p_back / p));
    (void)T_amb;
    return {rho_b, 0.5 * win.u, 0.5 * win.v, std::max(1.0, p_back)};
}

// Evaluate viscous stress & heat flux gradients at all cell centers via Green-Gauss
inline void compute_viscous_gradients_and_mu(
    BlockState& bs,
    double r_gas,
    double mu_ref,
    int viscous_mode,
    bool is_plume_block,
    double wall_r_ref,
    double pc_pa,
    double tc_k
) {
    if (viscous_mode <= 0) {
        return;
    }
    const GridBlock& g = *bs.g;
    const int nx = g.nx;
    const int nr = g.nr;

#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < nr; ++j) {
            const int c = i * nr + j;
            const auto get_w_nbr = [&](int ii, int jj, bool wall_top, bool sym_bot) -> Prim {
                if (ii < 0) return bs.w[0 * nr + j];
                if (ii >= nx) return bs.w[(nx - 1) * nr + j];
                if (jj < 0) {
                    const Prim& w0 = bs.w[i * nr];
                    return sym_bot ? Prim{w0.rho, w0.u, -w0.v, w0.p} : w0;
                }
                if (jj >= nr) {
                    const Prim& wt = bs.w[i * nr + (nr - 1)];
                    if (wall_top) {
                        return Prim{wt.rho, -wt.u, -wt.v, wt.p};
                    }
                    return wt;
                }
                return bs.w[ii * nr + jj];
            };

            const Prim wW = get_w_nbr(i - 1, j, false, false);
            const Prim wE = get_w_nbr(i + 1, j, false, false);
            const Prim wS = get_w_nbr(i, j - 1, false, true);
            const Prim wN = get_w_nbr(i, j + 1, !is_plume_block, false);

            const double TW = wW.p / (wW.rho * r_gas);
            const double TE = wE.p / (wE.rho * r_gas);
            const double TS = wS.p / (wS.rho * r_gas);
            const double TN = wN.p / (wN.rho * r_gas);

            // Finite-volume metric dx/dxi, dr/dxi, dx/deta, dr/deta at cell center
            const double dx_xi =
                (i + 1 < nx ? g.xc[(i + 1) * nr + j] : g.xc[c]) -
                (i > 0 ? g.xc[(i - 1) * nr + j] : g.xc[c]);
            const double dr_xi =
                (i + 1 < nx ? g.rc[(i + 1) * nr + j] : g.rc[c]) -
                (i > 0 ? g.rc[(i - 1) * nr + j] : g.rc[c]);

            double dx_eta, dr_eta;
            if (j > 0 && j + 1 < nr) {
                dx_eta = g.xc[i * nr + (j + 1)] - g.xc[i * nr + (j - 1)];
                dr_eta = g.rc[i * nr + (j + 1)] - g.rc[i * nr + (j - 1)];
            } else if (j == 0 && nr > 1) {
                dx_eta = 2.0 * (g.xc[i * nr + 1] - g.xc[c]);
                dr_eta = 2.0 * std::max(1e-7, g.rc[c]);
            } else {
                const int v0 = i * (nr + 1) + nr;
                const int v1 = (i + 1) * (nr + 1) + nr;
                const double xw = 0.5 * (g.xv[v0] + g.xv[v1]);
                const double rw = 0.5 * (g.rv[v0] + g.rv[v1]);
                dx_eta = 2.0 * (xw - g.xc[c]);
                dr_eta = 2.0 * (rw - g.rc[c]);
            }

            double det = dx_xi * dr_eta - dr_xi * dx_eta;
            if (std::abs(det) < 1e-18) det = std::copysign(1e-18, det + 1e-30);
            const double inv_det = 1.0 / det;

            auto eval_grad = [&](double qW, double qE, double qS, double qN, double& gx, double& gr) {
                const double d_xi = qE - qW;
                const double d_eta = qN - qS;
                gx = (d_xi * dr_eta - d_eta * dr_xi) * inv_det;
                gr = (-d_xi * dx_eta + d_eta * dx_xi) * inv_det;
            };

            eval_grad(wW.u, wE.u, wS.u, wN.u, bs.gradx_u[c], bs.gradr_u[c]);
            eval_grad(wW.v, wE.v, wS.v, wN.v, bs.gradx_v[c], bs.gradr_v[c]);
            eval_grad(TW, TE, TS, TN, bs.gradx_T[c], bs.gradr_T[c]);

            const double Tc = bs.w[c].p / (bs.w[c].rho * r_gas);
            const double mu_lam = sutherland_mu(Tc, mu_ref);
            double mu_t = 0.0;
            if (viscous_mode >= 2) {
                // Prandtl / Baldwin-Lomax inner-outer mixing length calibrated to local duct radius
                const double omega = std::abs(bs.gradx_v[c] - bs.gradr_u[c]);
                const double l_mix = is_plume_block
                                         ? (0.015 * wall_r_ref)
                                         : std::min(0.41 * g.dist_w[c], 0.015 * g.local_rw[c]);
                mu_t = bs.w[c].rho * l_mix * l_mix * omega;
                // Cap eddy viscosity safely by chamber scale
                const double rho_ch = pc_pa / (r_gas * tc_k);
                mu_t = std::min(mu_t, 800.0 * mu_lam + 0.004 * rho_ch * wall_r_ref * std::sqrt(r_gas * tc_k));
            }
            bs.mu_eff[c] = mu_lam + mu_t;
        }
    }
}

struct Params2D {
    double gamma = 1.21;
    double r_gas = 399.73;
    double tc_k = 3560.0;
    double pc_pa = 10.0e6;
    double pa_pa = 101325.0;
    double mu_ref = 1.716e-5;
    double prandtl = 0.72;
    double cfl = 0.42;
    double r_wall_exit = 0.5;
    double inflow_slope = 0.0;
    int max_iters = 2600;
    double tol = 2e-5;
    int limiter_id = 2;       // 0: 1st, 1: minmod, 2: van_leer, 3: mc
    int viscous_mode = 2;     // 0: euler, 1: laminar, 2: turbulent
    bool local_time_step = true;
    int use_isothermal_wall = 0;
    double wall_temp_k = 900.0;
};

struct SolveOutput2D {
    int iterations = 0;
    int converged = 0;
    double mass_flow_throat = 0.0;
    double mass_flow_exit = 0.0;
    double final_residual = 0.0;
    double solve_ms = 0.0;
    std::vector<double> res_hist;
};

inline void reconstruct_and_flux_block(
    BlockState& bs,
    const BlockState* plume_ptr,
    bool is_plume,
    const BlockState* nozzle_ptr,
    const Params2D& prm
) {
    const GridBlock& g = *bs.g;
    const int nx = g.nx;
    const int nr = g.nr;
    const double gamma = prm.gamma;
    const double gm1 = gamma - 1.0;
    const double cp = gamma * prm.r_gas / gm1;
    const double amb_rho = std::max(1e-6, prm.pa_pa / (prm.r_gas * 300.0));

    // ---- I-FACES (nx + 1) * nr ----
#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
    for (int i = 0; i <= nx; ++i) {
        for (int j = 0; j < nr; ++j) {
            const int f = i * nr + j;
            const double nx_f = g.i_nx[f];
            const double nr_f = g.i_nr[f];

            Prim wl, wr;
            bool couple_interface = false;

            if (i == 0) {
                if (!is_plume) {
                    // Nozzle inlet (subsonic Riemann Pc, Tc)
                    const Prim& w0 = bs.w[0 * nr + j];
                    const double y_frac = (j + 0.5) / nr;
                    wl = inflow_riemann_state(
                        w0, nx_f, nr_f, prm.pc_pa, prm.tc_k, gamma, prm.r_gas,
                        prm.inflow_slope * y_frac
                    );
                    wr = w0;
                } else {
                    // Plume inlet i == 0
                    wr = bs.w[0 * nr + j];
                    if (nozzle_ptr != nullptr && j < nozzle_ptr->g->nr) {
                        // Exact 1-to-1 interface with Block 1 exit!
                        const GridBlock& gn = *nozzle_ptr->g;
                        const int inm1 = (gn.nx - 1) * gn.nr + j;
                        const int inm2 = std::max(0, gn.nx - 2) * gn.nr + j;
                        const Prim& wm1 = nozzle_ptr->w[inm1];
                        const Prim& wm2 = nozzle_ptr->w[inm2];
                        const Prim& w0 = bs.w[0 * nr + j];
                        const Prim& w1 = bs.w[std::min(nx - 1, 1) * nr + j];
                        wl.rho = wm1.rho + 0.5 * limiter_slope(wm1.rho - wm2.rho, w0.rho - wm1.rho, prm.limiter_id);
                        wl.u   = wm1.u   + 0.5 * limiter_slope(wm1.u - wm2.u,     w0.u - wm1.u,     prm.limiter_id);
                        wl.v   = wm1.v   + 0.5 * limiter_slope(wm1.v - wm2.v,     w0.v - wm1.v,     prm.limiter_id);
                        wl.p   = wm1.p   + 0.5 * limiter_slope(wm1.p - wm2.p,     w0.p - wm1.p,     prm.limiter_id);

                        wr.rho = w0.rho - 0.5 * limiter_slope(w0.rho - wm1.rho, w1.rho - w0.rho, prm.limiter_id);
                        wr.u   = w0.u   - 0.5 * limiter_slope(w0.u - wm1.u,     w1.u - w0.u,     prm.limiter_id);
                        wr.v   = w0.v   - 0.5 * limiter_slope(w0.v - wm1.v,     w1.v - w0.v,     prm.limiter_id);
                        wr.p   = w0.p   - 0.5 * limiter_slope(w0.p - wm1.p,     w1.p - w0.p,     prm.limiter_id);
                        couple_interface = true;
                    } else {
                        // Nozzle outer lip wall / quiet ambient co-flow above nozzle exit radius
                        const double a_amb = std::sqrt(gamma * prm.r_gas * 300.0);
                        wl = {amb_rho, 0.08 * a_amb, 0.0, prm.pa_pa};
                    }
                }
            } else if (i == nx) {
                const Prim& wm1 = bs.w[(nx - 1) * nr + j];
                wl = wm1;
                if (!is_plume && plume_ptr != nullptr && j < plume_ptr->g->nr) {
                    // Identical flux to Plume i=0 interface so Block 1 & 2 conserve mass/momentum/energy 1-to-1!
                    bs.fi[f] = plume_ptr->fi[0 * plume_ptr->g->nr + j];
                    continue;
                }
                wr = outflow_state(wm1, nx_f, nr_f, prm.pa_pa, gamma);
            } else {
                const int cl = (i - 1) * nr + j;
                const int cr = i * nr + j;
                const Prim& wc_l = bs.w[cl];
                const Prim& wc_r = bs.w[cr];
                const Prim& w_ll = (i >= 2) ? bs.w[(i - 2) * nr + j] : wc_l;
                const Prim& w_rr = (i + 1 < nx) ? bs.w[(i + 1) * nr + j] : wc_r;

                wl.rho = wc_l.rho + 0.5 * limiter_slope(wc_l.rho - w_ll.rho, wc_r.rho - wc_l.rho, prm.limiter_id);
                wl.u   = wc_l.u   + 0.5 * limiter_slope(wc_l.u - w_ll.u,     wc_r.u - wc_l.u,     prm.limiter_id);
                wl.v   = wc_l.v   + 0.5 * limiter_slope(wc_l.v - w_ll.v,     wc_r.v - wc_l.v,     prm.limiter_id);
                wl.p   = wc_l.p   + 0.5 * limiter_slope(wc_l.p - w_ll.p,     wc_r.p - wc_l.p,     prm.limiter_id);

                wr.rho = wc_r.rho - 0.5 * limiter_slope(wc_r.rho - wc_l.rho, w_rr.rho - wc_r.rho, prm.limiter_id);
                wr.u   = wc_r.u   - 0.5 * limiter_slope(wc_r.u - wc_l.u,     w_rr.u - wc_r.u,     prm.limiter_id);
                wr.v   = wc_r.v   - 0.5 * limiter_slope(wc_r.v - wc_l.v,     w_rr.v - wc_r.v,     prm.limiter_id);
                wr.p   = wc_r.p   - 0.5 * limiter_slope(wc_r.p - wc_l.p,     w_rr.p - wc_r.p,     prm.limiter_id);

                if (wl.rho < 0.1 * wc_l.rho || wl.p < 0.1 * wc_l.p) wl = wc_l;
                if (wr.rho < 0.1 * wc_r.rho || wr.p < 0.1 * wc_r.p) wr = wc_r;
            }

            if (couple_interface) {
                if (wl.rho < 1e-6 || wl.p < 1.0) wl = nozzle_ptr->w[(nozzle_ptr->g->nx - 1) * nozzle_ptr->g->nr + j];
                if (wr.rho < 1e-6 || wr.p < 1.0) wr = bs.w[j];
            }

            Flux4 flx = hllc_rotated_flux(wl, wr, nx_f, nr_f, gamma);

            // Viscous contribution on interior I-faces (and Block1-Block2 interface approximated cleanly)
            if (prm.viscous_mode > 0 && i > 0 && i < nx) {
                const int cl = (i - 1) * nr + j;
                const int cr = i * nr + j;
                const double mu = 0.5 * (bs.mu_eff[cl] + bs.mu_eff[cr]);
                const double dudx = 0.5 * (bs.gradx_u[cl] + bs.gradx_u[cr]);
                const double dudr = 0.5 * (bs.gradr_u[cl] + bs.gradr_u[cr]);
                const double dvdx = 0.5 * (bs.gradx_v[cl] + bs.gradx_v[cr]);
                const double dvdr = 0.5 * (bs.gradr_v[cl] + bs.gradr_v[cr]);
                const double dTdx = 0.5 * (bs.gradx_T[cl] + bs.gradx_T[cr]);
                const double dTdr = 0.5 * (bs.gradr_T[cl] + bs.gradr_T[cr]);
                const double uf = 0.5 * (bs.w[cl].u + bs.w[cr].u);
                const double vf = 0.5 * (bs.w[cl].v + bs.w[cr].v);
                const double rf = std::max(1e-7, 0.5 * (g.rc[cl] + g.rc[cr]));
                const double divu = dudx + dvdr + vf / rf;
                const double txx = mu * (2.0 * dudx - (2.0 / 3.0) * divu);
                const double trr = mu * (2.0 * dvdr - (2.0 / 3.0) * divu);
                const double txr = mu * (dudr + dvdx);
                const double keff = mu * cp / prm.prandtl;
                const double fv1 = nx_f * txx + nr_f * txr;
                const double fv2 = nx_f * txr + nr_f * trr;
                const double fv3 =
                    nx_f * (uf * txx + vf * txr + keff * dTdx) +
                    nr_f * (uf * txr + vf * trr + keff * dTdr);
                flx.f1 -= fv1;
                flx.f2 -= fv2;
                flx.f3 -= fv3;
            }
            bs.fi[f] = flx;
        }
    }

    // ---- J-FACES nx * (nr + 1) ----
#ifdef _OPENMP
#pragma omp for schedule(static)
#endif
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j <= nr; ++j) {
            const int f = i * (nr + 1) + j;
            const double nx_f = g.j_nx[f];
            const double nr_f = g.j_nr[f];

            if (j == 0) {
                // Symmetry axis r = 0 (g.j_sf[f] == 0 so mass/x-mom/energy flux * 0 == 0, keep pressure finite cleanly)
                const Prim& wc = bs.w[i * nr];
                bs.fj[f] = {0.0, nx_f * wc.p, nr_f * wc.p, 0.0};
                continue;
            }

            if (j == nr) {
                const int cm1 = i * nr + (nr - 1);
                const Prim& wm1 = bs.w[cm1];
                if (!is_plume) {
                    // Solid nozzle wall
                    if (prm.viscous_mode <= 0) {
                        // Euler inviscid slip wall: purely normal wall pressure
                        bs.fj[f] = {0.0, nx_f * wm1.p, nr_f * wm1.p, 0.0};
                    } else {
                        // No-slip viscous wall: zero convective flux + skin-friction shear tau_w
                        const double xw = 0.5 * (g.xv[i * (nr + 1) + nr] + g.xv[(i + 1) * (nr + 1) + nr]);
                        const double rw = 0.5 * (g.rv[i * (nr + 1) + nr] + g.rv[(i + 1) * (nr + 1) + nr]);
                        const double dn = std::max(1e-7, (xw - g.xc[cm1]) * nx_f + (rw - g.rc[cm1]) * nr_f);
                        const double mu = bs.mu_eff[cm1];
                        const double ut_cell = -wm1.u * nr_f + wm1.v * nx_f;
                        // Wall shear opposing positive fluid ut_cell slows fluid down: tau_fluid_on_wall = +mu * (0 - ut)/dn
                        const double tau_t = -mu * ut_cell / dn;
                        const double tau_x = -tau_t * nr_f;
                        const double tau_r = tau_t * nx_f;
                        double q_wall = 0.0;
                        if (prm.use_isothermal_wall != 0) {
                            const double T_cell = wm1.p / (wm1.rho * prm.r_gas);
                            const double keff = mu * cp / prm.prandtl;
                            q_wall = keff * (prm.wall_temp_k - T_cell) / dn;
                        }
                        bs.fj[f] = {0.0, nx_f * wm1.p - tau_x, nr_f * wm1.p - tau_r, -q_wall};
                    }
                    continue;
                } else {
                    // Plume far-field top boundary r = r_max: characteristic outflow or quiet entrainment
                    const double a_amb = std::sqrt(gamma * prm.r_gas * 300.0);
                    const Prim w_far = (wm1.v > 0.0)
                        ? outflow_state(wm1, nx_f, nr_f, prm.pa_pa, gamma)
                        : Prim{amb_rho, 0.08 * a_amb, 0.0, prm.pa_pa};
                    bs.fj[f] = hllc_rotated_flux(wm1, w_far, nx_f, nr_f, gamma);
                    continue;
                }
            }

            // Interior J-face
            const int cb = i * nr + (j - 1);
            const int ct = i * nr + j;
            const Prim& wc_b = bs.w[cb];
            const Prim& wc_t = bs.w[ct];
            const Prim w_bb = (j >= 2) ? bs.w[i * nr + (j - 2)] : Prim{wc_b.rho, wc_b.u, -wc_b.v, wc_b.p};
            const Prim w_tt =
                (j + 1 < nr)
                    ? bs.w[i * nr + (j + 1)]
                    : (is_plume ? wc_t : Prim{wc_t.rho, prm.viscous_mode > 0 ? -wc_t.u : wc_t.u, -wc_t.v, wc_t.p});

            Prim wl, wr;
            wl.rho = wc_b.rho + 0.5 * limiter_slope(wc_b.rho - w_bb.rho, wc_t.rho - wc_b.rho, prm.limiter_id);
            wl.u   = wc_b.u   + 0.5 * limiter_slope(wc_b.u - w_bb.u,     wc_t.u - wc_b.u,     prm.limiter_id);
            wl.v   = wc_b.v   + 0.5 * limiter_slope(wc_b.v - w_bb.v,     wc_t.v - wc_b.v,     prm.limiter_id);
            wl.p   = wc_b.p   + 0.5 * limiter_slope(wc_b.p - w_bb.p,     wc_t.p - wc_b.p,     prm.limiter_id);

            wr.rho = wc_t.rho - 0.5 * limiter_slope(wc_t.rho - wc_b.rho, w_tt.rho - wc_t.rho, prm.limiter_id);
            wr.u   = wc_t.u   - 0.5 * limiter_slope(wc_t.u - wc_b.u,     w_tt.u - wc_t.u,     prm.limiter_id);
            wr.v   = wc_t.v   - 0.5 * limiter_slope(wc_t.v - wc_b.v,     w_tt.v - wc_t.v,     prm.limiter_id);
            wr.p   = wc_t.p   - 0.5 * limiter_slope(wc_t.p - wc_b.p,     w_tt.p - wc_t.p,     prm.limiter_id);

            if (wl.rho < 0.1 * wc_b.rho || wl.p < 0.1 * wc_b.p) wl = wc_b;
            if (wr.rho < 0.1 * wc_t.rho || wr.p < 0.1 * wc_t.p) wr = wc_t;

            Flux4 flx = hllc_rotated_flux(wl, wr, nx_f, nr_f, gamma);

            if (prm.viscous_mode > 0) {
                const double mu = 0.5 * (bs.mu_eff[cb] + bs.mu_eff[ct]);
                const double dudx = 0.5 * (bs.gradx_u[cb] + bs.gradx_u[ct]);
                const double dudr = 0.5 * (bs.gradr_u[cb] + bs.gradr_u[ct]);
                const double dvdx = 0.5 * (bs.gradx_v[cb] + bs.gradx_v[ct]);
                const double dvdr = 0.5 * (bs.gradr_v[cb] + bs.gradr_v[ct]);
                const double dTdx = 0.5 * (bs.gradx_T[cb] + bs.gradx_T[ct]);
                const double dTdr = 0.5 * (bs.gradr_T[cb] + bs.gradr_T[ct]);
                const double uf = 0.5 * (wc_b.u + wc_t.u);
                const double vf = 0.5 * (wc_b.v + wc_t.v);
                const double rf = std::max(1e-7, 0.5 * (g.rc[cb] + g.rc[ct]));
                const double divu = dudx + dvdr + vf / rf;
                const double txx = mu * (2.0 * dudx - (2.0 / 3.0) * divu);
                const double trr = mu * (2.0 * dvdr - (2.0 / 3.0) * divu);
                const double txr = mu * (dudr + dvdx);
                const double keff = mu * cp / prm.prandtl;
                const double fv1 = nx_f * txx + nr_f * txr;
                const double fv2 = nx_f * txr + nr_f * trr;
                const double fv3 =
                    nx_f * (uf * txx + vf * txr + keff * dTdx) +
                    nr_f * (uf * txr + vf * trr + keff * dTdr);
                flx.f1 -= fv1;
                flx.f2 -= fv2;
                flx.f3 -= fv3;
            }
            bs.fj[f] = flx;
        }
    }
}

inline void accumulate_and_step_block(
    BlockState& bs,
    double rk_stage,
    double global_dt,
    const Params2D& prm,
    bool record_res,
    std::vector<double>& res_per_thread
) {
    const GridBlock& g = *bs.g;
    const int nx = g.nx;
    const int nr = g.nr;
    const double gm1 = prm.gamma - 1.0;
    const double rho_ref = prm.pc_pa / (prm.r_gas * prm.tc_k);
    const double a_ref = std::sqrt(prm.gamma * prm.r_gas * prm.tc_k);
    const double scale = std::max(1e-6, rho_ref * a_ref);

    double sum_sq_local = 0.0;

#ifdef _OPENMP
#pragma omp for schedule(static) nowait
#endif
    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < nr; ++j) {
            const int c = i * nr + j;
            const int fw = i * nr + j;
            const int fe = (i + 1) * nr + j;
            const int fs = i * (nr + 1) + j;
            const int fn = i * (nr + 1) + (j + 1);

            const Flux4& fW = bs.fi[fw];
            const Flux4& fE = bs.fi[fe];
            const Flux4& fS = bs.fj[fs];
            const Flux4& fN = bs.fj[fn];

            const double sW = g.i_sf[fw], sE = g.i_sf[fe];
            const double sS = g.j_sf[fs], sN = g.j_sf[fn];

            double r0 = fW.f0 * sW - fE.f0 * sE + fS.f0 * sS - fN.f0 * sN;
            double r1 = fW.f1 * sW - fE.f1 * sE + fS.f1 * sS - fN.f1 * sN;
            double r2 = fW.f2 * sW - fE.f2 * sE + fS.f2 * sS - fN.f2 * sN;
            double r3 = fW.f3 * sW - fE.f3 * sE + fS.f3 * sS - fN.f3 * sN;

            // Exact axisymmetric wedge pressure + viscous hoop stress source
            double tau_tt = 0.0;
            if (prm.viscous_mode > 0) {
                const double divu = bs.gradx_u[c] + bs.gradr_v[c] + bs.w[c].v / std::max(1e-7, g.rc[c]);
                tau_tt = bs.mu_eff[c] * (2.0 * bs.w[c].v / std::max(1e-7, g.rc[c]) - (2.0 / 3.0) * divu);
            }
            r2 += (bs.w[c].p - tau_tt) * g.area[c];

            double dt_c = global_dt;
            if (prm.local_time_step) {
                const Prim& wc = bs.w[c];
                const double a = std::sqrt(prm.gamma * wc.p / wc.rho);
                const double ws = std::hypot(wc.u, wc.v) + a;
                dt_c = prm.cfl * g.h_min[c] / std::max(1e-6, ws);
                if (prm.viscous_mode > 0) {
                    const double nu_eff =
                        std::max(1e-12, prm.gamma * bs.mu_eff[c] / (wc.rho * prm.prandtl));
                    const double dt_v = 0.22 * g.h_min[c] * g.h_min[c] / nu_eff;
                    dt_c = std::min(dt_c, dt_v);
                }
            }

            const double inv_vol = 1.0 / g.vol[c];
            Cons u_next;
            if (rk_stage == 0) {
                bs.u0[c] = bs.u[c];
                u_next.r  = bs.u[c].r  + dt_c * inv_vol * r0;
                u_next.ru = bs.u[c].ru + dt_c * inv_vol * r1;
                u_next.rv = bs.u[c].rv + dt_c * inv_vol * r2;
                u_next.rE = bs.u[c].rE + dt_c * inv_vol * r3;
            } else {
                u_next.r  = 0.5 * (bs.u0[c].r  + bs.u[c].r  + dt_c * inv_vol * r0);
                u_next.ru = 0.5 * (bs.u0[c].ru + bs.u[c].ru + dt_c * inv_vol * r1);
                u_next.rv = 0.5 * (bs.u0[c].rv + bs.u[c].rv + dt_c * inv_vol * r2);
                u_next.rE = 0.5 * (bs.u0[c].rE + bs.u[c].rE + dt_c * inv_vol * r3);
            }

            if (!std::isfinite(u_next.r) || u_next.r <= 1e-8) {
                u_next.r = std::max(1e-6, bs.u[c].r);
                u_next.ru = bs.u[c].ru;
                u_next.rv = bs.u[c].rv;
                u_next.rE = bs.u[c].rE;
            }

            bs.u[c] = u_next;
            bs.w[c] = cons_to_prim(u_next, gm1);

            if (record_res) {
                const double cell_res = std::abs(r1) * inv_vol / scale * g.h_min[c];
                sum_sq_local += cell_res * cell_res;
            }
        }
    }
#ifdef _OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    if (record_res) {
        res_per_thread[tid * 8] += sum_sq_local;
    }
}

inline void run_coupled_ns2d(
    BlockState& b1,
    BlockState* b2_ptr,
    const Params2D& prm,
    SolveOutput2D& out
) {
    const auto t0 = std::chrono::steady_clock::now();
    out.res_hist.clear();
    out.res_hist.reserve(prm.max_iters);

    auto compute_global_dt = [&]() -> double {
        double dt_min = 1e9;
        auto scan_block = [&](const BlockState& bs) {
            const int nc = bs.g->nx * bs.g->nr;
            for (int c = 0; c < nc; ++c) {
                const double a = std::sqrt(prm.gamma * bs.w[c].p / bs.w[c].rho);
                double dt_c = prm.cfl * bs.g->h_min[c] / (std::hypot(bs.w[c].u, bs.w[c].v) + a);
                if (prm.viscous_mode > 0) {
                    const double nu = std::max(1e-12, prm.gamma * bs.mu_eff[c] / (bs.w[c].rho * prm.prandtl));
                    dt_c = std::min(dt_c, 0.22 * bs.g->h_min[c] * bs.g->h_min[c] / nu);
                }
                if (dt_c < dt_min) dt_min = dt_c;
            }
        };
        scan_block(b1);
        if (b2_ptr) scan_block(*b2_ptr);
        return dt_min;
    };

    const int total_cells = b1.g->nx * b1.g->nr + (b2_ptr ? b2_ptr->g->nx * b2_ptr->g->nr : 0);
#ifdef _OPENMP
    const int max_t = omp_get_max_threads();
    const int n_threads_opt = (total_cells < 1500) ? 1 : std::min(8, max_t);
#else
    const int n_threads_opt = 1;
#endif

    std::vector<double> res_per_thread(std::max(1, n_threads_opt) * 8, 0.0);
    double init_res = -1.0;
    bool stop_flag = false;
    double g_dt = 0.0;

#ifdef _OPENMP
#pragma omp parallel num_threads(n_threads_opt)
#endif
    {
        for (int it = 0; it < prm.max_iters && !stop_flag; ++it) {
            if (!prm.local_time_step) {
#ifdef _OPENMP
#pragma omp single
#endif
                {
                    g_dt = compute_global_dt();
                }
            }

            for (int stage = 0; stage < 2; ++stage) {
                if (prm.viscous_mode > 0) {
                    compute_viscous_gradients_and_mu(
                        b1, prm.r_gas, prm.mu_ref, prm.viscous_mode, false, prm.r_wall_exit, prm.pc_pa, prm.tc_k
                    );
                    if (b2_ptr != nullptr) {
                        compute_viscous_gradients_and_mu(
                            *b2_ptr, prm.r_gas, prm.mu_ref, prm.viscous_mode, true, prm.r_wall_exit, prm.pc_pa, prm.tc_k
                        );
                    }
                }
                if (b2_ptr != nullptr) {
                    reconstruct_and_flux_block(*b2_ptr, nullptr, true, &b1, prm);
                }
                reconstruct_and_flux_block(b1, b2_ptr, false, nullptr, prm);

                accumulate_and_step_block(b1, stage, g_dt, prm, stage == 1, res_per_thread);
                if (b2_ptr != nullptr) {
                    accumulate_and_step_block(*b2_ptr, stage, g_dt, prm, stage == 1, res_per_thread);
                }
#ifdef _OPENMP
#pragma omp barrier
#endif
                if (stage == 1) {
#ifdef _OPENMP
#pragma omp single
#endif
                    {
                        double sum_sq = 0.0;
                        for (int t = 0; t < n_threads_opt; ++t) {
                            sum_sq += res_per_thread[t * 8];
                            res_per_thread[t * 8] = 0.0;
                        }
                        const double r_tot = std::sqrt(sum_sq / std::max(1, total_cells));
                        if (init_res < 0.0 && r_tot > 1e-16) init_res = r_tot;
                        const double r_rel = (init_res > 0.0) ? (r_tot / init_res) : r_tot;
                        out.res_hist.push_back(r_tot);
                        out.final_residual = r_tot;
                        out.iterations = it + 1;
                        if (it > 80 && (r_tot < prm.tol || r_rel < 5e-4)) {
                            out.converged = 1;
                            stop_flag = true;
                        }
                    }
                }
            }
        }

        // Evaluate exact final FVM interface fluxes for mass flow conservation metrics
        if (prm.viscous_mode > 0) {
            compute_viscous_gradients_and_mu(
                b1, prm.r_gas, prm.mu_ref, prm.viscous_mode, false, prm.r_wall_exit, prm.pc_pa, prm.tc_k
            );
            if (b2_ptr != nullptr) {
                compute_viscous_gradients_and_mu(
                    *b2_ptr, prm.r_gas, prm.mu_ref, prm.viscous_mode, true, prm.r_wall_exit, prm.pc_pa, prm.tc_k
                );
            }
        }
        if (b2_ptr != nullptr) {
            reconstruct_and_flux_block(*b2_ptr, nullptr, true, &b1, prm);
        }
        reconstruct_and_flux_block(b1, b2_ptr, false, nullptr, prm);
    }

    auto calc_mdot_face = [&](const GridBlock& g, const BlockState& bs, int i_face) {
        double mdot = 0.0;
        for (int j = 0; j < g.nr; ++j) {
            const int f = i_face * g.nr + j;
            mdot += 2.0 * kPi * bs.fi[f].f0 * g.i_sf[f];
        }
        return mdot;
    };

    // Locate throat i-station by minimum outer radius
    int i_th = 0;
    double r_min = 1e9;
    for (int i = 0; i <= b1.g->nx; ++i) {
        const double rw = b1.g->rv[i * (b1.g->nr + 1) + b1.g->nr];
        if (rw < r_min) {
            r_min = rw;
            i_th = i;
        }
    }
    out.mass_flow_throat = calc_mdot_face(*b1.g, b1, i_th);
    out.mass_flow_exit = calc_mdot_face(*b1.g, b1, b1.g->nx);

    const auto t1 = std::chrono::steady_clock::now();
    out.solve_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
}

inline void compute_field_outputs(
    const BlockState& bs,
    double gamma,
    double r_gas,
    double* rho_out,
    double* u_out,
    double* v_out,
    double* p_out,
    double* T_out,
    double* M_out,
    double* schlieren_out
) {
    const GridBlock& g = *bs.g;
    const int nx = g.nx;
    const int nr = g.nr;

    for (int i = 0; i < nx; ++i) {
        for (int j = 0; j < nr; ++j) {
            const int c = i * nr + j;
            const Prim& w = bs.w[c];
            const double T = w.p / (w.rho * r_gas);
            const double a = std::sqrt(gamma * w.p / w.rho);
            rho_out[c] = w.rho;
            u_out[c] = w.u;
            v_out[c] = w.v;
            p_out[c] = w.p;
            T_out[c] = T;
            M_out[c] = std::hypot(w.u, w.v) / std::max(1e-6, a);

            // |grad(rho)| for optical Schlieren
            const int iL = std::max(0, i - 1);
            const int iR = std::min(nx - 1, i + 1);
            const int jB = std::max(0, j - 1);
            const int jT = std::min(nr - 1, j + 1);
            const double dx = std::max(1e-7, g.xc[iR * nr + j] - g.xc[iL * nr + j]);
            const double dr = std::max(1e-7, g.rc[i * nr + jT] - g.rc[i * nr + jB]);
            const double drho_dx = (bs.w[iR * nr + j].rho - bs.w[iL * nr + j].rho) / dx;
            const double drho_dr = (bs.w[i * nr + jT].rho - bs.w[i * nr + jB].rho) / dr;
            schlieren_out[c] = std::hypot(drho_dx, drho_dr);
        }
    }
}

}  // namespace ns2d_core

FVM2D_EXPORT int solve_axisymmetric_ns2d_c(
    int nx1,
    int nr1,
    const double* x1_v,
    const double* r1_v,
    const double* rho1_in,
    const double* u1_in,
    const double* v1_in,
    const double* p1_in,
    int nx2,
    int nr2,
    const double* x2_v,
    const double* r2_v,
    const double* rho2_in,
    const double* u2_in,
    const double* v2_in,
    const double* p2_in,
    double gamma,
    double r_specific,
    double tc_k,
    double pc_pa,
    double pa_pa,
    double mu_ref,
    double prandtl,
    double cfl,
    double inflow_slope,
    int max_iters,
    double tol,
    int limiter_id,
    int viscous_mode,
    int local_time_step,
    int use_isothermal_wall,
    double wall_temp_k,
    double* rho1_out,
    double* u1_out,
    double* v1_out,
    double* p1_out,
    double* T1_out,
    double* M1_out,
    double* schlieren1_out,
    double* rho2_out,
    double* u2_out,
    double* v2_out,
    double* p2_out,
    double* T2_out,
    double* M2_out,
    double* schlieren2_out,
    double* res_hist_out,
    int max_hist_len,
    int* out_iters,
    int* out_converged,
    double* out_mdot_throat,
    double* out_mdot_exit,
    double* out_solve_ms,
    int* out_n_threads
) {
    using namespace ns2d_core;

    if (nx1 < 4 || nr1 < 4 || !x1_v || !r1_v) {
        return -1;
    }

    const double rw_exit = r1_v[nx1 * (nr1 + 1) + nr1];
    GridBlock g1;
    g1.build(nx1, nr1, x1_v, r1_v, rw_exit);

    const int nc1 = nx1 * nr1;
    std::vector<Prim> w1_init(nc1);
    for (int c = 0; c < nc1; ++c) {
        w1_init[c] = {rho1_in[c], u1_in[c], v1_in[c], p1_in[c]};
    }
    BlockState b1;
    b1.init(g1, w1_init.data(), gamma - 1.0);

    GridBlock g2;
    BlockState b2;
    BlockState* b2_ptr = nullptr;
    if (nx2 > 2 && nr2 >= nr1 && x2_v && r2_v) {
        g2.build(nx2, nr2, x2_v, r2_v, rw_exit);
        const int nc2 = nx2 * nr2;
        std::vector<Prim> w2_init(nc2);
        for (int c = 0; c < nc2; ++c) {
            w2_init[c] = {rho2_in[c], u2_in[c], v2_in[c], p2_in[c]};
        }
        b2.init(g2, w2_init.data(), gamma - 1.0);
        b2_ptr = &b2;
    }

    Params2D prm;
    prm.gamma = gamma;
    prm.r_gas = r_specific;
    prm.tc_k = tc_k;
    prm.pc_pa = pc_pa;
    prm.pa_pa = pa_pa;
    prm.mu_ref = mu_ref;
    prm.prandtl = prandtl;
    prm.cfl = cfl;
    prm.r_wall_exit = rw_exit;
    prm.inflow_slope = inflow_slope;
    prm.max_iters = max_iters;
    prm.tol = tol;
    prm.limiter_id = limiter_id;
    prm.viscous_mode = viscous_mode;
    prm.local_time_step = (local_time_step != 0);
    prm.use_isothermal_wall = use_isothermal_wall;
    prm.wall_temp_k = wall_temp_k;

    SolveOutput2D sout;
    run_coupled_ns2d(b1, b2_ptr, prm, sout);

    compute_field_outputs(b1, gamma, r_specific, rho1_out, u1_out, v1_out, p1_out, T1_out, M1_out, schlieren1_out);
    if (b2_ptr != nullptr && rho2_out != nullptr) {
        compute_field_outputs(
            b2, gamma, r_specific, rho2_out, u2_out, v2_out, p2_out, T2_out, M2_out, schlieren2_out
        );
    }

    if (res_hist_out && max_hist_len > 0) {
        const int ncopy = std::min<int>(max_hist_len, static_cast<int>(sout.res_hist.size()));
        std::memcpy(res_hist_out, sout.res_hist.data(), ncopy * sizeof(double));
    }
    if (out_iters) *out_iters = sout.iterations;
    if (out_converged) *out_converged = sout.converged;
    if (out_mdot_throat) *out_mdot_throat = sout.mass_flow_throat;
    if (out_mdot_exit) *out_mdot_exit = sout.mass_flow_exit;
    if (out_solve_ms) *out_solve_ms = sout.solve_ms;
    if (out_n_threads) {
#ifdef _OPENMP
        *out_n_threads = omp_get_max_threads();
#else
        *out_n_threads = 1;
#endif
    }
    return 0;
}

#ifndef NOZZLE_FVM2D_BUILD_DLL
int main(int argc, char** argv) {
    using namespace ns2d_core;

    double pc_mpa = 10.0;
    double pa_kpa = 101.325;
    double eps = 12.0;
    double rt = 0.12;
    int nx1 = 80;
    int nr1 = 28;
    int viscous_mode = 2;
    bool json_out = false;

    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--pc-mpa" && i + 1 < argc) pc_mpa = std::stod(argv[++i]);
        else if (a == "--pa-kpa" && i + 1 < argc) pa_kpa = std::stod(argv[++i]);
        else if (a == "--eps" && i + 1 < argc) eps = std::stod(argv[++i]);
        else if (a == "--nx" && i + 1 < argc) nx1 = std::stoi(argv[++i]);
        else if (a == "--nr" && i + 1 < argc) nr1 = std::stoi(argv[++i]);
        else if (a == "--viscous-mode" && i + 1 < argc) viscous_mode = std::stoi(argv[++i]);
        else if (a == "--json") json_out = true;
    }

    // Build smooth cosine-transition conical-bell verification mesh directly in C++
    const double gamma = 1.21;
    const double r_spec = 399.73;
    const double tc_k = 3560.0;
    const double pc_pa = pc_mpa * 1e6;
    const double pa_pa = pa_kpa * 1e3;

    const double r_in = rt * std::sqrt(3.0);
    const double r_ex = rt * std::sqrt(eps);
    const double x_th = 0.25;
    const double len = x_th + 0.8 * (r_ex - rt) / std::tan(15.0 * kPi / 180.0);

    std::vector<double> x1v((nx1 + 1) * (nr1 + 1));
    std::vector<double> r1v((nx1 + 1) * (nr1 + 1));

    for (int i = 0; i <= nx1; ++i) {
        const double xi = static_cast<double>(i) / nx1;
        const double x = xi * len;
        double rw = rt;
        if (x < x_th) {
            const double s = 0.5 * (1.0 - std::cos(kPi * (x / x_th)));
            rw = r_in + (rt - r_in) * s;
        } else {
            const double s = (x - x_th) / (len - x_th);
            rw = rt + (r_ex - rt) * (0.4 * s + 0.6 * (1.0 - (1.0 - s) * (1.0 - s)));
        }
        for (int j = 0; j <= nr1; ++j) {
            const double eta = static_cast<double>(j) / nr1;
            const int idx = i * (nr1 + 1) + j;
            x1v[idx] = x;
            r1v[idx] = rw * eta;
        }
    }

    const int nc1 = nx1 * nr1;
    std::vector<double> rho1(nc1), u1(nc1), v1(nc1), p1(nc1), T1(nc1), M1(nc1), sch1(nc1);
    for (int i = 0; i < nx1; ++i) {
        const double xi = (i + 0.5) / nx1;
        const double x = xi * len;
        const double m_guess = (x < x_th) ? (0.25 + 0.72 * (x / x_th)) : (1.02 + 2.2 * (x - x_th) / (len - x_th));
        const double fac = 1.0 + 0.5 * (gamma - 1.0) * m_guess * m_guess;
        const double Tg = tc_k / fac;
        const double pg = pc_pa * std::pow(fac, -gamma / (gamma - 1.0));
        const double rg = pg / (r_spec * Tg);
        const double vg = m_guess * std::sqrt(gamma * r_spec * Tg);
        for (int j = 0; j < nr1; ++j) {
            const int c = i * nr1 + j;
            rho1[c] = rg;
            u1[c] = vg;
            v1[c] = 0.0;
            p1[c] = pg;
        }
    }

    std::vector<double> hist(3000, 0.0);
    int iters = 0, conv = 0, nth = 1;
    double mdot_t = 0.0, mdot_e = 0.0, ms = 0.0;
    solve_axisymmetric_ns2d_c(
        nx1, nr1, x1v.data(), r1v.data(), rho1.data(), u1.data(), v1.data(), p1.data(),
        0, 0, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        gamma, r_spec, tc_k, pc_pa, pa_pa, 1.716e-5, 0.72, 0.40, 0.0,
        2200, 1e-4, 2, viscous_mode, 1, 0, 900.0,
        rho1.data(), u1.data(), v1.data(), p1.data(), T1.data(), M1.data(), sch1.data(),
        nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
        hist.data(), static_cast<int>(hist.size()),
        &iters, &conv, &mdot_t, &mdot_e, &ms, &nth
    );

    const double exit_m_cl = M1[(nx1 - 1) * nr1];
    const double exit_m_wall = M1[(nx1 - 1) * nr1 + (nr1 - 1)];
    const double mass_err = 100.0 * std::abs(mdot_e - mdot_t) / std::max(1e-9, std::abs(mdot_t));

    if (json_out) {
        std::cout << std::fixed << std::setprecision(4)
                  << "{\n"
                  << "  \"solver\": \"cpp20_openmp_axisymmetric_ns2d\",\n"
                  << "  \"threads\": " << nth << ",\n"
                  << "  \"iterations\": " << iters << ",\n"
                  << "  \"converged\": " << (conv ? "true" : "false") << ",\n"
                  << "  \"solve_ms\": " << ms << ",\n"
                  << "  \"mass_flow_throat_kg_s\": " << mdot_t << ",\n"
                  << "  \"mass_flow_exit_kg_s\": " << mdot_e << ",\n"
                  << "  \"mass_error_pct\": " << mass_err << ",\n"
                  << "  \"exit_mach_centerline\": " << exit_m_cl << ",\n"
                  << "  \"exit_mach_wall\": " << exit_m_wall << "\n"
                  << "}\n";
    } else {
        std::cout << "[C++20/OpenMP 2D Axisymmetric NS] threads=" << nth << " iters=" << iters
                  << " conv=" << conv << " time=" << ms << " ms M_exit(CL)=" << exit_m_cl
                  << " mass_err=" << mass_err << "%\n";
    }
    return 0;
}
#endif
