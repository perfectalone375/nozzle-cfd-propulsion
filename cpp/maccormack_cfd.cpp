// High-Speed C++17 Conservative MacCormack Quasi-1D Rocket Nozzle CFD Solver
// Compile: g++ -O3 -std=c++17 -Wall -Wextra -o maccormack_cfd maccormack_cfd.cpp

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

struct SolverConfig {
    double gamma = 1.21;
    double pc_mpa = 10.0;
    double tc_k = 3560.0;
    double pa_kpa = 101.325;
    double eps = 12.0;
    double rt_m = 0.12;
    int nx = 161;
    int max_iter = 6000;
    double cfl = 0.45;
    double tol = 1e-6;
    bool json_out = false;
    std::string csv_path = "";
};

static double area_mach(double M, double gam) {
    double gm1 = gam - 1.0;
    double gp1 = gam + 1.0;
    double term = (2.0 / gp1) * (1.0 + 0.5 * gm1 * M * M);
    return (1.0 / M) * std::pow(term, gp1 / (2.0 * gm1));
}

static double solve_mach(double ar, double gam, bool supersonic) {
    if (ar <= 1.000001) return 1.0;
    double lo = supersonic ? 1.00001 : 1e-4;
    double hi = supersonic ? 18.0 : 0.99999;
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

int main(int argc, char* argv[]) {
    SolverConfig cfg;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--gamma" && i + 1 < argc) cfg.gamma = std::stod(argv[++i]);
        else if (arg == "--pc-mpa" && i + 1 < argc) cfg.pc_mpa = std::stod(argv[++i]);
        else if (arg == "--tc-k" && i + 1 < argc) cfg.tc_k = std::stod(argv[++i]);
        else if (arg == "--pa-kpa" && i + 1 < argc) cfg.pa_kpa = std::stod(argv[++i]);
        else if (arg == "--eps" && i + 1 < argc) cfg.eps = std::stod(argv[++i]);
        else if (arg == "--nx" && i + 1 < argc) cfg.nx = std::stoi(argv[++i]);
        else if (arg == "--max-iter" && i + 1 < argc) cfg.max_iter = std::stoi(argv[++i]);
        else if (arg == "--csv" && i + 1 < argc) cfg.csv_path = argv[++i];
        else if (arg == "--json") cfg.json_out = true;
    }

    auto t_start = std::chrono::steady_clock::now();

    const int n = std::max(41, cfg.nx);
    const double gam = cfg.gamma;
    const double gm1 = gam - 1.0;
    const double gp1 = gam + 1.0;
    const double pc_pa = cfg.pc_mpa * 1.0e6;
    const double pa_pa = std::max(1.0, cfg.pa_kpa * 1.0e3);

    // Build axisymmetric nozzle A(x)/At along nondimensional length 0..3.5
    std::vector<double> x(n), area(n), dA_dx(n);
    const double dx = 3.5 / static_cast<double>(n - 1);
    int throat_idx = 0;
    double min_area = 1e9;
    for (int i = 0; i < n; ++i) {
        x[i] = i * dx;
        if (x[i] <= 1.0) {
            area[i] = 1.0 + 2.2 * (1.0 - x[i]) * (1.0 - x[i]);
        } else {
            double s = (x[i] - 1.0) / 2.5;
            area[i] = 1.0 + (cfg.eps - 1.0) * std::pow(s, 1.4);
        }
        if (area[i] < min_area) {
            min_area = area[i];
            throat_idx = i;
        }
    }
    for (int i = 0; i < n; ++i) {
        if (i == 0) dA_dx[i] = (area[1] - area[0]) / dx;
        else if (i == n - 1) dA_dx[i] = (area[n - 1] - area[n - 2]) / dx;
        else dA_dx[i] = (area[i + 1] - area[i - 1]) / (2.0 * dx);
    }

    double m_e_sup = solve_mach(area.back(), gam, true);
    double p_e_sup = pc_pa * std::pow(1.0 + 0.5 * gm1 * m_e_sup * m_e_sup, -gam / gm1);
    double p_e_shocked = p_e_sup * (1.0 + (2.0 * gam / gp1) * (m_e_sup * m_e_sup - 1.0));
    bool subsonic_outflow = pa_pa > p_e_shocked;
    double pb_nd = pa_pa / pc_pa;
    double cx = subsonic_outflow ? 0.30 : 0.08;

    std::vector<double> U1(n), U2(n), U3(n);
    for (int i = 0; i < n; ++i) {
        double m0 = solve_mach(area[i], gam, i > throat_idx);
        if (subsonic_outflow && i > throat_idx) {
            double frac = static_cast<double>(i - throat_idx) / static_cast<double>(n - throat_idx);
            m0 = std::max(0.25, m0 * (1.0 - 0.65 * frac));
        }
        double t0 = 1.0 / (1.0 + 0.5 * gm1 * m0 * m0);
        double rho0 = std::pow(t0, 1.0 / gm1);
        double v0 = m0 * std::sqrt(t0);
        U1[i] = rho0 * area[i];
        U2[i] = rho0 * area[i] * v0;
        U3[i] = rho0 * area[i] * (t0 / gm1 + 0.5 * gam * v0 * v0);
    }

    std::vector<double> rho(n), vel(n), temp(n), pres(n), mach(n), sensor(n);
    std::vector<double> F1(n), F2(n), F3(n), U1p(n), U2p(n), U3p(n);
    std::vector<double> rho_p(n), Vp(n), Tp(n), pp(n), sensor_p(n), F1p(n), F2p(n), F3p(n);
    std::vector<double> U1n(n), U2n(n), U3n(n);

    bool converged = false;
    int iters = 0;
    double final_res = 1.0;

    for (int it = 1; it <= cfg.max_iter; ++it) {
        double max_char_speed = 1e-6;
        for (int i = 0; i < n; ++i) {
            double u1s = std::max(U1[i], 1e-8);
            rho[i] = std::max(u1s / area[i], 1e-8);
            vel[i] = U2[i] / u1s;
            temp[i] = std::max(gm1 * ((U3[i] / u1s) - 0.5 * gam * vel[i] * vel[i]), 1e-4);
            pres[i] = rho[i] * temp[i];
            max_char_speed = std::max(max_char_speed, std::abs(vel[i]) + std::sqrt(temp[i]));

            F1[i] = U2[i];
            double kin = 0.5 * gam * (U2[i] * U2[i]) / u1s;
            F2[i] = (U2[i] * U2[i]) / u1s + (gm1 / gam) * (U3[i] - kin);
            F3[i] = gam * (U2[i] * U3[i]) / u1s - 0.5 * gam * gm1 * (U2[i] * U2[i] * U2[i]) / (u1s * u1s);
        }
        double dt = cfg.cfl * dx / max_char_speed;

        std::fill(sensor.begin(), sensor.end(), 0.0);
        for (int i = 1; i < n - 1; ++i) {
            double denom = std::max(pres[i + 1] + 2.0 * pres[i] + pres[i - 1], 1e-8);
            sensor[i] = cx * std::abs(pres[i + 1] - 2.0 * pres[i] + pres[i - 1]) / denom;
        }

        U1p = U1; U2p = U2; U3p = U3;
        for (int i = 1; i < n - 1; ++i) {
            double j2 = (1.0 / gam) * pres[i] * dA_dx[i];
            U1p[i] = U1[i] - (dt / dx) * (F1[i + 1] - F1[i]) + sensor[i] * (U1[i + 1] - 2.0 * U1[i] + U1[i - 1]);
            U2p[i] = U2[i] - (dt / dx) * (F2[i + 1] - F2[i]) + dt * j2 + sensor[i] * (U2[i + 1] - 2.0 * U2[i] + U2[i - 1]);
            U3p[i] = U3[i] - (dt / dx) * (F3[i + 1] - F3[i]) + sensor[i] * (U3[i + 1] - 2.0 * U3[i] + U3[i - 1]);
        }

        U2p[0] = std::max(0.01 * area[0], 2.0 * U2p[1] - U2p[2]);
        double v0p = std::min(0.85, std::max(0.01, U2p[0] / std::max(U1[0], 1e-6)));
        double t0p = std::max(0.5, 1.0 - 0.5 * gm1 * v0p * v0p);
        U1p[0] = std::pow(t0p, 1.0 / gm1) * area[0];
        U3p[0] = U1p[0] * (t0p / gm1 + 0.5 * gam * v0p * v0p);

        U1p[n - 1] = std::max(1e-6, 2.0 * U1p[n - 2] - U1p[n - 3]);
        U2p[n - 1] = std::max(1e-6, 2.0 * U2p[n - 2] - U2p[n - 3]);
        if (subsonic_outflow) {
            U3p[n - 1] = (pb_nd * area[n - 1]) / gm1 + 0.5 * gam * (U2p[n - 1] * U2p[n - 1]) / U1p[n - 1];
        } else {
            U3p[n - 1] = std::max(1e-6, 2.0 * U3p[n - 2] - U3p[n - 3]);
        }

        for (int i = 0; i < n; ++i) {
            double u1s = std::max(U1p[i], 1e-8);
            rho_p[i] = std::max(u1s / area[i], 1e-8);
            Vp[i] = U2p[i] / u1s;
            Tp[i] = std::max(gm1 * ((U3p[i] / u1s) - 0.5 * gam * Vp[i] * Vp[i]), 1e-4);
            pp[i] = rho_p[i] * Tp[i];

            F1p[i] = U2p[i];
            double kin = 0.5 * gam * (U2p[i] * U2p[i]) / u1s;
            F2p[i] = (U2p[i] * U2p[i]) / u1s + (gm1 / gam) * (U3p[i] - kin);
            F3p[i] = gam * (U2p[i] * U3p[i]) / u1s - 0.5 * gam * gm1 * (U2p[i] * U2p[i] * U2p[i]) / (u1s * u1s);
        }

        std::fill(sensor_p.begin(), sensor_p.end(), 0.0);
        for (int i = 1; i < n - 1; ++i) {
            double denom = std::max(pp[i + 1] + 2.0 * pp[i] + pp[i - 1], 1e-8);
            sensor_p[i] = cx * std::abs(pp[i + 1] - 2.0 * pp[i] + pp[i - 1]) / denom;
        }

        U1n = U1; U2n = U2; U3n = U3;
        for (int i = 1; i < n - 1; ++i) {
            double j2p = (1.0 / gam) * pp[i] * dA_dx[i];
            U1n[i] = 0.5 * (U1[i] + U1p[i] - (dt / dx) * (F1p[i] - F1p[i - 1])) + 0.5 * sensor_p[i] * (U1p[i + 1] - 2.0 * U1p[i] + U1p[i - 1]);
            U2n[i] = 0.5 * (U2[i] + U2p[i] - (dt / dx) * (F2p[i] - F2p[i - 1]) + dt * j2p) + 0.5 * sensor_p[i] * (U2p[i + 1] - 2.0 * U2p[i] + U2p[i - 1]);
            U3n[i] = 0.5 * (U3[i] + U3p[i] - (dt / dx) * (F3p[i] - F3p[i - 1])) + 0.5 * sensor_p[i] * (U3p[i + 1] - 2.0 * U3p[i] + U3p[i - 1]);
        }

        U2n[0] = std::max(0.01 * area[0], 2.0 * U2n[1] - U2n[2]);
        double v0 = std::min(0.85, std::max(0.01, U2n[0] / std::max(U1n[0], 1e-6)));
        double t0 = std::max(0.5, 1.0 - 0.5 * gm1 * v0 * v0);
        U1n[0] = std::pow(t0, 1.0 / gm1) * area[0];
        U3n[0] = U1n[0] * (t0 / gm1 + 0.5 * gam * v0 * v0);

        U1n[n - 1] = std::max(1e-6, 2.0 * U1n[n - 2] - U1n[n - 3]);
        U2n[n - 1] = std::max(1e-6, 2.0 * U2n[n - 2] - U2n[n - 3]);
        if (subsonic_outflow) {
            U3n[n - 1] = (pb_nd * area[n - 1]) / gm1 + 0.5 * gam * (U2n[n - 1] * U2n[n - 1]) / U1n[n - 1];
        } else {
            U3n[n - 1] = std::max(1e-6, 2.0 * U3n[n - 2] - U3n[n - 3]);
        }

        double max_diff = 0.0;
        for (int i = 0; i < n; ++i) {
            max_diff = std::max(max_diff, std::abs(U2n[i] - U2[i]) / std::max(std::abs(U2[i]), 1e-4));
        }
        U1 = U1n; U2 = U2n; U3 = U3n;
        iters = it;
        final_res = max_diff;
        if (it > 200 && max_diff < cfg.tol) {
            converged = true;
            break;
        }
    }

    for (int i = 0; i < n; ++i) {
        rho[i] = U1[i] / area[i];
        vel[i] = U2[i] / U1[i];
        temp[i] = gm1 * ((U3[i] / U1[i]) - 0.5 * gam * vel[i] * vel[i]);
        pres[i] = rho[i] * temp[i];
        mach[i] = vel[i] / std::sqrt(temp[i]);
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed_ms = std::chrono::duration<double, std::milli>(t_end - t_start).count();

    if (!cfg.csv_path.empty()) {
        std::ofstream ofs(cfg.csv_path);
        ofs << "x_nd,area_ratio,mach,pressure_kpa,temperature_k\n";
        for (int i = 0; i < n; ++i) {
            ofs << std::fixed << std::setprecision(5)
                << x[i] << "," << area[i] << "," << mach[i] << ","
                << (pres[i] * cfg.pc_mpa * 1000.0) << "," << (temp[i] * cfg.tc_k) << "\n";
        }
    }

    if (cfg.json_out) {
        std::cout << std::fixed << std::setprecision(4)
                  << "{"
                  << "\"solver\":\"cpp17_maccormack\","
                  << "\"nx\":" << n << ","
                  << "\"iterations\":" << iters << ","
                  << "\"converged\":" << (converged ? "true" : "false") << ","
                  << "\"solve_ms\":" << elapsed_ms << ","
                  << "\"throat_mach\":" << mach[throat_idx] << ","
                  << "\"exit_mach\":" << mach.back() << ","
                  << "\"analytical_supersonic_exit_mach\":" << m_e_sup << ","
                  << "\"exit_pressure_kpa\":" << (pres.back() * cfg.pc_mpa * 1000.0) << ","
                  << "\"subsonic_shock_mode\":" << (subsonic_outflow ? "true" : "false") << ","
                  << "\"final_residual\":" << std::scientific << final_res
                  << "}\n";
    } else {
        std::cout << "C++17 MacCormack CFD | nx=" << n
                  << " | iters=" << iters
                  << " | converged=" << (converged ? "YES" : "NO")
                  << " | elapsed_ms=" << std::fixed << std::setprecision(2) << elapsed_ms
                  << " | M_throat=" << std::setprecision(4) << mach[throat_idx]
                  << " | M_exit=" << mach.back()
                  << " | P_exit_kPa=" << std::setprecision(2) << (pres.back() * cfg.pc_mpa * 1000.0)
                  << "\n";
    }
    return converged ? 0 : 1;
}
