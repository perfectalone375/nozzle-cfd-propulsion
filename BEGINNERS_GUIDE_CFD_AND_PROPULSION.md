# Beginner's Guide to Rocket Propulsion, Supersonic Nozzles & CFD

Welcome! This guide explains **every physics equation and every line of CFD code** in this repository from first principles—no PhD required.

---

## 1. Why Is a Rocket Nozzle Shaped Like an Hourglass (De Laval Nozzle)?

If you squeeze a garden hose, slow **subsonic** water speeds up because the same mass must squeeze through a smaller hole. You might guess making a rocket nozzle narrower all the way to the end would shoot exhaust fastest—but that stops working at **Mach 1.0 (the speed of sound)**!

Above Mach 1, hot gas becomes **compressible**: when pressure drops, density drops faster than pressure, so to keep accelerating **supersonically**, the nozzle actually has to **widen again**:

```text
  Combustion Chamber         Throat (M = 1.0)         Rao Bell Diverging Section
  High Pressure (10 MPa)     Choked Sonic Flow        Supersonic Exhaust (M > 3.4)
  Slow Subsonic (M ~ 0.18)   Peak Heat Flux!          Converts Heat -> Kinetic Energy
   _____________________
  |                     \       _                    ______________________
  |  HOT COMBUSTION GAS  \_____/ \__________________/   SUPERSONIC PLUME
  |       3560 K         /     \_/                  \______________________
  |_____________________/
```

That fundamental law is captured by the **Area–Velocity equation**:
$$\frac{dA}{A} = (M^2 - 1)\frac{du}{u}$$
* When $M < 1$ (subsonic): $M^2 - 1 < 0$, so **shrinking** area ($dA < 0$) speeds flow up ($du > 0$).
* When $M = 1$ (sonic throat): $dA = 0$ — the exact geometric minimum throat!
* When $M > 1$ (supersonic): $M^2 - 1 > 0$, so **widening** area ($dA > 0$) blasts gas to 3,000+ m/s!

---

## 2. Multi-Propellant Tradeoffs: Hydrogen vs Methane vs Kerosene vs Solid

Rocket performance depends on two key numbers:
1. **Characteristic Velocity ($c^*$)** — measures how hot and light the chamber combustion gas is ($c^* \propto \sqrt{T_c / M_w}$).
2. **Specific Impulse ($I_{sp}$)** — fuel efficiency (how many seconds 1 kg of propellant produces 1 kg-force of thrust).

| Propellant | Real Rocket Examples | Flame Temp $T_c$ | Exhaust $M_w$ | Why Engineers Pick It |
| :--- | :--- | :--- | :--- | :--- |
| **Hydrolox** (LOX/LH2) | RS-25 (SSME), RL-10 | ~3500 K | 13.4 g/mol | **Highest $I_{sp}$ (>400 s vac)** because H2O+H2 molecules are ultra-light! Hard to store at 20 K. |
| **Methalox** (LOX/CH4) | SpaceX Raptor, BE-4 | ~3560 K | 20.8 g/mol | **Best reusable balance**: high $I_{sp}$, clean soot-free turbine/cooling channels, liquid near LOX temp. |
| **Kerolox** (LOX/RP-1) | Falcon 9 Merlin, Saturn V F-1 | ~3670 K | 23.2 g/mol | **Dense & compact** room-temperature fuel -> tiny tanks for sea-level liftoff thrust. |
| **Solid APCP** | SLS SRBs, Ariane P120C | ~3380 K | 26.8 g/mol | **Simple & instant readiness** -> massive liftoff punch, no turbopumps, cannot throttle/restart. |

---

## 3. What Is CFD (Computational Fluid Dynamics) Actually Doing?

Analytical textbooks give clean formulas *only* when friction and complex transients are ignored. Real CFD splits the nozzle into $N = 141$ tiny slices along $x$ and marches three conserved physical quantities forward in tiny time steps $\Delta t$:

1. $U_1 = \rho' A'$ — **Conservation of Mass** (gas mass inside slice)
2. $U_2 = \rho' A' V'$ — **Conservation of Momentum** (fluid push + wall pressure force)
3. $U_3 = \rho' A' \left(\frac{T'}{\gamma - 1} + \frac{\gamma}{2} V'^2\right)$ — **Conservation of Total Energy** (thermal + kinetic)

### MacCormack's Two-Step Predictor–Corrector Method
At every iteration:
1. **Predictor Step (Forward Difference):** Looks slightly *ahead* (`F[i+1] - F[i]`) to estimate where the gas will be at $t + \Delta t$.
2. **Corrector Step (Backward Difference):** Looks *behind* (`F_p[i] - F_p[i-1]`) using the predicted values and **averages** both slopes. This cancels 1st-order error, making the solver **2nd-order accurate in both space and time**!
3. **Shock Sensor (Jameson Artificial Viscosity):** Right across a razor-thin shock wave where pressure jumps abruptly, raw 2nd-order finite differences oscillate like guitar strings. Our pressure sensor detects steep $|p_{i+1} - 2p_i + p_{i-1}|$ curvature and applies local smoothing **only** at the shock front, leaving smooth supersonic flow untouched.

### Why Must We Obey the CFL Speed Limit?
Every wave moves through the grid at max speed $|V| + a$ (flow speed plus local sound speed). If our timestep $\Delta t$ is too large, fluid information skips over a whole grid cell in one jump and the math **explodes to `NaN`**! So we clamp:
$$\Delta t = C_{\text{CFL}} \frac{\Delta x}{\max(|V_i| + a_i)}, \quad C_{\text{CFL}} = 0.45 < 1.0$$

---

## 4. Normal Shocks vs Plume Shock Diamonds

* **Normal Shock Inside the Nozzle:** If chamber pressure is too low compared to ambient air (say $P_c = 1\text{ MPa}$ fighting $380\text{ kPa}$ back-pressure), the supersonic gas overexpands so far below ambient pressure that air pushes *back into the bell*, creating an irreversible **standing normal shock wave** inside the nozzle that slams flow from **Mach > 2.5 down to subsonic Mach ~0.5** in milliseconds!
* **Shock Diamonds Outside in the Plume:** At launch ($P_c = 10\text{ MPa}$, sea level), the exhaust exits supersonic ($M > 3.2$) with exit pressure slightly mismatched against atmospheric pressure $P_a = 101.3\text{ kPa}$. Oblique expansion fans and oblique compression shocks bounce back and forth between the centerline and plume boundary, creating glowing Mach diamonds spaced by **Pack's shock-cell wavelength $L_{\text{cell}}$**.

---

## 5. Why Don't Rocket Nozzles Melt at 3,560 K? (Bartz Cooling)

Inside a 10 MPa Methalox throat, convective heat flux exceeds **45–50 MW/m²**—roughly **1,000× hotter than a kitchen oven**!
Using the **Bartz Equation**, our solver shows how heat flux spikes steeply at the narrowest throat ($q'' \propto (A_t/A)^{0.9}$):
* **Inconel 718** (low thermal conductivity $k = 21.5\text{ W/mK}$) hits **>2,400 K** and **melts (`wall_survives = False`)**.
* **NASA GRCop-42 Copper-Chrome-Niobium** ($k = 310\text{ W/mK}$) cooled internally by cold liquid fuel keeps the hot-gas wall near **~950–1,010 K**, giving a safe thermal margin (`wall_survives = True`)!

---

## 6. Code Rosetta Stone: Python vs MATLAB vs C++17 Side-by-Side

Seeing the **exact same CFD predictor equation** in all three industry aerospace languages builds deep fluency:

### Python (`src/nozzle_cfd/cfd_solver.py` — Vectorized NumPy)
```python
du1_p = -(f1[2:] - f1[1:-1]) / dx_nd
du2_p = -(f2[2:] - f2[1:-1]) / dx_nd + j2[1:-1]
u1_p[1:-1] = u1[1:-1] + dt * du1_p + s1_p
u2_p[1:-1] = u2[1:-1] + dt * du2_p + s2_p
```

### MATLAB / GNU Octave (`matlab/solve_nozzle_cfd_maccormack.m` — 1-Indexed Engineering Matrix)
```matlab
for i = 2:(nx - 1)
    J2 = (1.0 / gamma) * p(i) * dA_dx(i);
    U1_p(i) = U1(i) - (dt / dx) * (F1(i+1) - F1(i)) + S1;
    U2_p(i) = U2(i) - (dt / dx) * (F2(i+1) - F2(i)) + dt * J2 + S2;
end
```

### C++17 (`cpp/maccormack_cfd.cpp` — Compiled `g++ -O3` Solves 6,000 Steps in Milliseconds)
```cpp
for (int i = 1; i < n - 1; ++i) {
    double j2 = (1.0 / gam) * pres[i] * dA_dx[i];
    U1p[i] = U1[i] - (dt / dx) * (F1[i + 1] - F1[i]) + sensor[i] * (U1[i + 1] - 2.0 * U1[i] + U1[i - 1]);
    U2p[i] = U2[i] - (dt / dx) * (F2[i + 1] - F2[i]) + dt * j2 + sensor[i] * (U2[i + 1] - 2.0 * U2[i] + U2[i - 1]);
}
```

### How to Run Everything Yourself
```bash
# 1. Python full multi-propellant CFD & plots + ParaView .vtk mesh
nozzle-cfd --propellant METHALOX --export-all --output-dir deliverables

# 2. C++17 ultra-fast compiled benchmark
g++ -O3 -std=c++17 -o maccormack_cfd cpp/maccormack_cfd.cpp
./maccormack_cfd --pc-mpa 1.0 --pa-kpa 380.0 --eps 8.0 --json

# 3. MATLAB or GNU Octave validation
octave --no-gui matlab/solve_nozzle_cfd_maccormack.m
```
