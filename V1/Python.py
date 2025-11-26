import numpy as np
import matplotlib.pyplot as plt
from scipy.odr import *

# ===================================================================
#      1. USER INPUT SECTION (STUDENTS EDIT THIS)
# ===================================================================

# --- Aluminium (Al) Measurements ---
# Mass in [g], Dimensions in [mm]
m_Al_raw = [9.90, 9.89, 9.91, 9.90, 9.92, 9.89, 9.90, 9.91, 9.90, 9.91]
d_Al_raw = [14.00, 14.01, 13.99, 14.00, 14.02, 14.00, 14.01, 13.99, 14.00, 14.01]
h_Al_raw = [23.82, 23.83, 23.82, 23.81, 23.83, 23.82, 23.82, 23.81, 23.83, 23.82]

# --- Lithium Fluoride (LiF) Measurements ---
m_LiF_raw = [16.11, 16.12, 16.11, 16.10, 16.11, 16.12, 16.11, 16.11, 16.12, 16.11]
d_LiF_raw = [25.58, 25.57, 25.58, 25.58, 25.57, 25.58, 25.58, 25.57, 25.58, 25.58]
h_LiF_raw = [11.64, 11.63, 11.64, 11.63, 11.64, 11.64, 11.63, 11.64, 11.63, 11.64]

# --- Calcium Fluoride (CaF2) Measurements ---
m_CaF2_raw = [10.02, 10.03, 10.02, 10.02, 10.03, 10.02, 10.02, 10.03, 10.02, 10.02]
d_CaF2_raw = [19.99, 19.99, 19.98, 19.99, 19.99, 19.98, 19.99, 19.99, 19.99, 19.99]
h_CaF2_raw = [10.06, 10.06, 10.07, 10.06, 10.06, 10.06, 10.07, 10.06, 10.06, 10.06]

# --- Constants ---
# Molar Mass [g/mol], Lattice Constant a [cm] (converted from Angstrom)
M_Al, a_Al, Nz_Al = 26.982, 4.049e-8, 4   
M_LiF, a_LiF, Nz_LiF = 25.939, 4.026e-8, 4

# For CaF2, calculate a from ionic radius r = 2.365 Angstrom
r_CaF2_Angstrom = 2.365
a_CaF2 = (4 / np.sqrt(3)) * r_CaF2_Angstrom * 1e-8 # Convert A to cm
M_CaF2, Nz_CaF2 = 78.07, 4

# ===================================================================
#      2. CALCULATION LOGIC
# ===================================================================

def get_stats(data):
    """Calculates Mean and Standard Uncertainty of the Mean"""
    mean = np.mean(data)
    std_mean = np.std(data, ddof=1) / np.sqrt(len(data))
    return mean, std_mean

def format_si(val, err):
    """Helper to format numbers for LaTeX: 1.23 +/- 0.04"""
    # Determine decimal places based on error
    if err == 0: return f"{val}"
    digits = -int(np.floor(np.log10(err))) 
    if digits < 0: digits = 0
    # We usually want 1 or 2 significant digits in the error
    return f"{val:.{digits+1}f} \\pm {err:.{digits+1}f}"

def calculate_sample(name, m_raw, d_raw, h_raw, M, a, Nz):
    m, dm = get_stats(m_raw)
    d, dd = get_stats(d_raw)
    h, dh = get_stats(h_raw)
    
    # Convert mm to cm for calculations
    d_cm, dd_cm = d / 10.0, dd / 10.0
    h_cm, dh_cm = h / 10.0, dh / 10.0
    
    # Volume V = pi * (d/2)^2 * h
    V_cm3 = np.pi * (d_cm / 2)**2 * h_cm
    # Error propagation
    dV_cm3 = V_cm3 * np.sqrt((2 * dd_cm / d_cm)**2 + (dh_cm / h_cm)**2)
    
    # Single Point Avogadro: Na = (Nz * M * V) / (m * a^3)
    Na = (Nz * M * V_cm3) / (m * a**3)
    dNa = Na * np.sqrt((dV_cm3 / V_cm3)**2 + (dm / m)**2)
    
    # Regression coordinates
    # x = (a^3 * m) / M
    # y = V_zyl
    x_val = (a**3 * m) / M
    dx_val = x_val * (dm / m)
    
    return {
        "name": name,
        "m": (m, dm), "d": (d, dd), "h": (h, dh),
        "V_mm3": (V_cm3 * 1000, dV_cm3 * 1000),
        "Na": (Na, dNa),
        "reg_x": (x_val, dx_val),
        "reg_y": (V_cm3, dV_cm3) 
    }

samples = [
    calculate_sample("Al", m_Al_raw, d_Al_raw, h_Al_raw, M_Al, a_Al, Nz_Al),
    calculate_sample("LiF", m_LiF_raw, d_LiF_raw, h_LiF_raw, M_LiF, a_LiF, Nz_LiF),
    calculate_sample("CaF2", m_CaF2_raw, d_CaF2_raw, h_CaF2_raw, M_CaF2, a_CaF2, Nz_CaF2)
]

# ===================================================================
#      3. LATEX OUTPUT GENERATOR
# ===================================================================
print("-" * 60)
print("COPY & PASTE THESE LINES INTO YOUR LATEX TABLES")
print("-" * 60)

print("\n% --- Table: Messdaten (Masse, Durchmesser, Höhe) ---")
for s in samples:
    line = f"{s['name']:<8} & {format_si(*s['m'])} & {format_si(*s['d'])} & {format_si(*s['h'])} \\\\"
    print(line)

print("\n% --- Table: Einzelwerte (Volumen, Na) ---")
for s in samples:
    # Normalize Na to 10^23
    na_base = s['Na'][0] / 1e23
    na_err = s['Na'][1] / 1e23
    line = f"{s['name']:<8} & {format_si(*s['V_mm3'])} & \\num{{{format_si(na_base, na_err)}e23}} \\\\"
    print(line)

# ===================================================================
#      4. LINEAR REGRESSION (ODR) & PLOTTING
# ===================================================================
x = np.array([s['reg_x'][0] for s in samples])
y = np.array([s['reg_y'][0] for s in samples])
sx = np.array([s['reg_x'][1] for s in samples])
sy = np.array([s['reg_y'][1] for s in samples])

# Linear Model y = A * x + B
def linear_func(p, x): return p[0] * x + p[1]

# ODR Regression (Weights both x and y errors)
model = Model(linear_func)
data = RealData(x, y, sx=sx, sy=sy)
odr = ODR(data, model, beta0=[6e23/4, 0])
out = odr.run()

slope, slope_err = out.beta[0], out.sd_beta[0]
intercept, intercept_err = out.beta[1], out.sd_beta[1]

# Calculate Final Na from Slope (Slope = Na / Nz -> Na = Slope * 4)
# Note: Nz is 4 for all crystals in this experiment
Na_final = slope * 4
dNa_final = slope_err * 4

print(f"\n% --- Final Result (Graphical) ---")
print(f"% Slope: {slope:.4e} +/- {slope_err:.4e}")
print(f"N_A = \\SI{{{format_si(Na_final/1e23, dNa_final/1e23)}e23}}{{\\per\\mole}}")

# Plotting
fig, ax = plt.subplots(figsize=(8, 6))

# Data points
ax.errorbar(x, y, xerr=sx, yerr=sy, fmt='ko', capsize=3, label='Messwerte', ecolor='black', mfc='blue')

# Fit line (extrapolate to 0)
x_fit = np.linspace(0, max(x)*1.1, 100)
y_fit = linear_func(out.beta, x_fit)
ax.plot(x_fit, y_fit, 'r-', label=f'Linearer Fit\n$N_A \\approx {Na_final/1e23:.3f} \\cdot 10^{{23}}$')

# Labels
ax.set_xlabel(r'$\frac{a^3 \cdot m}{M}$ / $cm^3 \cdot mol$')
ax.set_ylabel(r'$V_{Zyl}$ / $cm^3$')
ax.set_title(r'Bestimmung von $N_A$ durch Lineare Regression')
ax.legend()
ax.grid(True, which='both', linestyle='--', alpha=0.7)

# Save as PDF for LaTeX
filename = 'plot_avogadro.pdf'
plt.savefig(filename, bbox_inches='tight')
print(f"\n[INFO] Plot saved as '{filename}'. Upload this file to Overleaf.")
plt.show()