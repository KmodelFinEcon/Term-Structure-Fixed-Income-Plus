#### CIR Monte Carlo model and Bond price/YTM/Convexity ####
#           by K.Tomov

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Parameters
k = 0.1       # Speed of mean reversion (NEEDS MARKET CALIBRATION)
theta = 0.045  # Long-term mean interest rate (NEEDS MARKET CALIBRATION)
sigma = 0.01   # Volatility (NEEDS MARKET CALIBRATION)
r0 = 0.043     # Initial interest rate
T = 1         # Time horizon (years)
N = 1000       # Number of time steps
t = 0         # Current time (e.g., today)
dt = T / N
fv = 100  # Face Value of Bond

num_paths_plot = 10   # Number of paths to plot
n_scenarios = 3500   # Monte Carlo simulations

# Vectorized CIR simulation for multiple paths
def simulate_cir_paths(k, theta, sigma, r0, T, N, n_paths):
    dt = T / N
    paths = np.empty((n_paths, N + 1))
    paths[:, 0] = r0
    for t in range(1, N + 1):
        r_prev = paths[:, t - 1]
        Z = np.random.randn(n_paths)
        paths[:, t] = r_prev + k * (theta - r_prev) * dt \
                      + sigma * np.sqrt(dt) * np.sqrt(np.maximum(r_prev, 0)) * Z
    return paths

# OLS estimation function for one simulation path
def ols_cir(data, dt):
    rs = data[:-1]
    rt = data[1:]
    y = (rt - rs) / np.sqrt(np.maximum(rs, 1e-6))
    z1 = dt / np.sqrt(np.maximum(rs, 1e-6))
    z2 = np.sqrt(np.maximum(rs, 1e-6)) * dt
    X = np.column_stack((z1, z2))
    model = LinearRegression(fit_intercept=False).fit(X, y)
    y_hat = model.predict(X)
    residuals = y - y_hat
    beta1, beta2 = model.coef_
    k_est = -beta2
    theta_est = beta1 / k_est
    sigma_est = np.std(residuals) / np.sqrt(dt)
    return k_est, theta_est, sigma_est

np.random.seed(123)

# Simulate paths for plotting
paths_plot = simulate_cir_paths(k, theta, sigma, r0, T, N, num_paths_plot)
time = np.linspace(0, T, N + 1)

plt.figure(figsize=(10, 6))
for path in paths_plot:
    plt.plot(time, path, lw=1)
plt.title(f"CIR Model: {num_paths_plot} Simulated Paths")
plt.xlabel("Time (Years)")
plt.ylabel("Interest Rate")
plt.grid(alpha=0.5)
plt.show()

# Simulate all scenarios for Monte Carlo estimation
paths = simulate_cir_paths(k, theta, sigma, r0, T, N, n_scenarios)

# Compute OLS estimates for each simulation path
ols_estimates = np.array([ols_cir(paths[i], dt) for i in range(n_scenarios)])
mean_estimates = ols_estimates.mean(axis=0)

print(f"The theoretical parameters are: k={k}, theta={theta}, sigma={sigma}")
print(f"Average estimates over {n_scenarios} simulations: "
      f"k={mean_estimates[0]:.3f}, theta={mean_estimates[1]:.3f}, sigma={mean_estimates[2]:.3f}")

KMC = max(float(mean_estimates[0]), 1e-6)
thetaMC = max(abs(float(mean_estimates[1])), 1e-6)
sigmamc = max(float(mean_estimates[2]), 1e-6)

# calculation of the zero-coupon bond price under the CIR model
def cir_bond_price(KMC, thetaMC, sigmamc, r0, t, T):
    gamma = np.sqrt(KMC**2 + 2 * sigmamc**2)
    numerator_B = 2 * (np.exp(gamma * (T - t)) - 1)
    denominator_B = (gamma + KMC) * (np.exp(gamma * (T - t)) - 1) + 2 * gamma
    B = numerator_B / denominator_B

    numerator_A = 2 * gamma * np.exp((KMC + gamma) * (T - t) / 2)
    denominator_A = (gamma + KMC) * (np.exp(gamma * (T - t)) - 1) + 2 * gamma
    A = (numerator_A / denominator_A) ** (2 * KMC * thetaMC / sigmamc**2)

    bond_price = A * np.exp(-B * r0)
    return bond_price

#bond price from CIR
print("Bond price from CIR:", cir_bond_price(KMC, thetaMC, sigmamc, r0, t, T))

# Getting the YTM from the CIR model
def cir_ytm(KMC, thetaMC, sigmamc, r0, t, T):
    bond_price = cir_bond_price(KMC, thetaMC, sigmamc, r0, t, T)
    ytm = -np.log(bond_price) / (T - t)
    return ytm
ytms = max(cir_ytm(KMC, thetaMC, sigmamc, r0, t, T), 1e-6)
print("Yield to Maturity (YTM):", ytms)

# Bond convexity
def zcb_convexity(fv, ytms, T):
    convexity = T * (T + 1) / (1 + ytms)**2
    return convexity

print("Zero Coupon Bond Convexity:", zcb_convexity(fv, ytms, T))