
####HULL WHITE SINGLE FACTOR INTEREST RATE MODEL #### QUANT LIB
#by KT.

import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np

# Parameters
sigma = 0.1
a = 0.1
timestep = 360
length = 30  # in years
forward_rate = 0.05
day_count = ql.Actual360()
todays_date = ql.Date(15, 1, 2024)

ql.Settings.instance().evaluationDate = todays_date

# Spot curve setup
spot_curve = ql.FlatForward(todays_date, ql.QuoteHandle(ql.SimpleQuote(forward_rate)), day_count)
spot_curve_handle = ql.YieldTermStructureHandle(spot_curve)

# Hull-White process
hw_process = ql.HullWhiteProcess(spot_curve_handle, a, sigma)
rng = ql.GaussianRandomSequenceGenerator(ql.UniformRandomSequenceGenerator(timestep, ql.UniformRandomGenerator()))
seq = ql.GaussianPathGenerator(hw_process, length, timestep, rng, False)

# Path generation
def generate_paths(num_paths, timestep):
    arr = np.zeros((num_paths, timestep + 1))
    for i in range(num_paths):
        sample_path = seq.next()
        path = sample_path.value()
        time = [path.time(j) for j in range(len(path))]
        value = [path[j] for j in range(len(path))]
        arr[i, :] = np.array(value)
    return np.array(time), arr

# Short rate simulation
num_paths = 10
time, paths = generate_paths(num_paths, timestep)
for i in range(num_paths):
    plt.plot(time, paths[i, :], lw=0.8, alpha=0.6)
plt.title("Hull-White Short Rate Simulation")
plt.xlabel("Time")
plt.ylabel("Short Rate")
plt.show()

# Simulation for variance and mean
num_paths = 1000
time, paths = generate_paths(num_paths, timestep)

# Variance of short rates
vol = [np.var(paths[:, i]) for i in range(timestep + 1)]
plt.plot(time, vol, "r-.", lw=3, alpha=0.6, label="Simulated Variance")
plt.plot(time, sigma * sigma / (2 * a) * (1.0 - np.exp(-2.0 * a * np.array(time))), "b-", lw=2, alpha=0.5, label="Theoretical Variance")
plt.title("Variance of Short Rates")
plt.xlabel("Time")
plt.ylabel("Variance")
plt.legend()
plt.show()

# Mean of short rates
def alpha(forward, sigma, a, t):
    return forward + 0.5 * np.power(sigma / a * (1.0 - np.exp(-a * t)), 2)

avg = [np.mean(paths[:, i]) for i in range(timestep + 1)]
plt.plot(time, avg, "r-.", lw=3, alpha=0.6, label="Simulated Mean")
plt.plot(time, alpha(forward_rate, sigma, a, time), "b-", lw=2, alpha=0.6, label="Theoretical Mean")
plt.title("Mean of Short Rates")
plt.xlabel("Time")
plt.ylabel("Mean")
plt.legend()
plt.show()