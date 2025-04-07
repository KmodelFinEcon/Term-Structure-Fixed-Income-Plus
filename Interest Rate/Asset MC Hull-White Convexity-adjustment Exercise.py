#derivative asset Libor rate convexity-correction exercise by K.Tomov taken from mathematical modeling in computational finance by C. W. Oosterlee and Lech A Grzelak

#summary:
#the model visualizes the effect of the convexity correction as a function of volatility (sigma), and compares the derivative’s market price with the convexity-corrected price.
#the code attempts to explain how to price interest rate products under hull-white framework from both analyticial and MC approach and take into account IR convexity/ vol and mean reversion

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

# Global Model Parameters
LAMBDA    = 0.03       # Mean reversion parameter
ETA       = 0.05       # Volatility parameter
paths1 = 15000     # Monte Carlo simulation paths
steps = 1400      # time-steps for simulation
numdif   = 0.0002      # Time-step for numerical differentiation
EPS       = 1e-6         # Small time to initialize forward rate calculation
P0T = lambda T: np.exp(-0.1 * T) 

# Euler discretization under Hull-White framework

def GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T, P0T_func, lambd, eta):
    # Numerical differentiation function for the instantaneous forward rate
    f0T = lambda t: - (np.log(P0T_func(t + numdif)) - np.log(P0T_func(t - numdif))) / (2 * numdif)
    r0 = f0T(EPS) #initial forward rate
    theta = lambda t: 1.0 / lambd * (f0T(t + numdif) - f0T(t - numdif)) / (2 * numdif) \
                        + f0T(t) \
                        + eta**2 / (2 * lambd**2) * (1.0 - np.exp(-2 * lambd * t))
    Z = np.random.normal(0.0, 1.0, [NoOfPaths, NoOfSteps])
    W = np.zeros([NoOfPaths, NoOfSteps + 1]) #array for integration 1
    R = np.zeros([NoOfPaths, NoOfSteps + 1]) #array for integration 2
    R[:, 0] = r0
    time = np.zeros(NoOfSteps + 1)
    
    dt_sim = T / float(NoOfSteps)
    for i in range(NoOfSteps):
        if NoOfPaths > 1:
            Z[:, i] = (Z[:, i] - np.mean(Z[:, i])) / np.std(Z[:, i])
        W[:, i + 1] = W[:, i] + np.sqrt(dt_sim) * Z[:, i]
        R[:, i + 1] = R[:, i] + lambd * (theta(time[i]) - R[:, i]) * dt_sim + eta * (W[:, i + 1] - W[:, i])
        time[i + 1] = time[i] + dt_sim

    return {"time": time, "R": R}

def HW_theta(lambd, eta, P0T_func):
    f0T = lambda t: - (np.log(P0T_func(t + numdif)) - np.log(P0T_func(t - numdif))) / (2 * numdif)
    theta = lambda t: 1.0 / lambd * (f0T(t + numdif) - f0T(t - numdif)) / (2 * numdif) \
                        + f0T(t) \
                        + eta**2 / (2 * lambd**2) * (1.0 - np.exp(-2 * lambd * t))
                        
    return theta

def HW_A(lambd, eta, P0T_func, T1, T2):
    tau = T2 - T1
    zGrid = np.linspace(0.0, tau, 250)  
    B_r = lambda tau_val: 1.0 / lambd * (np.exp(-lambd * tau_val) - 1.0)
    theta = HW_theta(lambd, eta, P0T_func)
    
    temp1 = lambd * integrate.simpson(theta(T2 - zGrid) * B_r(zGrid), zGrid)
    temp2 = eta**2 / (4.0 * lambd**3) * (np.exp(-2 * lambd * tau) * (4 * np.exp(lambd * tau) - 1.0) - 3.0) \
            + eta**2 * tau / (2 * lambd**2)
    
    return temp1 + temp2

def HW_B(lambd, eta, T1, T2):
    
    return 1.0 / lambd * (np.exp(-lambd * (T2 - T1)) - 1.0)

# ZCB under the Hull-White model
def HW_ZCB(lambd, eta, P0T_func, T1, T2, rT1):
    B_r = HW_B(lambd, eta, T1, T2)
    A_r = HW_A(lambd, eta, P0T_func, T1, T2)
    
    return np.exp(A_r + B_r * rT1)

def HWMean_r(P0T_func, lambd, eta, T):
    f0T = lambda t: - (np.log(P0T_func(t + numdif)) - np.log(P0T_func(t - numdif))) / (2 * numdif)
    r0 = f0T(EPS)
    theta = HW_theta(lambd, eta, P0T_func)
    zGrid = np.linspace(0.0, T, 2500)
    temp = lambda z: theta(z) * np.exp(-lambd * (T - z))
    r_mean = r0 * np.exp(-lambd * T) + lambd * integrate.simpson(temp(zGrid), zGrid)
    
    return r_mean

def HW_r_0(P0T_func, lambd, eta):
    f0T = lambda t: - (np.log(P0T_func(t + numdif)) - np.log(P0T_func(t - numdif))) / (2 * numdif)
    r0 = f0T(EPS)
    
    return r0

#main computation

def mainCalculation():
    lambd = LAMBDA
    eta = ETA
    
    r0 = HW_r_0(P0T, lambd, eta)
    
    N = 40
    T_end = 34
    Tgrid = np.linspace(0, T_end, N)
    
    Exact = np.zeros(N)
    Proxy = np.zeros(N)
    for i, Ti in enumerate(Tgrid):
        Proxy[i] = HW_ZCB(lambd, eta, P0T, 0.0, Ti, r0)
        Exact[i] = P0T(Ti)
    
    plt.figure(1)
    plt.grid(True)
    plt.plot(Tgrid, Exact, '-k', label="Analytical ZCB")
    plt.plot(Tgrid, Proxy, '--r', label="Monte Carlo ZCB")
    plt.legend()
    plt.title('P(0,T): Market vs. Analytical Expression')
    
    # Libor period to measure the convexity effect>>>
    T1 = 4.0
    T2 = 8.0
    
    paths = GeneratePathsHWEuler(paths1, steps, T1, P0T, lambd, eta)
    r = paths["R"]
    timeGrid = paths["time"]
    dt_sim = timeGrid[1] - timeGrid[0]
    
    M_t = np.exp(np.cumsum(r[:, :-1] * dt_sim, axis=1))
    
    P_T1_T2 = HW_ZCB(lambd, eta, P0T, T1, T2, r[:, -1])
    L_T1_T2 = 1.0 / (T2 - T1) * (1.0 / P_T1_T2 - 1)
    MC_Result = np.mean(1 / M_t[:, -1] * L_T1_T2)
    print('MC Price = {0:.6f}'.format(MC_Result))
    
    L_T0_T1_T2 = 1.0 / (T2 - T1) * (P0T(T1) / P0T(T2) - 1.0)
    
    # Define the convexity correction function
    cc = lambda sigma: P0T(T2) * (L_T0_T1_T2 + (T2 - T1) * L_T0_T1_T2**2 * np.exp(sigma**2 * T1)) - L_T0_T1_T2
    
    sigma = 0.4  #size of convexity correction
    
    print('Price without convexity correction: {0:.6f}'.format(L_T0_T1_T2))
    print('Price with convexity correction {0:.2f}: {1:.6f}'.format(sigma, L_T0_T1_T2 + cc(sigma)))
    
    plt.figure(2)
    sigma_range = np.linspace(0.0, 0.6, 100)
    plt.plot(sigma_range, cc(sigma_range))
    plt.grid(True)
    plt.xlabel('sigma')
    plt.ylabel('CC')
    plt.title('CC vs. sigma')
    
    plt.figure(3)
    plt.plot(sigma_range, MC_Result * np.ones_like(sigma_range), label='Market Price')
    plt.plot(sigma_range, L_T0_T1_T2 + cc(sigma_range), '--r', label='Price with convexity adjustment')
    plt.grid(True)
    plt.xlabel('sigma')
    plt.ylabel('derivative val')
    plt.legend()
    plt.title('Derivative Price Compared')
    plt.show()

if __name__ == '__main__':
    mainCalculation()
