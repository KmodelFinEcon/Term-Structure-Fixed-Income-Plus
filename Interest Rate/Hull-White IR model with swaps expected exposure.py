#hybrid stochastic interest rate HW & Expected Exposure of 2 swaps model implementation by K.Tomov

import numpy as np
import enum
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.integrate as integrate

#global parameters 

# Hull-White model parameters
LAMBDA = 0.3 #speed of mean reversion(short rates)
HWvol = 0.025 # Volatility parameter in Hull-White(from calibrated vol surface stochastic or BSM)

# Swap and simulation parameters
notion  = 11000.0 #Specifies the principal amount underlying the contract, scaling all cash flows and resulting exposures in the swap valuation.
notion2 = 11000.0
Galpha = 0.99 #quantile of distribution ( confidence interval) for PFE
Galpha2 = 0.95 #second variation of confidence interval 
PATHS = 3000 #MC number of sim
STEPS = 1200 #MC number of steps

#ZCB parameters
T_FINAL = 45.0  # end time for ZCB comparison
T_SWAP = 9.0  # maturity date of the swap

# Swap payment dates parameters
TI_SWAP  = 1.0   # beginning of the swap
N_PAYMENTS = 12   # number of swap payments
P0T = lambda T: np.exp(-0.01 * T) # Assumption

#objective functions

def fwd_rate(P0T, t, fd_dt=1e-4): #instantaenous forward rate using finite difference
    return - (np.log(P0T(t + fd_dt)) - np.log(P0T(t - fd_dt))) / (2 * fd_dt)

def HW_theta(lambd, eta, P0T, fd_dt=1e-4): #hull white theta function
    f0 = lambda t: fwd_rate(P0T, t, fd_dt) 
    return lambda t: (1.0 / lambd) * (f0(t + fd_dt) - f0(t - fd_dt)) / (2.0) + \
                     f0(t) + (eta**2 / (2 * lambd**2)) * (1 - np.exp(-2 * lambd * t))

def HW_A(lambd, eta, P0T, T1, T2): #HW Temp1 and Temp2
    tau = T2 - T1
    zGrid = np.linspace(0.0, tau, 250)
    B_r = lambda tau_val: 1.0 / lambd * (np.exp(-lambd * tau_val) - 1.0)
    theta = HW_theta(lambd, eta, P0T)
    temp1 = lambd * integrate.trapezoid(theta(T2 - zGrid) * B_r(zGrid), zGrid)
    temp2 = (eta**2 / (4 * lambd**3)) * (np.exp(-2 * lambd * tau) * (4 * np.exp(lambd * tau) - 1) - 3) \
            + (eta**2 * tau) / (2 * lambd**2)
    return temp1 + temp2

def HW_B(lambd, eta, T1, T2): #HW B function
    return 1.0 / lambd * (np.exp(-lambd * (T2 - T1)) - 1.0)

def HW_ZCB(lambd, eta, P0T, T1, T2, rT1): #pricing ZCB using the HW framework
    if T1 < T2:
        B_r = HW_B(lambd, eta, T1, T2)
        A_r = HW_A(lambd, eta, P0T, T1, T2)
        return np.exp(A_r + B_r * rT1)
    else:
        return np.ones_like(rT1)


def HWMean_r(P0T, lambd, eta, T, fd_dt=1e-4): #short rate mean
    f0 = lambda t: fwd_rate(P0T, t, fd_dt)
    r0 = f0(1e-5)
    theta = HW_theta(lambd, eta, P0T, fd_dt)
    zGrid = np.linspace(0.0, T, 2500)
    temp = lambda z: theta(z) * np.exp(-lambd * (T - z))
    r_mean = r0 * np.exp(-lambd * T) + lambd * integrate.trapezoid(temp(zGrid), zGrid)
    return r_mean


def HW_r_0(P0T, lambd, eta, fd_dt=1e-4): #initial short rate return
    return fwd_rate(P0T, 1e-5, fd_dt)


def HW_Mu_FrwdMeasure(P0T, lambd, eta, T, fd_dt=1e-4): #forward measure drift
    f0 = lambda t: fwd_rate(P0T, t, fd_dt)
    r0 = f0(1e-5)
    theta = HW_theta(lambd, eta, P0T, fd_dt)
    zGrid = np.linspace(0.0, T, 500)
    theta_hat = lambda t, T_val: theta(t) + (eta**2 / lambd**2) * (np.exp(-lambd * (T_val - t)) - 1.0)
    temp = lambda z: theta_hat(z, T) * np.exp(-lambd * (T - z))
    r_mean = r0 * np.exp(-lambd * T) + lambd * integrate.trapezoid(temp(zGrid), zGrid)
    return r_mean


def HWVar_r(lambd, eta, T):
    return (eta**2 / (2 * lambd)) * (1 - np.exp(-2 * lambd * T))


def HWDensity(P0T, lambd, eta, T):
    r_mean = HWMean_r(P0T, lambd, eta, T)
    r_var = HWVar_r(lambd, eta, T)
    return lambda x: st.norm.pdf(x, r_mean, np.sqrt(r_var))


class OptionTypeSwap(enum.Enum):
    RECEIVER = 1.0
    PAYER = -1.0


def HW_SwapPrice(CP, notional, K, t, Ti, Tm, n, r_t, P0T, lambd, eta):
    # Payment grid
    if n == 1:
        ti_grid = np.array([Ti, Tm])
    else:
        ti_grid = np.linspace(Ti, Tm, n)
    tau = ti_grid[1] - ti_grid[0]
    past_payments = ti_grid[ti_grid < t]
    if past_payments.size > 0:
        Ti = past_payments[-1]
    ti_grid = ti_grid[ti_grid > t]
    
    V = np.zeros_like(r_t)
    P_t_TiLambda = lambda T_val: HW_ZCB(lambd, eta, P0T, t, T_val, r_t)
    for ti in ti_grid:
        if ti > Ti:
            V += tau * P_t_TiLambda(ti)
    
    P_t_Ti = P_t_TiLambda(Ti)
    P_t_Tm = P_t_TiLambda(Tm)
    
    if CP == OptionTypeSwap.PAYER:
        swap_val = (P_t_Ti - P_t_Tm) - K * V
    elif CP == OptionTypeSwap.RECEIVER:
        swap_val = K * V - (P_t_Ti - P_t_Tm)
    else:
        raise ValueError("can't figure out the swap type.")
    return swap_val * notional


def GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T, P0T, lambd, eta): #generate short rate paths
    dt_sim = T / NoOfSteps
    fd_dt = 1e-4
    r0 = fwd_rate(P0T, 1e-5, fd_dt)
    time = np.linspace(0, T, NoOfSteps + 1)
    Z = np.random.normal(size=(NoOfPaths, NoOfSteps))
    W = np.zeros((NoOfPaths, NoOfSteps + 1))
    R = np.empty((NoOfPaths, NoOfSteps + 1))
    R[:, 0] = r0
    theta = HW_theta(lambd, eta, P0T, fd_dt)
    
    # Euler simulation
    for i in range(NoOfSteps):
        if NoOfPaths > 1:
            Z_i = (Z[:, i] - Z[:, i].mean()) / Z[:, i].std()
        else:
            Z_i = Z[:, i]
        W[:, i+1] = W[:, i] + np.sqrt(dt_sim) * Z_i
        R[:, i+1] = R[:, i] + lambd * (theta(time[i]) - R[:, i]) * dt_sim + eta * (W[:, i+1] - W[:, i])
    
    return {"time": time, "R": R}

#code execusion

def mainCalculation():
    lambd = LAMBDA
    eta = HWvol 
    notional = notion
    notional2 = notion
    alpha = Galpha
    alpha2 = Galpha2
    
    r0 = HW_r_0(P0T, lambd, eta) #short rate starting point
    
    N = 40 #sim
    Tgrid = np.linspace(0, T_FINAL, N)
    Exact = np.array([P0T(T) for T in Tgrid])
    Proxy = np.array([HW_ZCB(lambd, eta, P0T, 0.0, T, r0) for T in Tgrid])
    
    # Simulation of exposure profiles for a swap using the Hull–White model
    K = 0.02      # Strike rate
    Ti = TI_SWAP  # Swap start
    Tm = T_SWAP   # Swap maturity
    
    paths = GeneratePathsHWEuler(PATHS, STEPS, Tm + 1.0, P0T, lambd, eta)
    r_paths = paths["R"]
    timeGrid = paths["time"]
    dt_sim = timeGrid[1] - timeGrid[0]
    
    # Moneey market account
    M_t = np.exp(np.cumsum(r_paths[:, :-1] * dt_sim, axis=1))
    
    # Calculate exposures (Value, Exposure, Discounted Expected Exposure, and Potential Future Exposure)
    Value = np.empty((PATHS, len(timeGrid)))
    E = np.empty((PATHS, len(timeGrid)))
    EE = np.empty(len(timeGrid))
    PFE = np.empty(len(timeGrid))
    PFE2 = np.empty(len(timeGrid))
    
    for idx, ti in enumerate(timeGrid[:-2]):
        V = HW_SwapPrice(OptionTypeSwap.PAYER, notional, K, timeGrid[idx], Ti, Tm, N_PAYMENTS, r_paths[:, idx], P0T, lambd, eta)
        Value[:, idx] = V
        E[:, idx] = np.maximum(V, 0.0)
        
        # Discounting expected exposure along the Monte Carlo paths
        EE[idx] = np.mean(E[:, idx] / M_t[:, idx])
        PFE[idx] = np.quantile(E[:, idx], alpha)
        PFE2[idx] = np.quantile(E[:, idx], alpha2)
    
    # Portfolio exposure with netting
    ValuePort = np.empty((PATHS, len(timeGrid)))
    EPort = np.empty((PATHS, len(timeGrid)))
    EEPort = np.empty(len(timeGrid))
    PFEPort = np.empty(len(timeGrid))
    
    for idx, ti in enumerate(timeGrid[:-2]):
        Swap1 = HW_SwapPrice(OptionTypeSwap.PAYER, notional, K, timeGrid[idx], Ti, Tm, N_PAYMENTS,r_paths[:, idx], P0T, lambd, eta)
        Swap2 = HW_SwapPrice(OptionTypeSwap.RECEIVER, notional2, 0.0, timeGrid[idx], Tm - 2.0*(Tm - Ti)/N_PAYMENTS, Tm, 1, r_paths[:, idx], P0T, lambd, eta)
        VPort = Swap1 + Swap2
        ValuePort[:, idx] = VPort
        EPort[:, idx] = np.maximum(VPort, 0.0)
        EEPort[idx] = np.mean(EPort[:, idx] / M_t[:, idx])
        PFEPort[idx] = np.quantile(EPort[:, idx], alpha)
    
    #graphical output
    
    plt.figure(figsize=(8, 6))
    plt.plot(timeGrid, Value[:100, :].T)
    plt.grid(True)
    plt.xlabel('T')
    plt.ylabel('SV')
    plt.title('Swap Value in defined paths')
    
    plt.figure(figsize=(8, 6))
    plt.plot(timeGrid, E[:100, :].T, 'r')
    plt.grid(True)
    plt.xlabel('T')
    plt.ylabel('Positive Exposure')
    plt.title('Positive Exposure in defined paths')
    
    plt.figure(figsize=(8, 6))
    plt.plot(timeGrid, EE, 'r', label='EE')
    plt.plot(timeGrid, PFE, 'k', label='PFE')
    plt.plot(timeGrid, PFE2, '--b', label='PFEsecondvariant')
    plt.grid(True)
    plt.xlabel('Time')
    plt.ylabel('Exposure')
    plt.title('Discounted EE & PFE')
    plt.legend()
    
    plt.figure(figsize=(8, 6))
    plt.plot(timeGrid, EEPort, 'r', label='EE - Portfolio')
    plt.plot(timeGrid, PFEPort, 'k', label='PFE - Portfolio')
    plt.grid(True)
    plt.xlabel('Time')
    plt.title('Exposure for a Portfolio of Two Swaps')
    plt.legend()
    
    plt.figure(figsize=(8, 6))
    plt.plot(timeGrid, EE, 'r', label='EE Swap')
    plt.plot(timeGrid, EEPort, '--r', label='EE Portfolio')
    plt.grid(True)
    plt.title('Comparison of EEs')
    plt.legend()
    
    plt.figure(figsize=(8, 6))
    plt.plot(timeGrid, PFE, 'k', label='PFE Swap')
    plt.plot(timeGrid, PFEPort, '--k', label='PFE Portfolio')
    plt.grid(True)
    plt.title('Comparison of PFEs')
    plt.legend()
    plt.show()

if __name__ == '__main__':
    mainCalculation()
