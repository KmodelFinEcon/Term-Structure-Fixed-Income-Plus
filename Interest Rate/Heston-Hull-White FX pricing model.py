#Hybrid Heston Hull-White COS and MC for FX pricing model (single strike) from Mathematical Modeling for Computational Finance textbook by C. W. Oosterlee and Lech A Grzelak
#           Implementation and optimization by. Ktomov

import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import scipy.special as sp
import scipy.integrate as integrate
import scipy.optimize as optimize
import enum

#global parameters

# Macro-economic variables
T = 4.0                             # Maturity
y0 = 1.25                           #Spot FX rate
rdomestic = 0.04                    # Domestic interest rate
rforeign = 0.02                    # Foreign interest rate

#Monte-Carlo simulation
PATHS = 500
STEPS = int(T * 100)

# COS method simulation
NEXPANSION = 300 
LTRUNC = 8 #cos method truncation

# Heston assumptions
kappa = 0.5
gamma = 0.3
vbar = 0.1
v0 = 0.1

# Hull–White model assumptions
lambdd = 0.01 
lambdf = 0.05
etad = 0.007
etaf = 0.012

# Correlations (very complex calibration ONG!)
Rxv   = -0.3
Rxrd  = -0.07
Rxrf  = -0.03
Rvrd  = 0.4
Rvrf  = 0.2
Rrdrf = 0.14

# ZCB functions and forward FX rate
P0Td = lambda t: np.exp(-rdomestic * t)
P0Tf = lambda t: np.exp(-rforeign * t)
frwdFX = y0 * P0Tf(T) / P0Td(T)

#option type as class for eary access
class OptionType(enum.Enum):
    CALL = 1.0
    PUT = -1.0

# Implied vol and BS
def BS_Call_Put_Option_Price(CP, S_0, K, sigma, tau, r):
    if isinstance(K, list):
        K = np.array(K).reshape([-1, 1])
    d1 = (np.log(S_0 / K) + (r + 0.5 * sigma**2) * tau) / (sigma * np.sqrt(tau))
    d2 = d1 - sigma * np.sqrt(tau)
    if CP == OptionType.CALL:
        value = st.norm.cdf(d1) * S_0 - st.norm.cdf(d2) * K * np.exp(-r * tau)
    elif CP == OptionType.PUT:
        value = st.norm.cdf(-d2) * K * np.exp(-r * tau) - st.norm.cdf(-d1) * S_0
    return value

def ImpliedVolatilityBlack76(CP, marketPrice, K, T, S_0): #using the Black 76 framework
    sigmaGrid = np.linspace(0.001, 5.0, 5000)
    optPriceGrid = BS_Call_Put_Option_Price(CP, S_0, K, sigmaGrid, T, 0.0)
    sigmaInitial = np.interp(marketPrice, optPriceGrid.flatten(), sigmaGrid)
    func = lambda sigma: BS_Call_Put_Option_Price(CP, S_0, K, sigma, T, 0.0) - marketPrice
    impliedVol = optimize.newton(func, sigmaInitial, tol=1e-15)
    if impliedVol > 2.0:
        impliedVol = 0.0
    return impliedVol

#The COS method
def CallPutOptionPriceCOSMthd_StochIR(cf, CP, S0, tau, K, N, L, P0T):
    if not isinstance(K, np.ndarray):
        K = np.array(K).reshape([-1, 1])
    i = np.complex128(0.0 + 1.0j)
    x0 = np.log(S0 / K)
    a = -L * np.sqrt(tau)
    b = L * np.sqrt(tau)
    k = np.arange(N).reshape([N, 1])
    u = k * np.pi / (b - a)
    H_k = CallPutCoefficients(OptionType.PUT, a, b, k)
    mat = np.exp(i * (x0 - a) * u.T)
    temp = cf(u) * H_k
    temp[0] = 0.5 * temp[0]
    value = K * np.real(np.dot(mat, temp))
    
    #call put parity for call options
    if CP == OptionType.CALL:
        value = value + S0 - K * P0T
    return value

def CallPutCoefficients(CP, a, b, k):
    if CP == OptionType.CALL:
        c = 0.0
        d = b
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        if a < b and b < 0.0:
            H_k = np.zeros((len(k), 1))
        else:
            H_k = 2.0 / (b - a) * (Chi_k - Psi_k)
    elif CP == OptionType.PUT:
        c = a
        d = 0.0
        coef = Chi_Psi(a, b, c, d, k)
        Chi_k = coef["chi"]
        Psi_k = coef["psi"]
        H_k = 2.0 / (b - a) * (-Chi_k + Psi_k)
    return H_k

def Chi_Psi(a, b, c, d, k):
    psi = np.sin(k * np.pi * (d - a) / (b - a)) - np.sin(k * np.pi * (c - a) / (b - a))
    psi[1:] = psi[1:] * (b - a) / (k[1:] * np.pi)
    psi[0] = d - c
    chi = 1.0 / (1.0 + (k * np.pi / (b - a))**2)
    expr1 = np.cos(k * np.pi * (d - a) / (b - a)) * np.exp(d) - np.cos(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    expr2 = (k * np.pi / (b - a)) * np.sin(k * np.pi * (d - a) / (b - a)) - (k * np.pi / (b - a)) * np.sin(k * np.pi * (c - a) / (b - a)) * np.exp(c)
    chi = chi * (expr1 + expr2)
    return {"chi": chi, "psi": psi}

#monte-carlo foreign exchange pricing

def EUOptionPriceFromMCPathsGeneralizedFXFrwd(CP, S, K):
    result = np.zeros((len(K), 1))
    if CP == OptionType.CALL:
        for idx, k in enumerate(K):
            result[idx] = np.mean(np.maximum(S - k, 0.0))
    elif CP == OptionType.PUT:
        for idx, k in enumerate(K):
            result[idx] = np.mean(np.maximum(k - S, 0.0))
    return result

def GeneratePathsHHWFXHWEuler(NoOfPaths, NoOfSteps, T, frwdFX, v0, vbar, kappa, gamma, lambdd, lambdf, etad, etaf, rhoxv, rhoxrd, rhoxrf, rhovrd, rhovrf, rhordrf):
    Wx = np.zeros((NoOfPaths, NoOfSteps+1))
    Wv = np.zeros((NoOfPaths, NoOfSteps+1))
    Wrd = np.zeros((NoOfPaths, NoOfSteps+1))
    Wrf = np.zeros((NoOfPaths, NoOfSteps+1))
    V = np.zeros((NoOfPaths, NoOfSteps+1))
    FX = np.zeros((NoOfPaths, NoOfSteps+1))
    
    V[:, 0] = v0
    FX[:, 0] = frwdFX
    dt = T / NoOfSteps
    
    Bd = lambda t, T_val: (np.exp(-lambdd * (T_val - t)) - 1.0) / lambdd
    Bf = lambda t, T_val: (np.exp(-lambdf * (T_val - t)) - 1.0) / lambdf
    
    # Covariance matrix creation
    cov = np.array([[1.0,    rhoxv,  rhoxrd, rhoxrf],
                    [rhoxv,  1.0,    rhovrd, rhovrf],
                    [rhoxrd, rhovrd, 1.0,    rhordrf],
                    [rhoxrf, rhovrf, rhordrf, 1.0]])
    
    time = np.zeros(NoOfSteps+1)
    
    for i in range(NoOfSteps):
        Z = np.random.multivariate_normal(np.zeros(4), cov, NoOfPaths)
        Z = (Z - Z.mean(axis=0)) / Z.std(axis=0)
        
        Wx[:, i+1] = Wx[:, i] + np.sqrt(dt) * Z[:, 0]
        Wv[:, i+1] = Wv[:, i] + np.sqrt(dt) * Z[:, 1]
        Wrd[:, i+1] = Wrd[:, i] + np.sqrt(dt) * Z[:, 2]
        Wrf[:, i+1] = Wrf[:, i] + np.sqrt(dt) * Z[:, 3]
        
        # Variance process (Euler discretization)
        V[:, i+1] = V[:, i] + kappa * (vbar - V[:, i]) * dt \
                    + gamma * rhovrd * etad * Bd(time[i], T) * np.sqrt(V[:, i]) * dt \
                    + gamma * np.sqrt(V[:, i]) * (Wv[:, i+1] - Wv[:, i])
        V[:, i+1] = np.maximum(V[:, i+1], 0.0)
        
        # FX process under the forward measure
        FX[:, i+1] = FX[:, i] * (1.0 + np.sqrt(V[:, i]) * (Wx[:, i+1] - Wx[:, i])
                                 - etad * Bd(time[i], T) * (Wrd[:, i+1] - Wrd[:, i])
                                 + etaf * Bf(time[i], T) * (Wrf[:, i+1] - Wrf[:, i]))
        time[i+1] = time[i] + dt
    
    return {"time": time, "FX": FX}

#From reference calculation of heston-hull-white (Mathematical modeling for computation finance by C. W. Oosterlee and Lech A Grzelak )

def meanSqrtV_3(kappa, v0, vbar, gamma):
    delta = 4.0 * kappa * vbar / (gamma**2)
    c = lambda t: gamma**2 / (4.0 * kappa) * (1.0 - np.exp(-kappa * t))
    kappaBar = lambda t: 4.0 * kappa * v0 * np.exp(-kappa * t) / (gamma**2 * (1.0 - np.exp(-kappa * t)))
    temp1 = lambda t: np.sqrt(2.0 * c(t)) * sp.gamma((1.0 + delta) / 2.0) / sp.gamma(delta / 2.0) * sp.hyp1f1(-0.5, delta / 2.0, -kappaBar(t) / 2.0)
    return temp1

def C_H1HW_FX(u, tau, kappa, gamma, rhoxv):
    i = np.complex128(0.0 + 1.0j)
    D1 = np.sqrt((kappa - gamma * rhoxv * i * u)**2 + (u**2 + i*u) * gamma**2)
    g = (kappa - gamma * rhoxv * i * u - D1) / (kappa - gamma * rhoxv * i * u + D1)
    C = (1.0 - np.exp(-D1 * tau)) / (gamma**2 * (1.0 - g * np.exp(-D1 * tau))) * (kappa - gamma * rhoxv * i * u - D1)
    return C

def ChFH1HW_FX(u, tau, gamma, Rxv, Rxrd, Rxrf, Rrdrf, Rvrd, Rvrf,
                lambdd, etad, lambdf, etaf, kappa, vbar, v0):
    i = np.complex128(0.0 + 1.0j)
    C_func = lambda u_val, tau_val: C_H1HW_FX(u_val, tau_val, kappa, gamma, Rxv)
    Bd = lambda t, T_val: (np.exp(-lambdd * (T_val - t)) - 1.0) / lambdd
    Bf = lambda t, T_val: (np.exp(-lambdf * (T_val - t)) - 1.0) / lambdf
    G = meanSqrtV_3(kappa, v0, vbar, gamma)
    
    zeta = lambda t: (Rxrd * etad * Bd(t, tau) - Rxrf * etaf * Bf(t, tau)) * G(t) \
                     + Rrdrf * etad * etaf * Bd(t, tau) * Bf(t, tau) \
                     - 0.5 * (etad**2 * Bd(t, tau)**2 + etaf**2 * Bf(t, tau)**2)
    
    N_integration = 500
    z = np.linspace(1e-10, tau - 1e-10, N_integration)
    
    temp1 = lambda z_val: kappa * vbar + Rvrd * gamma * etad * G(tau - z_val) * Bd(tau - z_val, tau)
    temp2 = lambda z_val, u_val: -Rvrd * gamma * etad * G(tau - z_val) * Bd(tau - z_val, tau) * i * u_val
    temp3 = lambda z_val, u_val: Rvrf * gamma * etaf * G(tau - z_val) * Bf(tau - z_val, tau) * i * u_val
    f = lambda z_val, u_val: (temp1(z_val) + temp2(z_val, u_val) + temp3(z_val, u_val)) * C_func(u_val, z_val)
    
    # Integration for each u value
    int1 = np.array([integrate.trapezoid(f(z, u_val).real, z) + 1j * integrate.trapezoid(f(z, u_val).imag, z)
                     for u_val in u.flatten()]).reshape(u.shape)
    int2 = (u**2 + i*u) * integrate.trapezoid(zeta(tau - z), z)
    A = int1 + int2
    cf = np.exp(A + v0 * C_func(u, tau))
    return cf

#strike array

def GenerateStrikes(frwd, Ti):
    c_n = np.array([-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5])
    return frwd * np.exp(0.1 * c_n * np.sqrt(Ti))


#main execution

def mainCalculation():
    CP = OptionType.CALL
    K = GenerateStrikes(frwdFX, T)
    K = np.array(K).reshape([-1, 1])
    
    # Monte Carlo: run simulations over different seeds
    SeedV = range(200)
    optMCM = np.zeros((len(SeedV), len(K)))
    
    for idx, seed in enumerate(SeedV):
        np.random.seed(seed)
        paths = GeneratePathsHHWFXHWEuler(PATHS, STEPS, T, frwdFX, v0, vbar, kappa, gamma, lambdd, lambdf, etad, etaf, Rxv, Rxrd, Rxrf, Rvrd, Rvrf, Rrdrf)
        frwdfxT = paths["FX"]
        optMC = P0Td(T) * EUOptionPriceFromMCPathsGeneralizedFXFrwd(CP, frwdfxT[:, -1], K)
        optMCM[idx, :] = optMC.squeeze()
    
    optionMC_E = np.mean(optMCM, axis=0)
    optionMC_StDev = np.std(optMCM, axis=0)
    
    cf = lambda u: ChFH1HW_FX(u, T, gamma, Rxv, Rxrd, Rxrf, Rrdrf,
                                Rvrd, Rvrf, lambdd, etad, lambdf, etaf,
                                kappa, vbar, v0)
    valCOS_H1HW = P0Td(T) * CallPutOptionPriceCOSMthd_StochIR(cf, CP, frwdFX, T, K, NEXPANSION, LTRUNC, 1.0)
    
    EyT = P0Td(T)/P0Tf(T) * EUOptionPriceFromMCPathsGeneralizedFXFrwd(CP, frwdfxT[:, -1], [0.0])
    print("Martingale validation: = {:.4f} & y0 = {}".format(EyT[0][0], y0))
    print("Maturity chosen to (T) = {}".format(T))
    
    for idx, k in enumerate(K):
        print("Option price for given strike (K) = {:.4f} and for COS method = {:.4f}, MC = {:.4f}, stdDev = {:.4f}"
              .format(k[0], valCOS_H1HW[idx][0], optionMC_E[idx], optionMC_StDev[idx]))
    
    # Plot option prices
    plt.figure(1)
    plt.plot(K, optionMC_E, label='MC Option Price')
    plt.plot(K, valCOS_H1HW, '--r', label='COS approach')
    plt.grid()
    plt.legend()
    plt.title("FX Option Price list")
    
    IVCos = np.zeros((len(K), 1))
    IVMC = np.zeros((len(K), 1))
    for idx, k in enumerate(K):
        priceCOS = valCOS_H1HW[idx] / P0Td(T)
        IVCos[idx] = ImpliedVolatilityBlack76(CP, priceCOS, k, T, frwdFX) * 100.0
        priceMC = optionMC_E[idx] / P0Td(T)
        IVMC[idx] = ImpliedVolatilityBlack76(CP, priceMC, k, T, frwdFX) * 100.0

    plt.figure(2)
    plt.plot(K, IVMC, label='Implied volatility MC')
    plt.plot(K, IVCos, label='Implied volatility COS')
    plt.grid()
    plt.legend()
    plt.title("FX Implied Volatilities")
    
    plt.show()

if __name__ == "__main__":
    mainCalculation()
