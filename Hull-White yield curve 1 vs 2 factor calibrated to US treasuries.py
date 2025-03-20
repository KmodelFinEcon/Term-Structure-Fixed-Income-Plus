### Analytical vs MC HULL-WHITE Yield curve construction 1 factor vs 2 factor model calibrated on US Treasury Yields ###
#                       by K.Tomov

import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy import interpolate
import yfinance as yf

#retrieving treasury data

tickers = {
    '^IRX': 0.25,  # 13-week T-bill (~3 months)
    '^FVX': 5.0,   # 5-year Treasury yield
    '^TNX': 10.0,  # 10-year Treasury yield
    '^TYX': 30.0   # 30-year Treasury yield
}
yields = {}
for ticker, maturity in tickers.items():
    data = yf.Ticker(ticker).history(period="5y")
    if not data.empty:
        latest_yield = data['Close'].iloc[-1] / 100
        yields[maturity] = latest_yield

maturities = np.array(sorted(yields.keys()))
yields_arr = np.array([yields[m] for m in maturities])
Pi = np.exp(-yields_arr * maturities) #Convert Yields to Discount Factors (Zero-Coupon Prices) Assuming continuous compounding

# --- Print the Data Arrays ---
print("Maturity times=", list(maturities))
print("Discount factor=", list(Pi))

ti = list(maturities)
pi = list(Pi)

DT_DERIV = 0.01 # Used a fixed dt for derivative approximation in f0T

def f0T(t, P0T):
    return - (np.log(P0T(t + DT_DERIV)) - np.log(P0T(t - DT_DERIV))) / (2 * DT_DERIV) # central difference approximation

def GeneratePathsHW2FEuler(NoOfPaths, NoOfSteps, T, P0T, lambd1, lambd2, eta1, eta2, rho):
    dt = T / NoOfSteps
    time = np.linspace(0, T, NoOfSteps + 1)

    phi_vals = (np.array([f0T(t, P0T) for t in time]) +
                (eta1**2) / (2 * lambd1**2) * (1 - np.exp(-lambd1 * time))**2 +
                (eta2**2) / (2 * lambd2**2) * (1 - np.exp(-lambd2 * time))**2 +
                rho * eta1 * eta2 / (lambd1 * lambd2) * (1 - np.exp(-lambd1 * time)) * (1 - np.exp(-lambd2 * time)))

    Z1 = np.random.normal(0.0, 1.0, (NoOfPaths, NoOfSteps))
    Z2 = np.random.normal(0.0, 1.0, (NoOfPaths, NoOfSteps))
    Z2 = rho * Z1 + np.sqrt(1 - rho**2) * Z2

    # Brownian motion implementation
    dW1 = np.sqrt(dt) * Z1
    dW2 = np.sqrt(dt) * Z2
    W1 = np.hstack([np.zeros((NoOfPaths, 1)), np.cumsum(dW1, axis=1)])
    W2 = np.hstack([np.zeros((NoOfPaths, 1)), np.cumsum(dW2, axis=1)])

    X = np.zeros((NoOfPaths, NoOfSteps + 1))
    Y = np.zeros((NoOfPaths, NoOfSteps + 1))
    R = np.empty((NoOfPaths, NoOfSteps + 1))
    R[:, 0] = phi_vals[0]

    # Euler scheme looped
    for i in range(NoOfSteps):
        X[:, i + 1] = X[:, i] - lambd1 * X[:, i] * dt + eta1 * dW1[:, i]
        Y[:, i + 1] = Y[:, i] - lambd2 * Y[:, i] * dt + eta2 * dW2[:, i]
        R[:, i + 1] = X[:, i + 1] + Y[:, i + 1] + phi_vals[i + 1]

    paths = {"time": time, "R": R, "X": X, "Y": Y}
    return paths

def GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T, P0T, lambd, eta):
    dt = T / NoOfSteps
    time = np.linspace(0, T, NoOfSteps + 1)
    
    # Initial interest rate using forward rate approximation
    r0 = f0T(DT_DERIV, P0T)
    
    theta = np.array([1.0 / lambd * (f0T(t + dt, P0T) - f0T(t - dt, P0T)) / (2 * dt)
                      + f0T(t, P0T)
                      + eta**2 / (2 * lambd**2) * (1 - np.exp(-2 * lambd * t))
                      for t in time])
    
    # Generates standard normal increments 
    Z = np.random.normal(0.0, 1.0, (NoOfPaths, NoOfSteps))
    dW = np.sqrt(dt) * Z
    W = np.hstack([np.zeros((NoOfPaths, 1)), np.cumsum(dW, axis=1)])
    
    R = np.empty((NoOfPaths, NoOfSteps + 1))
    R[:, 0] = r0
    
    for i in range(NoOfSteps):
        R[:, i + 1] = R[:, i] + lambd * (theta[i] - R[:, i]) * dt + eta * dW[:, i]
    
    paths = {"time": time, "R": R}
    return paths

def HW_theta(lambd, eta, P0T):
    return lambda t: (1.0 / lambd * (f0T(t + DT_DERIV, P0T) - f0T(t - DT_DERIV, P0T)) / (2 * DT_DERIV)
                      + f0T(t, P0T)
                      + eta**2 / (2 * lambd**2) * (1 - np.exp(-2 * lambd * t)))

def HW_A(lambd, eta, P0T, T1, T2):
    tau = T2 - T1
    zGrid = np.linspace(0.0, tau, 250)
    B_r = lambda tau_val: 1.0 / lambd * (np.exp(-lambd * tau_val) - 1.0)
    theta = HW_theta(lambd, eta, P0T)
    temp1 = lambd * np.trapezoid(theta(T2 - zGrid) * B_r(zGrid), zGrid)
    temp2 = (eta**2 / (4 * lambd**3)) * (np.exp(-2 * lambd * tau) * (4 * np.exp(lambd * tau) - 1) - 3) + eta**2 * tau / (2 * lambd**2)
    return temp1 + temp2

def HW_B(lambd, T1, T2):
    return 1.0 / lambd * (np.exp(-lambd * (T2 - T1)) - 1.0)

def HW2F_ZCB(lambd1, lambd2, eta1, eta2, rho, P0T, T1, T2, xT1, yT1):
    V = lambda t, T: (
        (eta1**2) / (lambd1**2) * ((T-t) + 2.0/lambd1*np.exp(-lambd1*(T-t)) - 1.0/(2*lambd1)*np.exp(-2*lambd1*(T-t)) - 3.0/(2*lambd1)) +
        (eta2**2) / (lambd2**2) * ((T-t) + 2.0/lambd2*np.exp(-lambd2*(T-t)) - 1.0/(2*lambd2)*np.exp(-2*lambd2*(T-t)) - 3.0/(2*lambd2)) +
        2 * rho * eta1 * eta2 / (lambd1 * lambd2) * ((T-t) + 1.0/lambd1*(np.exp(-lambd1*(T-t))-1) +
                                                     1.0/lambd2*(np.exp(-lambd2*(T-t))-1) -
                                                     1.0/(lambd1+lambd2)*(np.exp(-(lambd1+lambd2)*(T-t))-1))
    )
    intPhi = - np.log(P0T(T2) / P0T(T1) * np.exp(-0.5*(V(0, T2) - V(0, T1))))
    A = 1.0 / lambd1 * (1.0 - np.exp(-lambd1 * (T2 - T1)))
    B = 1.0 / lambd2 * (1.0 - np.exp(-lambd2 * (T2 - T1)))
    return np.exp(-intPhi - A * xT1 - B * yT1 + 0.5 * V(T1, T2))

def HW_ZCB(lambd, eta, P0T, T1, T2, rT1):
    B_r = HW_B(lambd, T1, T2)
    A_r = HW_A(lambd, eta, P0T, T1, T2)
    return np.exp(A_r + B_r * rT1)

def HW_r_0(P0T, lambd, eta):
    return f0T(0.001, P0T)

def mainCalculation():
    NoOfPaths = 2000
    NoOfSteps = 200
    
    # Hull-White 1F parameters
    lambd = 0.01
    eta = 0.002
    
    # Hull-White 2F parameters
    lambd2 = 0.1
    eta2 = 0.002
    rho = -0.2
    
    # cubic spline interpolation
    interpolator = interpolate.splrep(ti, Pi, s=0.0001)
    P0T = lambda T_val: interpolate.splev(T_val, interpolator, der=0)
    
    r0 = HW_r_0(P0T, lambd, eta)
    
    N = 20    # Compares the ZCB prices from Market and standard expression
    T_end0 = 39.0
    Tgrid = np.linspace(0.1, T_end0, N)
    
    Exact = np.array([P0T(Ti) for Ti in Tgrid])
    Proxy_1FHW = np.array([HW_ZCB(lambd, eta, P0T, 0.0, Ti, r0) for Ti in Tgrid])
    Proxy_2FHW = np.array([HW2F_ZCB(lambd, lambd2, eta, eta2, rho, P0T, 0.0, Ti, 0.0, 0.0) for Ti in Tgrid])
    Yield = -np.log(Proxy_1FHW) / Tgrid
    
    plt.figure(1)
    plt.grid()
    plt.plot(Tgrid, Exact, '-k', label="Standard ZCB")
    plt.plot(Tgrid, Proxy_1FHW, '--r', label="ZCB - 1F Model")
    plt.plot(Tgrid, Proxy_2FHW, '.k', label="ZCB - 2F Model")
    plt.legend()
    plt.title('Path from Monte Carlo vs. Standard method')
    
    # Monte Carlo simulation for HW1F
    T_end = 10.0
    paths = GeneratePathsHWEuler(NoOfPaths, NoOfSteps, T_end, P0T, lambd, eta)
    r_paths = paths["R"]
    timeGrid = paths["time"]
    
    # yield curve (1F)
    plt.figure(2)
    plt.grid()
    plt.plot(Tgrid, Yield, '-k')
    plt.figure(3)
    plt.xlabel('Time')
    plt.ylabel('r(t)')
    plt.title('MC Paths + Yield Curve (Hull-White 1F)')
    plt.grid()
    T_end2 = T_end + 40.0
    Tgrid2 = np.linspace(T_end + 0.001, T_end2 - 0.01, N)

    for i in range(min(20, NoOfPaths)):
        ZCB_vals = np.array([HW_ZCB(lambd, eta, P0T, T_end, Tj, r_paths[i, -1]) for Tj in Tgrid2])
        yield_curve = -np.log(ZCB_vals) / (Tgrid2 - T_end)
        plt.plot(Tgrid2, yield_curve, color='gray', alpha=0.5)
        plt.plot(timeGrid, r_paths[i, :], color='blue', alpha=0.3)
    
    paths_2F = GeneratePathsHW2FEuler(NoOfPaths, NoOfSteps, T_end, P0T, lambd, lambd2, eta, eta2, rho)
    x = paths_2F["X"]
    y = paths_2F["Y"]
    r_2F = paths_2F["R"]
    timeGrid = paths_2F["time"]
    
    plt.figure(4)
    plt.xlabel('Time')
    plt.ylabel('r(t)')
    plt.title('MC Paths + Yield Curve (Hull-White 2F)')
    plt.grid()
 
    for i in range(min(20, NoOfPaths)):
        x_T = x[i, -1]
        y_T = y[i, -1]
        ZCB_vals = np.array([HW2F_ZCB(lambd, lambd2, eta, eta2, rho, P0T, T_end, Tj, x_T, y_T) for Tj in Tgrid2])
        yield_curve = -np.log(ZCB_vals) / (Tgrid2 - T_end)
        plt.plot(timeGrid, r_2F[i, :], color='green', alpha=0.3)
        plt.plot(Tgrid2, yield_curve, color='orange', alpha=0.5)
    
    plt.show()

mainCalculation()
