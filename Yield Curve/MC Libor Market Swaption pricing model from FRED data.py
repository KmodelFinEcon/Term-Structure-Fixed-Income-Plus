
#Monte Carlo simluation of Swaption prices using Libor Market Model.

import pandas_datareader.data as web
import datetime
import numpy as np
import math

# Global Parameters
dt = 0.10 # Time step for simulation (make sure this unit fits your simulation horizon)
tau = 0.1 # Tenor period
beta_param = 0.3 # Correlation decay parameter for the LIBOR market model

# Objective functions

def GetForwardRatesAndTerms(P, tau):
    n = P.shape[0]
    theRates = np.zeros(n - 1)
    theTerms = np.zeros(n - 1)
    for i in range(n - 1):
        val = (P[i] / P[i + 1] - 1) / tau
        theRates[i] = val
        theTerms[i] = i * tau
    return theTerms, theRates

def CreateCorrelation(beta, terms):
    numberOfTerms = len(terms)
    corrMatrix = np.zeros((numberOfTerms, numberOfTerms))
    for countX, i in enumerate(terms):
        for countY, j in enumerate(terms):
            corrMatrix[countX, countY] = math.exp(-beta * abs(i - j))
    return corrMatrix

def corr(Tj, Tk, beta):
    return math.exp(-beta * abs(Tj - Tk))

def vol(Tj, T_o):  # volatility calibrated from market (implied volatility surface)
    a = -0.03
    b = 0.2
    c = 0.2
    d = 0.10
    return ((a + b * (Tj - T_o)) * math.exp(-c * (Tj - T_o)) + d)

def GetSwapRate(F, alpha, beta_index):
    tmp = 1.0
    for j in range(alpha, beta_index):
        tmp *= 1.0 / (1 + tau * F[j])
    SR = 1 - tmp
    tmpSum = 0.0
    for i in range(alpha, beta_index):
        tmp = 1.0
        for j in range(alpha, i + 1):
            tmp *= 1.0 / (1 + tau * F[j])
        tmpSum += tau * tmp
    SR = SR / tmpSum
    return SR

def MonteCarloLiborMarketModel(beta, tau, forwardRates, terms, numberOfScenarios, g_alpha, g_beta):
    swapRate = GetSwapRate(forwardRates, g_alpha, g_beta)
    T_sim = g_alpha * tau
    number_of_steps = int(math.ceil(T_sim / dt))
    
    corrMatrix = CreateCorrelation(beta, terms)
    cholMatrix = np.linalg.cholesky(corrMatrix).T

    randVecList = []

    libor_simulations = np.zeros((numberOfScenarios, g_beta))
    finalFVec = np.zeros(g_beta)
    discountCurve = np.zeros(g_beta)

    sqrDt = math.sqrt(dt)
    payoff_sum = 0.0
    useRandom = True
    np.random.seed(10000)

    for scenario in range(numberOfScenarios):
        currLog = np.log(forwardRates.copy())
        currLog_t = np.log(forwardRates.copy())
    
        antiTheticFlag = (scenario % 2 == 1)
        randVecList = []
        
        t = 0.0
        step = 4  # 
        
        while t < T_sim:
            if not antiTheticFlag:
                randNumbers = np.random.randn(g_beta)
                randChol = np.dot(randNumbers, cholMatrix)
                randVecList.append(randChol)
            else:
                if step < len(randVecList):
                    randChol = -1.0 * randVecList[step]
                else:
                    randNumbers = np.random.randn(g_beta)
                    randChol = -1.0 * np.dot(randNumbers, cholMatrix)
            
            nextResetIdx = int(math.floor(t / tau)) + 1
            nextResetIdx = min(nextResetIdx, g_beta)
            
            for k in range(nextResetIdx, g_beta):
                driftSum = 0.0
                for j in range(nextResetIdx, k + 1):
                    tmp = (corr(terms[k], terms[j], beta) * tau *
                           vol(terms[j], t) * math.exp(currLog_t[j]))
                    tmp = tmp / (1 + tau * math.exp(currLog_t[j]))
                    driftSum += tmp
                dLogF = 0.0
                vol_Tk_t = vol(terms[k], t)
                dLogF += vol_Tk_t * driftSum * dt
                dLogF -= 0.5 * vol_Tk_t**2 * dt
                if useRandom:
                    dLogF += vol_Tk_t * randChol[k] * sqrDt
                else:
                    dLogF += vol_Tk_t * 1.0 * sqrDt
                currLog[k] += dLogF
            currLog_t = currLog.copy()
            step += 1
            t += dt

        for i in range(g_beta):
            libor_simulations[scenario, i] = math.exp(currLog_t[i])
            finalFVec[i] = math.exp(currLog_t[i])
        
        discountCurve[0] = 1.0 / (1 + tau * finalFVec[0])
        for i in range(1, g_beta):
            discountCurve[i] = discountCurve[i - 1] / (1 + tau * finalFVec[i])
            
        payoff = 0.0
        for i in range(g_alpha, g_beta):
            payoff += (swapRate - finalFVec[i]) * tau * discountCurve[i]
        payoff_sum += max(payoff, 0.0)
        
    return 100 * payoff_sum / numberOfScenarios

# Execution

if __name__ == '__main__':
    
    numberOfScenarios = 1000
    start = datetime.datetime(2020, 1, 1)
    end = datetime.datetime(2024, 12, 31)
    print("Swap data fetch from:", start, "to", end)
    beta = beta_param
    
    try:
        swap_df = web.DataReader("IRLTLT01USM156N", 'fred', start, end)
        swap_df.dropna(inplace=True)
        swaprices = swap_df.values.flatten() 
        print("Sample rates:", swaprices[:10])
    except Exception as e:
        print("Can't get rates:", e)
        swaprices = np.array([])

    if swaprices.size < 10:
        print("Not sufficient rates available")
    else:
        terms, forwardRates = GetForwardRatesAndTerms(swaprices, tau)
        
        g_beta = len(terms)
        g_alpha = max(1, int(g_beta / 2))
        print("Using galpha =", g_alpha, "and gbeta =", g_beta)
        
        # Run the Monte Carlo LIBOR market model simulation.
        swaptionPrice = MonteCarloLiborMarketModel(beta, tau, forwardRates, terms, numberOfScenarios, g_alpha, g_beta)
        print("Final swaption price =", swaptionPrice)
