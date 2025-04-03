
#BDT Interest rate model implementation from treasury data by. K.Tomov

import yfinance as yf
import math 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

# ZCB data fetch:
tickers = {
    "^IRX": "13-Week (3-Month) T-Bill",
    "^FVX": "5-Year Treasury",
    "^TNX": "10-Year Treasury",
    "^TYX": "30-Year Treasury",
}

data = yf.download(
    tickers=list(tickers.keys()),
    period="1y", 
    interval="1d", 
    group_by="ticker"
)

for ticker, description in tickers.items():
    if ticker in data:
        print(f"\n{description} ({ticker}) Data:")
        print(data[ticker].tail()) 


def payoff(x, typ):
    if typ == "bond":
        return x
    else:
        return 0
        
def cf_floor(rates, strike, delta, notion, cpn):
    cf = np.zeros([len(rates)+1, len(rates)+1])
    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            rate = rates[row, col]
            cf[row, col] = delta * notion * max(strike/100 - rate, 0)
    return cf 

def cf_cap(rates, strike, delta, notion, cpn):
    cf = np.zeros([len(rates)+1, len(rates)+1])
    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            rate = rates[row, col]
            cf[row, col] = delta * notion * max(rate - strike/100, 0)
    return cf

def cf_bond(rates, strike, delta, notion, cpn):
    cf = np.zeros([len(rates)+1, len(rates)+1])
    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            cf[row, col] = delta * notion * cpn/100  
    return cf

def cf_swap(rates, strike, delta, notion, cpn):
    cf = np.zeros([len(rates)+1, len(rates)+1])
    for col in range(0, len(cf)-1):
        for row in range(0, col+1):
            rate = rates[row, col]
            cf[row, col] = delta * notion * (rate - strike/100)
    return cf

def display(arr):
    for i in arr:
        for j in i:
            print("{:8.4f}".format(j), end="  ")
        print() 
    print("\n")

def probTree(length):
    prob = np.zeros((length, length))
    prob[np.triu_indices(length, 0)] = 0.5
    return prob

def solver(theta, tree, zcb, i, sigma, delta):
    price = np.zeros([i+2, i+2])
    price[:, i+1] = 1 

    for row in range(0, i+1):
        if row == 0: 
            tree[row, i] = tree[row, i-1] + theta*delta + sigma*math.sqrt(delta)
        else:
            tree[row, i] = tree[row-1, i-1] + theta*delta - sigma*math.sqrt(delta)
    
    pricingTree = np.exp(tree)
    for col in reversed(range(0, i+1)):
        for row in range(0, col+1):
            if np.isnan(pricingTree[row, col]):
                return 1e8
            node = np.exp(-pricingTree[row, col] * delta)
            price[row, col] = node * (0.5 * price[row, col+1] + 0.5 * price[row+1, col+1])
    
    result = price[0,0] - zcb
    if np.isnan(result):
        return 1e8 
    return result


def calibrate(tree, zcb, i, sigma, delta):
    t0 = 0.5 
    def solver_wrapper(theta):
        return solver(theta, tree, zcb, i, sigma, delta)
    theta = brentq(solver_wrapper, a=0.1, b=2.0, xtol=1e-8)
    for row in range(0, i+1):
        if row == 0: 
            tree[row, i] = tree[row, i-1] + theta*delta + sigma*math.sqrt(delta)
        else:
            tree[row, i] = tree[row-1, i-1] + theta*delta - sigma*math.sqrt(delta)
    return theta, tree

            
def build(zcb, sigma, delta):
    tree  = np.zeros([zcb.shape[1], zcb.shape[1]])
    theta = np.zeros([zcb.shape[1]]) 

    r0 = -np.log(zcb[0,0]) / delta
    tree[0,0] = np.log(r0)
    
    for i in range(1, len(theta)):
        theta[i], tree = calibrate(tree, zcb[0,i], i, sigma, delta)
    return r0, tree, theta
    
def rateTree(r0, theta, sigma, delta):
    tree = np.zeros([len(theta), len(theta)])
    tree[0,0] = np.log(r0)
    for col in range(1, len(tree)):
        tree[0, col] = tree[0, col-1] + theta[col]*delta + sigma*math.sqrt(delta)
    for col in range(1, len(tree)):
        for row in range(1, col+1):
            tree[row, col] = tree[row-1, col] - 2*sigma*math.sqrt(delta)
    return np.exp(tree)

def priceOption(rates, prob, cf, delta, typ, notion, ptree, bond_px):
    tree = np.zeros([len(rates)+1, len(rates)+1])
    for col in reversed(range(0, len(tree)-1)):
        for row in range(0, col+1):
            rate = rates[row, col]
            call_ex   =  ptree[row, col]  - notion
            call_wait =  np.exp(-rate * delta) * (prob * tree[row, col+1] + prob * tree[row+1, col+1])
            tree[row, col] = max(call_ex, call_wait)             
    option_px   = tree[0,0]
    callable_px = bond_px - option_px        
    return callable_px, option_px, bond_px, tree
    
def priceTree(rates, prob, cf, delta, typ, notion):
    tree = np.zeros([len(rates)+1, len(rates)+1])
    tree[:, -1] = payoff(notion, typ)
    for col in reversed(range(0, len(tree)-1)):  
        for row in range(0, col+1):
            rate = rates[row, col]
            tree[row, col] = np.exp(-rate * delta) * (0.5*(tree[row, col+1] + cf[row, col+1]) + 0.5*(tree[row+1, col+1] + cf[row+1, col+1]))
    return tree[0,0], tree

#main Excusion

if __name__ == '__main__':
    sample_ticker = list(tickers.keys())[0]
    df = data[sample_ticker].reset_index()
    
    try:
        df['Date'] = pd.to_datetime(df['Date'])
        target_date = pd.to_datetime('2024-03-08')
        df_filtered = df[df['Date'] == target_date]
        if not df_filtered.empty:
            df = df_filtered
    except Exception as e:
        print("Date filtering error:", e)
        
    zeros = np.array([df['Close'].values])

    if zeros.shape[1] < 60:
        pad_width = 60 - zeros.shape[1]
        zeros = np.pad(zeros, ((0,0), (0, pad_width)), mode='edge')
    else:
        zeros = zeros[:, :60]
    
    x = build(zeros, sigma=0.21, delta=1/12)
    tree_bdt = rateTree(x[0], x[2], sigma=0.21, delta=1/12)
    cashfl = cf_bond(tree_bdt, strike=5.00, delta=1/12, notion=1, cpn=0.00) #assumptions given for model creation
    pricing, ptree = priceTree(tree_bdt, prob=0.5, cf=cashfl, delta=1/12, typ="bond", notion=1)
    
    #printed output
    
    print("final value", zeros[0, -1])
    print("Bond price from BDT:", pricing)
    print("displayed differences:", pricing - zeros[0, -1])