
#YTM optimization under ZCB, C-bond, FRN by kt

from itertools import product
import numpy as np
from numpy.random import randint
from scipy.optimize import minimize_scalar

# Global bond parameters and grid search settings
FV= 1000         # payoff at time T
Mat = 10 # T, in years
Price = 850 # market price at time t0
Couponr = 0.05 # coupon rate for the C-bond
Floatr = np.asarray(randint(1, 10, Mat) / 100)  # random floating rates
DIM = 5 # dimensions for grid search
SCALE = 0.005# scale for grid search

class Bond:
    def __init__(self, face_value, maturity, price, coupon_rate=0.0, frn=False):
        self.face_value = face_value
        self.maturity = maturity
        self.price = price
        self.coupon_rate = coupon_rate
        self.frn = frn
        self.cf = None
        self.dt = None

    def cash_flow(self):
        if self.frn:
            # For FRN: coupon_rate is an array of rates for each period
            self.cf = np.append(self.face_value * self.coupon_rate[:-1],
                                self.face_value * (1 + self.coupon_rate[-1]))
            return self.cf
        else:
            # For ZCB and C-bond: coupon_rate is a scalar value
            self.cf = np.append(
                np.full(self.maturity - 1, self.face_value * self.coupon_rate),
                self.face_value * (1 + self.coupon_rate))
            return self.cf

    def discount_factors(self, rate):
        self.dt = []
        if isinstance(rate, (float, int)):
            for t in range(1, self.maturity + 1):
                self.dt.append(1 / (1 + rate) ** t)
        else:
            # rate is expected to be an array for grid search
            for gr in rate:
                dt_i = []
                for t in range(1, self.maturity + 1):
                    dt_i.append(1 / (1 + gr) ** t)
                self.dt.append(dt_i)
            self.dt = np.asmatrix(self.dt)
        return self.dt

    def net_present_value(self):
        if isinstance(self.dt, list):
            return - self.price + np.dot(self.cf, self.dt)
        else:
            return - self.price * np.ones(len(self.dt)) + np.tensordot(self.dt, self.cf, axes=1)


def solve_brent(bond):
    def objective_function(bond, ytm):
        bond.discount_factors(ytm)
        npv = bond.net_present_value()
        return np.abs(npv)

    # Minimize the objective function using Brent's method
    res = minimize_scalar(lambda ytm: objective_function(bond, ytm), method='Brent')
    bond.discount_factors(res.x)
    return res.x, bond.net_present_value()


def solve_integer(dim, scale, bond):
    mu = np.geomspace(scale, 2 ** dim * scale, num=dim, endpoint=False)
    mu = np.sort(mu)[::-1]
    sigma_space = np.asarray(list(product(range(2), repeat=dim)))
    sigma = sigma_space[sigma_space[:, 0].argsort()]
    tau = np.dot(sigma, mu)
    bond.discount_factors(tau)
    omega = bond.net_present_value()
    i = np.argmin(np.abs(omega))
    return tau[i], omega[i]


def main():
    # Build bonds using the global parameters
    # Zero-Coupon Bond (ZCB)
    zcb = Bond(FV, Mat, Price)
    zcb.cash_flow()
    # Coupon Bond (C-bond)
    c_bond = Bond(FV, Mat, Price, Couponr)
    c_bond.cash_flow()
    # Floating-Rate Note (FRN)
    frn = Bond(FV, Mat, Price, Floatr, frn=True)
    frn.cash_flow()

    # Display assumptions
    print('\n==Assumptions=='
          f'\nFace value   = {FV}'
          f'\nMaturity     = {Mat}'
          f'\nMarket price = {Price}'
          f'\nCoupon rate  = {Couponr}'
          f'\nFloating rates = {Floatr}\n')

    # Solve using scalar minimization (Brent method)
    print('>Brent method< \nBond\t|   YTM   |   NPV')
    ytm, npv = solve_brent(zcb)
    print(f'ZCB     \t| {ytm * 100:6.4f}% | {npv:4.4f}')
    ytm, npv = solve_brent(c_bond)
    print(f'C-bond  \t| {ytm * 100:6.4f}% | {npv:4.4f}')
    ytm, npv = solve_brent(frn)
    print(f'FRN     \t| {ytm * 100:6.4f}% | {npv:4.4f}')

    # Solve using integer grid search
    print('\n>Integer GS< \nBond\t|   YTM   |   NPV')
    ytm, npv = solve_integer(DIM, SCALE, zcb)
    print(f'ZCB     \t| {ytm * 100:6.4f}% | {npv:4.4f}')
    ytm, npv = solve_integer(DIM, SCALE, c_bond)
    print(f'C-bond  \t| {ytm * 100:6.4f}% | {npv:4.4f}')
    ytm, npv = solve_integer(DIM, SCALE, frn)
    print(f'FRN     \t| {ytm * 100:6.4f}% | {npv:4.4f}')
    print('Grid size =', 2 ** DIM)

if __name__ == '__main__':
    main()