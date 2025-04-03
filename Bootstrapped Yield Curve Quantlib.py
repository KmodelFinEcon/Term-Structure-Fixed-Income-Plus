#Boostrapped yield curve implementation using Quantlib by k.tomov

import QuantLib as ql
import matplotlib.pyplot as plt
import numpy as np

# global  parameters
quote_date = ql.Date(28, 3, 2023)
ql.Settings.instance().evaluationDate = quote_date
sigma = 0.023 #vol
yield_curve_basis = ql.Actual365Fixed()
compounding = ql.CompoundedThenSimple
day_count = ql.Actual365Fixed()
dates = [ql.Date(30, 6, 2023), ql.Date(30, 9, 2023)]
future_prices = [0.03, 0.013]

# TTM in years and convexity adjustments
time_to_maturities = [ (date - quote_date) / 365.0 for date in dates ]
convexity_adjustments = [0.5 * sigma**2 * t**2 for t in time_to_maturities]

#yield curve instrument for benchmark
instruments = [
    ql.DepositRateHelper(
        ql.QuoteHandle(ql.SimpleQuote(price + adjustment)),
        ql.Period(date - quote_date, ql.Days),0, 
        ql.TARGET(),
        ql.ModifiedFollowing,
        False,
        day_count
    )
    
    for price, adjustment, date in zip(future_prices, convexity_adjustments, dates)
]

# Bootstrap the yield curve using PiecewiseLogCubicDiscount function
curve = ql.PiecewiseLogCubicDiscount(quote_date, instruments, yield_curve_basis)
curve.enableExtrapolation() 

print("Zero Rates at established Maturities:")
for date, t in zip(dates, time_to_maturities):
    zero_rate = curve.zeroRate(t, compounding, True).rate()
    print(f" The Zero rate for {date}: {zero_rate * 100:.2f}%")

#result plot of curve

t_max = max(time_to_maturities) * 1.5
times = np.linspace(0, t_max, 100)
zero_rates = [curve.zeroRate(t, compounding, True).rate() for t in times]

plt.figure(figsize=(10, 6))
plt.plot(times, [zr * 100 for zr in zero_rates], label="ZR")
plt.xlabel("Time ")
plt.ylabel("Zero Rate in percent")
plt.title("Bootstrapped Yield Curve")
plt.grid(True)
plt.legend()
plt.show()

