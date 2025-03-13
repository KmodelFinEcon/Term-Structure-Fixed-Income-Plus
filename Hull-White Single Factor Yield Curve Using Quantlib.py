####Hull-White Single Factor Yield Curve model###
#               by. K.Tomov

import math
from QuantLib import *
import numpy as np
import matplotlib.pyplot as plt

#Hull White objects

def main():
    # Setup start and end dates
    startDate = Date(4, September, 2023)
    endDate = Date(2, October, 2026)
    tenor = Period(1, Days)
    grid = Grid(startDate, endDate, tenor)
    
    # Yield Curve
    flatRate = 0.04875825 #current RFR
    dayCounter = Actual365Fixed() 
    curve = YieldTermStructureHandle(FlatForward(startDate, flatRate, dayCounter))
    reversionSpeed = 0.05 #Speed of mean-reversion
    rateVolatility = 0.0024 #IR Volatitlity
    hw_process = HullWhiteProcess(curve, reversionSpeed, rateVolatility)

    # Simulation parameters
    nPaths = 15000
    timeGrid = grid.GetTimeGrid()
    times = grid.GetTimes()
    gridSize = grid.GetSize()
    dt = grid.GetDt()
    paths = GeneratePaths(hw_process, timeGrid, nPaths) # short-rate paths using the Hull-White process
    cumulative_integrals = np.cumsum(paths, axis=1) * dt 
    simulated_discount_factors = np.mean(np.exp(-cumulative_integrals), axis=0) #discount factor from paths
    simulated_discount_factors[0] = 1.0
    dates = grid.GetDates()
    simulatedCurve = DiscountCurve(dates, simulated_discount_factors, dayCounter, NullCalendar())
    original_discount_factors = np.array([curve.discount(t) for t in times]) # OG discount factor vs new

    #ploting comparables
    plt.figure(figsize=(10, 6))
    plt.title('Discount Factors: MC Simulated vs OG Yield Curve')
    plt.plot(times, simulated_discount_factors, linestyle='dashed', label='MC simulated curve')
    plt.plot(times, original_discount_factors, linestyle='solid', label='OG curve')
    plt.xlabel("Time")
    plt.ylabel("Discount Factor")
    plt.legend()
    plt.grid(True)
    plt.show()

    #discount factor
    plt.figure(figsize=(10, 6))
    plt.title('Discount factor differential')
    plt.plot(times, (original_discount_factors - simulated_discount_factors) * 10000)
    plt.xlabel("Time")
    plt.ylabel("Dif")
    plt.grid(True)
    plt.show()

    # average short-rates dynamics
    avg_short_rate = np.mean(paths, axis=0)
    plt.figure(figsize=(10, 6))
    plt.title('Hull White rate evolution')
    plt.plot(times, avg_short_rate, label='Average Simulated Short Rate')
    plt.axhline(y=flatRate, color='b', linestyle='--', label='benchmark forward rate')
    plt.xlabel("Time")
    plt.ylabel("Short Rate")
    plt.legend()
    plt.grid(True)
    plt.show()

    num_sample_paths = 10  # Plot a few sample paths for visualization
    plt.figure(figsize=(10, 6))
    plt.title('Hull White short rates')
    for i in range(num_sample_paths):
        plt.plot(times, paths[i, :], lw=1, alpha=0.7)
    plt.xlabel("Time")
    plt.ylabel("Short Rate")
    plt.grid(True)
    plt.show()

# Path generator for a single-factor Hull-White process using GaussianPathGenerator
def GeneratePaths(process, timeGrid, n):
    sequenceGenerator = UniformRandomSequenceGenerator(len(timeGrid), UniformRandomGenerator())
    gaussianSequenceGenerator = GaussianRandomSequenceGenerator(sequenceGenerator)
    maturity = timeGrid[-1]
    pathGenerator = GaussianPathGenerator(process, maturity, len(timeGrid), gaussianSequenceGenerator, False)
    paths = np.zeros((n, len(timeGrid)))
    for i in range(n):
        path = pathGenerator.next().value()# Store the path values
        paths[i, :] = np.array([path[j] for j in range(len(timeGrid))])
    return paths

class Grid:
    def __init__(self, startDate, endDate, tenor):
        self.schedule = Schedule(startDate, endDate, tenor, NullCalendar(), 
                                 Unadjusted, Unadjusted, DateGeneration.Forward, False)
        self.dayCounter = Actual365Fixed()

    def GetDates(self):
        return [self.schedule[i] for i in range(self.GetSize())]

    def GetTimes(self):
        return [self.dayCounter.yearFraction(self.schedule[0], self.schedule[i]) for i in range(self.GetSize())]

    def GetSize(self):
        return len(self.schedule)

    def GetTimeGrid(self):
        return TimeGrid(self.GetTimes(), self.GetSize())

    def GetDt(self):
        return (self.dayCounter.yearFraction(self.schedule[0], self.schedule[-1])) / (self.GetSize() - 1)        # Calculate constant time step from the start to end date

if __name__ == "__main__":
    main()