import numpy as np
from simulator import simulate_gbm
from analytics import GBMAnalytics
from plotter import plot_simulation

def main() -> None:
    """
    Main orchestrator for the Quantitative Analytics Pipeline.
    Defines market parameters, executes the Monte Carlo simulation,
    and triggers the statistical risk engine.
    """
    print("[INFO] Initializing quantitative analytics pipeline...")

    # =========================================================================
    # 1. MARKET PARAMETERIZATION (Example with spot asset at $100)
    # =========================================================================
    S0 = 100.0          # Initial asset price (Spot)
    drift = 0.08        # Annualized expected return rate (represented by μ)
    sigma = 0.25        # Annualized historical/implied volatility (represented by σ)
    T = 1.0             # Time horizon in years (1 year)
    n_steps = 252       # Trading days in a standard investment year (Time Steps)
    n_paths = 10_000    # Number of independent Monte Carlo scenarios (Paths)

    # =========================================================================
    # 2. SIMULATION ENGINE EXECUTION (VECTORIZED)
    # =========================================================================
    paths = simulate_gbm(S0, drift, sigma, T, n_steps, n_paths)

    # =========================================================================
    # 3. ANALYTICS ENGINE INSTANTIATION & OUTPUT
    # =========================================================================
    # The analyzer object consumes the generated matrix and isolates risk logic
    analyzer = GBMAnalytics(paths)
    
    # Displays the structured Trading Desk Report in the terminal
    analyzer.generate_desk_report()

    # Generates the institutional-grade visual plot
    plot_simulation(paths, T)

if __name__ == "__main__":
    main()