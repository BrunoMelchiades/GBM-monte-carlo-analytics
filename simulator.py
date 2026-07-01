import numpy as np


def simulate_gbm(S0: float, drift: float, sigma: float, T: float, n_steps: int, n_paths: int) -> np.ndarray:
    """
    Monte Carlo simulator for Geometric Brownian Motion (GBM).
    Utilizes pure NumPy vectorization for high-performance execution.
    
    Parameters:
        S0      : Initial asset price (float)
        drift   : Annualized expected return rate (represented by μ) (float)
        sigma   : Annualized volatility rate (represented by σ) (float)
        T       : Total time horizon projected in years (float)
        n_steps : Number of discrete time intervals (int)
        n_paths : Number of independent simulated scenarios (int)
    
    Returns:
        paths   : NumPy matrix of shape (n_steps + 1, n_paths) containing simulated price trajectories
    """
    # 1. Time delta for each discrete step
    dt = T / n_steps
    
    # 2. Generate matrix of random Gaussian shocks ~ N(0, 1)
    Z = np.random.normal(size=(n_steps, n_paths))
    
    # 3. Compute continuous return factors using Itô's Lemma
    factor = np.exp((drift - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    # 4. Initialize final matrix and anchor initial price S0 at step zero
    paths = np.empty((n_steps + 1, n_paths))
    paths[0, :] = S0
    
    # 5. Apply cumulative product across the time axis (Efficient matrix slicing)
    paths[1:, :] = S0 * np.cumprod(factor, axis=0)
    
    return paths

if __name__ == "__main__":
    # Test suite executing with standard desk parameters
    S0 = 100.0
    drift = 0.10
    sigma = 0.20
    T = 1.0
    n_steps = 252
    n_paths = 10000

    simulation_results = simulate_gbm(S0, drift, sigma, T, n_steps, n_paths)

    # Validate tensor properties directly in terminal
    print(f"Final matrix shape: {simulation_results.shape}")
    print(f"Initial asset price on the first path: {simulation_results[0, 0]:.2f}")
    print(f"Estimated final price on the first path: {simulation_results[-1, 0]:.2f}")