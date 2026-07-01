import numpy as np

class GBMAnalytics:
    """
    Consolidates statistical analysis and market risk metrics derived from 
    asset price trajectories generated via Monte Carlo simulations.
    """

    def __init__(self, paths: np.ndarray):
        if paths.ndim != 2:
            raise ValueError(f"The 'paths' matrix must be 2D. Received dimension: {paths.ndim}")
        
        self.paths = paths
        self.s0 = paths[0, 0]
        self.final_prices = paths[-1, :]

    def mean(self) -> float:
        """Calculates the arithmetic mean of terminal prices (Expected Value)."""
        return float(np.mean(self.final_prices))
    
    def median(self) -> float:
        """Calculates the median of terminal prices (Robust central tendency for skewed distributions)."""
        return float(np.median(self.final_prices))

    def calculate_win_probability(self) -> float:
        """Returns the proportion (0.0 to 1.0) of simulated trajectories that closed above S0."""
        winning_paths = np.sum(self.final_prices > self.s0)
        total_paths = self.final_prices.shape[0]
        return float(winning_paths / total_paths)
    
    def calculate_var(self, confidence_level: float = 0.95) -> float:
        """Calculates the absolute financial Value at Risk (VaR) on the left tail of the distribution."""
        alpha = (1.0 - confidence_level) * 100
        threshold_price = np.percentile(self.final_prices, alpha)
        return float(self.s0 - threshold_price)
    
    def calculate_confidence_intervals(self, confidence_level: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Calculates dynamic volatility bands (lower and upper percentiles) for each discrete time step."""
        lower_percentile = ((1.0 - confidence_level) / 2.0) * 100
        upper_percentile = (confidence_level + (1.0 - confidence_level) / 2.0) * 100
        
        lower_bound = np.percentile(self.paths, lower_percentile, axis=1)
        upper_bound = np.percentile(self.paths, upper_percentile, axis=1)
        
        return lower_bound, upper_bound
    
    def generate_desk_report(self) -> None:
        """
        Generates a structured Trading Desk Report directly to the terminal,
        consolidating core simulation parameters and extracted risk metadata.
        """
        # Execute underlying calculation pipeline
        mean_p = self.mean()
        median_p = self.median()
        win_prob = self.calculate_win_probability()
        abs_var = self.calculate_var(confidence_level=0.95)
        
        # Additional descriptive statistics crucial for tail risk analysis
        min_p = float(np.min(self.final_prices))
        max_p = float(np.max(self.final_prices))
        
        # Metadata extracted from matrix dimensions
        n_steps = self.paths.shape[0] - 1
        n_paths = self.paths.shape[1]

        # Terminal Visual Layout
        print("=" * 60)
        print(f"{'QUANT PORTFOLIO RISK & ANALYTICS REPORT':^60}")
        print("=" * 60)
        
        # Section 1: Pipeline Parameters & Dimensions
        print(f"[{'SIMULATION METADATA':-<54}]")
        print(f"{'Initial Price (S0):':<30} $ {self.s0:<15.2f}")
        print(f"{'Number of Steps:':<30} {n_steps:<15d}")
        print(f"{'Simulated Paths:':<30} {n_paths:<15,d}")
        print("-" * 60)
        
        # Section 2: Central Tendency & Log-Normal Properties
        print(f"[{'CENTRAL TENDENCY METRICS':-<54}]")
        print(f"{'Final Prices Mean:':<30} $ {mean_p:<15.2f}")
        print(f"{'Final Prices Median:':<30} $ {median_p:<15.2f}")
        print(f"{'Minimum Price Reached:':<30} $ {min_p:<15.2f}")
        print(f"{'Maximum Price Reached:':<30} $ {max_p:<15.2f}")
        print("-" * 60)
        
        # Section 3: Tail Risk & Performance
        print(f"[{'RISK AND PERFORMANCE METRICS':-<54}]")
        print(f"{'Win Probability (P > S0):':<30} {win_prob * 100:.2f} %")
        print(f"{'Value at Risk (Absolute 95% VaR):':<30} $ {abs_var:<15.2f}")
        
        # Portfolio Manager contextual insight
        print("\n[RISK INFO]")
        print(f" With 95% confidence, the worst-case financial loss projected")
        print(f" from the initial price S0 is $ {abs_var:.2f} per asset unit.")
        print("=" * 60)