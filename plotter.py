import numpy as np
import matplotlib.pyplot as plt
from analytics import GBMAnalytics

def plot_simulation(paths: np.ndarray, T: float, max_paths_to_plot: int = 100) -> None:
    """
    Generates an institutional-grade visualization profile for the Monte Carlo simulation.
    
    Plots a highly transparent sample cloud of trajectories and overlays the
    95% confidence intervals and the expected mean trajectory extracted from GBMAnalytics.
    """
    # 1. Extract metadata from matrix dimensions (n_steps + 1, n_paths)
    n_steps = paths.shape[0] - 1
    n_paths = paths.shape[1]
    
    # 2. Construct aligned temporal X-axis
    time_steps = np.linspace(0, T, n_steps + 1)
    
    # 3. Instantiate the analytics engine to extract temporal statistical curves
    analyzer = GBMAnalytics(paths)
    temporal_mean = np.mean(paths, axis=1)  # Step-by-step mean evolution
    lower_bound, upper_bound = analyzer.calculate_confidence_intervals(confidence_level=0.95)
    
    # 4. Initialize figure with standard widescreen report aspect ratio (16:9 compressed)
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # 5. Plot a sample cloud of simulated paths (Ensures UI rendering performance)
    # Restricts plotting to 'max_paths_to_plot' to avoid thread blocking
    sampled_paths = paths[:, :max_paths_to_plot]
    
    # Matplotlib automatically maps each column as an isolated line
    ax.plot(time_steps, sampled_paths, color="gray", alpha=0.15, linewidth=0.8, label="_nolegend_")
    
    # 6. Plot statistical control signal curves
    ax.plot(time_steps, temporal_mean, color="#1f77b4", linestyle="--", linewidth=2, label="Expected Mean Price")
    ax.plot(time_steps, lower_bound, color="#d62728", linestyle="-", linewidth=1.5, label="Lower Bound (95% CI)")
    ax.plot(time_steps, upper_bound, color="#2ca02c", linestyle="-", linewidth=1.5, label="Upper Bound (95% CI)")
    
    # 7. Shading the Confidence Envelope (Standard Risk Management reporting layout)
    ax.fill_between(time_steps, lower_bound, upper_bound, color="gray", alpha=0.07, label="Confidence Zone (95%)")
    
    # 8. Fine-tuning and interface styling (Bloomberg/Reuters Terminal style)
    ax.set_title(f"Monte Carlo Simulation - GBM ({n_paths:,} Paths)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Time Horizon (Years)", fontsize=11, labelpad=10)
    ax.set_ylabel("Asset Price ($)", fontsize=11, labelpad=10)
    
    # Clip X-axis origin strictly at day zero
    ax.set_xlim(0, T)
    
    # Subtle grid configuration for coordinate tracking
    ax.grid(True, linestyle=":", alpha=0.6, color="gray")
    
    # Strategic legend positioning (prevents overlaying the S0 initial price cluster)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", shadow=False)
    
    # Optimize layout margins and display rendering
    plt.tight_layout()
    plt.show()