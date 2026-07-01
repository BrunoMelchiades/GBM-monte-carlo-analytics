# High-Performance Monte Carlo Simulator & Risk Analytics Engine

A production-grade, vectorized quantitative pipeline built in Python to simulate asset price trajectories using **Geometric Brownian Motion (GBM)** and extract advanced financial risk metrics.

## 1. Mathematical Framework

The asset price dynamic is modeled using the classical Black-Scholes-Merton framework, governed by the following Stochastic Differential Equation (SDE):

$$dS_t = \mu S_t dt + \sigma S_t dW_t$$

Where:
* $S_t$ is the asset price at time $t$.
* $\mu$ (drift) represents the annualized expected return.
* $\sigma$ (diffusion) represents the annualized volatility.
* $dW_t$ is a standard Wiener process (Brownian motion) satisfying $dW_t \sim \mathcal{N}(0, dt)$.

Applying Itô's Lemma to $f(S_t) = \ln(S_t)$, we obtain the analytical solution for the stochastic integration used in our vectorized simulator:

$$S_t = S_0 \exp\left( \left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t \right)$$

---

## 2. Architecture & Algorithmic Efficiency

Unlike naive iterative implementations that rely on slow Python `for` loops, this engine utilizes **pure NumPy vectorization** to build the entire path matrix simultaneously. This ensures high computational throughput and memory efficiency when scaling to millions of scenarios.

### Repository Structure
* `simulator.py`: Implements the geometric random walk integration using `np.random.normal`, cumulative products via `np.cumprod`, and matrix stacking.
* `analytics.py`: Encapsulates the statistical post-processing risk engine (Value at Risk, dynamic confidence bands, central tendency properties).
* `plotter.py`: Generates institutional-grade visualization profiles (Matplotlib) displaying probability envelopes over multi-path simulations.
* `main.py`: Orchestrates the complete quantitative data pipeline.

---

## 3. Trading Desk Report Output

Upon execution, the analytics engine processes the multi-dimensional path array and outputs a structured risk assessment report directly to the terminal, adopting global institutional standards:

```text
============================================================
          QUANT PORTFOLIO RISK & ANALYTICS REPORT           
============================================================
[SIMULATION METADATA----------------------------------------]
Initial Price (S0):            $ 100.00         
Number of Steps:               252            
Simulated Paths:               10,000         
------------------------------------------------------------
[CENTRAL TENDENCY METRICS-----------------------------------]
Final Prices Mean:             $ 107.84         
Final Prices Median:           $ 104.36         
Minimum Price Reached:         $ 37.24          
Maximum Price Reached:         $ 262.29         
------------------------------------------------------------
[RISK AND PERFORMANCE METRICS-------------------------------]
Win Probability (P > S0):      57.09%         
Value at Risk (Absolute 95% VaR): $ 30.80         

[RISK INFO]
 With 95% confidence, the worst-case financial loss projected
 from the initial price S0 is $ 30.80 per asset unit.
============================================================
```
*Note: The positive skewness (Mean > Median) naturally emerges from the log-normal distribution properties of the GBM model, capturing the asymmetric tail risk expected by Portfolio Managers.*
