import numpy as np


def simulate_gbm(S0, mu, sigma, T, n_steps, n_paths):
    """
    Simulador de Monte Carlo para o Movimento Geométrico Browniano (GBM).
    Utiliza vetorização pura em NumPy para alta performance.
    
    Parâmetros:
    S0      : Preço inicial do ativo (float)
    mu      : Drift/Tendência média de retorno anualizado (float)
    sigma   : Volatilidade anualizada (float)
    T       : Tempo total da projeção em anos (float)
    n_steps : Número de intervalos de tempo (int)
    n_paths : Número de cenários independentes a simular (int)
    
    Retorna:
    paths   : Matrix NumPy de formato (n_steps + 1, n_paths) com as evoluções dos preços
    """
    # 1. Delta de tempo para cada passo
    dt = T / n_steps
    
    # 2. Geração da matriz de choques aleatórios Gaussianos ~ N(0, 1)
    Z = np.random.normal(size=(n_steps, n_paths))
    
    # 3. Cálculo dos fatores de retorno contínuo usando o Lema de Itô
    factor = np.exp((mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)
    
    # 4. Criação da matriz final com o preço inicial S0 na linha zero
    paths = np.empty((n_steps + 1, n_paths))
    paths[0, :] = S0
    
    # 5. Aplicação do produtório acumulado diretamente na fatia da matriz (Vetorização eficiente)
    paths[1:, :] = S0 * np.cumprod(factor, axis=0)
    
    return paths

if __name__ == "__main__":
    # Testando com parâmetros padrão
    S0 = 100
    mu = 0.10
    sigma = 0.20
    T = 1.0
    n_steps = 252
    n_paths = 10000

    resultado = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths)

    print(f"Formato da matriz final: {resultado.shape}")
    print(f"Preço inicial no primeiro caminho: {resultado[0, 0]}")
    print(f"Preço final estimado no primeiro caminho: {resultado[-1, 0]}")