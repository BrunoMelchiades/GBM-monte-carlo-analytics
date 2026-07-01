import numpy as np
from simulator import simulate_gbm
from analytics import GBMAnalytics
from plotter import plotar_simulacao

def main() -> None:
    """
    Orquestrador principal do Pipeline Quantitativo.
    Define os parâmetros de mercado, executa a simulação de Monte Carlo
    e dispara a engine estatística de risco.
    """
    print("[INFO] Inicializando pipeline de análise quantitativa...")

    # =========================================================================
    # 1. PARAMETRIZAÇÃO DE MERCADO (Exemplo com ativo spot a $100)
    # =========================================================================
    S0 = 100.0          # Preço inicial do ativo (Spot)
    mu = 0.08           # Taxa de retorno esperada (Anualizada)
    sigma = 0.25        # Volatilidade implícita/histórica (Anualizada)
    T = 1.0             # Horizonte temporal em anos (1 ano)
    n_steps = 252       # Dias de negociação em um ano investido (Time Steps)
    n_paths = 10_000    # Número de cenários independentes via Monte Carlo

    # =========================================================================
    # 2. EXECUÇÃO DA ENGINE DE SIMULAÇÃO (VETORIZADA)
    # =========================================================================
    paths = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths)

    # =========================================================================
    # 3. INSTANCIAÇÃO DA CLASSE ANALÍTICA E OUTPUT
    # =========================================================================
    # O objeto consome a matriz gerada e isola a lógica de risco
    analisador = GBMAnalytics(paths)
    
    # Exibe o Relatório de Mesa estruturado no terminal
    analisador.exibir_relatorio()

    # Exibe uma representação visual do relatório
    plotar_simulacao(paths, T)

if __name__ == "__main__":
    main()