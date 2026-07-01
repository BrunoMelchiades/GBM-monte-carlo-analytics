import numpy as np
import matplotlib.pyplot as plt
from analytics import GBMAnalytics

def plotar_simulacao(paths: np.ndarray, T: float, max_caminhos_plot: int = 100) -> None:
    """
    Gera uma visualização de padrão institucional para a simulação de Monte Carlo.
    
    Plota uma nuvem amostral de trajetórias com alta transparência e sobrepõe
    as bandas dos intervalos de confiança (95%) e a média esperada calculadas
    pela classe GBMAnalytics.
    """
    # 1. Extração de Metadados da Matriz (n_steps + 1, n_paths)
    n_steps = paths.shape[0] - 1
    n_paths = paths.shape[1]
    
    # 2. Construção do Eixo X Temporal Alinhado
    tempo = np.linspace(0, T, n_steps + 1)
    
    # 3. Instanciação do Analytics para extração das curvas temporais
    analisador = GBMAnalytics(paths)
    media_temporal = np.mean(paths, axis=1)  # Evolução da média passo a passo
    limite_inf, limite_sup = analisador.calcular_intervalos_confianca(nivel_confianca=0.95)
    
    # 4. Inicialização da Figura com Proporção de Tela de Relatório (16:9 abreviado)
    fig, ax = plt.subplots(figsize=(11, 6))
    
    # 5. Plotagem da Nuvem de Caminhos Amostrados (Garante performance visual)
    # Seleciona apenas os primeiros 'max_caminhos_plot' para não travar o renderizador
    caminhos_amostra = paths[:, :max_caminhos_plot]
    
    # O Matplotlib plota cada coluna como uma linha isolada automaticamente
    ax.plot(tempo, caminhos_amostra, color="gray", alpha=0.15, linewidth=0.8, label="_nolegend_")
    
    # 6. Plotagem das Curvas Estatísticas de Controle (Sinal)
    ax.plot(tempo, media_temporal, color="#1f77b4", linestyle="--", linewidth=2, label="Preço Médio Esperado")
    ax.plot(tempo, limite_inf, color="#d62728", linestyle="-", linewidth=1.5, label="Limite Inferior (IC 95%)")
    ax.plot(tempo, limite_sup, color="#2ca02c", linestyle="-", linewidth=1.5, label="Limite Superior (IC 95%)")
    
    # 7. Sombreamento da Zona de Confiança (Padrão de Relatório de Risco)
    ax.fill_between(tempo, limite_inf, limite_sup, color="gray", alpha=0.07, label="Zona de Confiança (95%)")
    
    # 8. Estilização e Ajustes Finos de Interface (Estilo Bloomberg/Reuters Terminal)
    ax.set_title(f"Simulação de Monte Carlo - GBM ({n_paths:,} Caminhos)", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Horizonte Temporal (Anos)", fontsize=11, labelpad=10)
    ax.set_ylabel("Preço do Ativo ($)", fontsize=11, labelpad=10)
    
    # Fixar a origem do eixo X exatamente no dia zero
    ax.set_xlim(0, T)
    
    # Configuração de Grid sutil para leitura de coordenadas
    ax.grid(True, linestyle=":", alpha=0.6, color="gray")
    
    # Posicionamento estratégico da legenda (evita sobrepor o início da simulação S0)
    ax.legend(loc="upper left", frameon=True, facecolor="white", edgecolor="none", shadow=False)
    
    # Otimização de margens e exibição
    plt.tight_layout()
    plt.show()