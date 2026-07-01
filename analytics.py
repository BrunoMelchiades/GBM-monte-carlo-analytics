import numpy as np

class GBMAnalytics:
    """
    Consolida a análise estatística e métricas de risco de mercado baseadas 
    nas trajetórias geradas por uma simulação de Monte Carlo.
    """

    def __init__(self, paths: np.ndarray):
        if paths.ndim != 2:
            raise ValueError(f"A matriz 'paths' deve ser 2D. Dimensão recebida: {paths.ndim}")
        
        self.paths = paths
        self.s0 = paths[0, 0]
        self.final_prices = paths[-1, :]

    def media(self) -> float:
        """Calcula a média aritmética dos preços finais (Valor Esperado)."""
        return float(np.mean(self.final_prices))
    
    def mediana(self) -> float:
        """Calcula a mediana dos preços finais (Métrica central robusta para assimetria)."""
        return float(np.median(self.final_prices))

    def calcular_probabilidade_ganho(self) -> float:
        """Retorna a proporção (0.0 a 1.0) de caminhos que terminaram acima de S0."""
        caminhos_com_ganho = np.sum(self.final_prices > self.s0)
        total_caminhos = self.final_prices.shape[0]
        return float(caminhos_com_ganho / total_caminhos)
    
    def calcular_var(self, nivel_confianca: float = 0.95) -> float:
        """Calcula o Value at Risk (VaR) financeiro absoluto na cauda esquerda."""
        alfa = (1.0 - nivel_confianca) * 100
        preco_limite = np.percentile(self.final_prices, alfa)
        return float(self.s0 - preco_limite)
    
    def calcular_intervalos_confianca(self, nivel_confianca: float = 0.95) -> tuple[np.ndarray, np.ndarray]:
        """Calcula as bandas de volatilidade (inferior e superior) para cada passo temporal."""
        percentil_inferior = ((1.0 - nivel_confianca) / 2.0) * 100
        percentil_superior = (nivel_confianca + (1.0 - nivel_confianca) / 2.0) * 100
        
        limite_inferior = np.percentile(self.paths, percentil_inferior, axis=1)
        limite_superior = np.percentile(self.paths, percentil_superior, axis=1)
        
        return limite_inferior, limite_superior
    
    def exibir_relatorio(self) -> None:
        """
        Gera um Relatório de Mesa (Desk Report) estruturado no terminal,
        consolidando os parâmetros de entrada e as métricas de risco extraídas.
        """
        # Executa os cálculos internos do pipeline analítico
        media_p = self.media()
        mediana_p = self.mediana()
        prob_ganho = self.calcular_probabilidade_ganho()
        var_abs = self.calcular_var(nivel_confianca=0.95)
        
        # Estatísticas descritivas adicionais importantes para a cauda
        min_p = float(np.min(self.final_prices))
        max_p = float(np.max(self.final_prices))
        
        # Metadados extraídos do shape do array
        n_steps = self.paths.shape[0] - 1
        n_paths = self.paths.shape[1]

        # Início do Layout do Relatório
        print("=" * 60)
        print(f"{'QUANT PORTFOLIO RISK & ANALYTICS REPORT':^60}")
        print("=" * 60)
        
        # Seção 1: Parâmetros e Dimensões do Pipeline
        print(f"[{'METADADOS DA SIMULAÇÃO':-<54}]")
        print(f"{'Preço Inicial (S0):':<30} $ {self.s0:<15.2f}")
        print(f"{'Número de Passos (Steps):':<30} {n_steps:<15d}")
        print(f"{'Caminhos Simulados (Paths):':<30} {n_paths:<15,d}")
        print("-" * 60)
        
        # Seção 2: Tendência Central (Distribuição Log-Normal)
        print(f"[{'MÉTRICAS DE TENDÊNCIA CENTRAL':-<54}]")
        print(f"{'Média dos Preços Finais:':<30} $ {media_p:<15.2f}")
        print(f"{'Mediana dos Preços Finais:':<30} $ {mediana_p:<15.2f}")
        print(f"{'Preço Mínimo Atingido:':<30} $ {min_p:<15.2f}")
        print(f"{'Preço Máximo Atingido:':<30} $ {max_p:<15.2f}")
        print("-" * 60)
        
        # Seção 3: Risco e Probabilidade de Cauda
        print(f"[{'MÉTRICAS DE RISCO E SUCCESSO':-<54}]")
        print(f"{'Probabilidade de Ganho (P > S0):':<30} {prob_ganho * 100:.2f} %")
        print(f"{'Value at Risk (VaR 95% Absoluto):':<30} $ {var_abs:<15.2f}")
        
        # Contextualização do VaR para o Portfolio Manager
        print("\n[INFO RISK]")
        print(f" Com 95% de confiança, a pior perda financeira projetada")
        print(f" a partir do preço inicial S0 é de $ {var_abs:.2f} por ativo.")
        print("=" * 60)