# TP1 - Reconhecimento e Appariement de Imagens

## Estrutura do Projeto

```
tp1/
├── TP1_Rapport_Technique_Complet.pdf  # RELATÓRIO TÉCNICO COMPLETO (PDF do TP)
├── q1_q3_convolutions.py         # Solução para Q1-Q3
├── q4_q6_detectors.py            # Solução para Q4-Q6
├── q7_q9_descriptors.py          # Solução para Q7-Q9
├── generate_report.py            # Gerador do relatório PDF
├── non_deliverables_backup/      # Material auxiliar (código base, venv, etc.)
└── results/                      # Resultados e visualizações
    ├── q1_convolution_comparison.png
    ├── q3_gradients.png
    ├── q4_harris_detection.png
    ├── q5_window_sizes.png
    ├── q6_orb_kaze_comparison.png
    ├── q6_repetibility_assessment.png
    ├── q8_matching_strategies.png
    └── q9_quantitative_evaluation.png
```

## Requisitos

- Python 3.8+
- OpenCV 4.1.0+
- NumPy
- SciPy
- Matplotlib
- ReportLab

## Instalação

```bash
# Criar e ativar virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Instalar dependências
pip install opencv-python numpy scipy matplotlib reportlab pillow
```

## Executar as Soluções

### Q1-Q3: Convoluções e Gradientes
```bash
python q1_q3_convolutions.py
```

Resultados:
- Comparação Método Direto vs filter2D (speedup 246.7x)
- Análise de rehaussement de contraste
- Cálculo de gradientes (Ix, Iy, ||∇I||)

### Q4-Q6: Detectores
```bash
python q4_q6_detectors.py
```

Resultados:
- Implementação do detector de Harris
- Análise de parâmetros e multiscala
- Comparação ORB vs KAZE
- Avaliação de repetibilidade

### Q7-Q9: Descritores e Appariement
```bash
python q7_q9_descriptors.py
```

Resultados:
- Análise de descritores ORB vs KAZE
- Comparação de 3 estratégias de appariement:
  1. Cross-Check
  2. Ratio Test (Lowe's)
  3. FLANN
- Avaliação quantitativa com transformação afim

### Gerar Relatório
O relatório técnico completo já foi gerado e está em:

- `TP1_Rapport_Technique_Complet.pdf`

Esse PDF contém todas as análises, figuras e conclusões das questões Q1–Q9.

## Resultados Resumidos

### Q1-Q3: Convoluções e Gradientes
- **Q1**: Velocidade filter2D é 246x mais rápida que loop Python
- **Q2**: Kernel realiza rehaussement via amplificação de diferenças locais
- **Q3**: Gradientes calculados com Sobelacek; cuidados com normalização essenciais

### Q4-Q6: Detectores
- **Q4**: Detector Harris implementado com 628 pontos detectados
- **Q5**: Tamanho de janela afeta sensibilidade (small=detalhes, large=estruturas)
- **Q6**: ORB rápido (500 pts), KAZE mais denso (1270 pts) e robusto

### Q7-Q9: Appariement
- **Q7**: ORB binário (32B, rápido); KAZE float (256B, preciso)
- **Q8**: Cross-Check conservador; Ratio test equilibrado; FLANN rápido
- **Q9**: Erro de reprojeção é métrica fundamental de qualidade

## Observações Importantes

### Convoluções (Q1-Q3)
- Sempre usar funções otimizadas (cv2.filter2D, cv2.Sobel)
- Precauções para exibição: abs(), normalização, clipping
- Gradientes detectam bordas (variações de intensidade)

### Detectores (Q4-Q6)
- Harris teórico; ORB/KAZE práticos
- Parâmetros críticos: nfeatures, scaleFactor, nlevels
- Multiscala essencial para robustez
- Repetibilidade avalia performance em transformações

### Descritores (Q7-Q9)
- Invariância à rotação e escala muito importante
- Diferentes distâncias para diferentes tipos de descritor
- Ratio test muito efetivo para eliminar ambígüidades
- RANSAC recomendado para filtragem final

## Experimentos Adicionais Recomendados

1. Variar limiares e parâmetros (explorar sensibilidade)
2. Testar com imagens de diferentes domínios
3. Implementar SIFT/SURF para comparação
4. Estudar invariância a iluminação
5. Aplicar a panorama stitching ou estrutura 3D

## Métricas de Avaliação

### Qualitativas
- Visualização de detecções
- Visualização de casamentos
- Análise de falsosítivos/negativos

### Quantitativas
- Número de pontos detectados
- Taxa de repetibilidade
- Erro de reprojeção (RMSE)
- Precisão/Recall
- Tempo de processamento

## Referências

- OpenCV Docs: https://docs.opencv.org/4.1.0/
- Harris Corner Detection: "A Combined Corner and Edge Detector" (Harris & Stephens, 1988)
- ORB: "ORB: An Efficient Alternative to SIFT or SURF" (Rublee et al., 2011)
- AKAZE: "AKAZE: an algorithm to extract keypoints and descriptors from videos"

## Autores

- Lucca de Souza Amodio
- Vinicius Zancheta
- João Victor Ferro

Trabalho desenvolvido para o curso MI204 - ENSTA Paris
