"""
Q4-Q6: Detectores de Pontos de Interesse (Harris, ORB, KAZE)
TP1 - Reconnaissances de Points Caractéristiques
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

print("="*70)
print("Q4: Detector de Harris - Função de Interesse")
print("="*70)

# Leitura da imagem
img_path = 'TP1_Features/Image_Pairs/Graffiti0.png'
img = np.float64(cv2.imread(img_path, cv2.IMREAD_GRAYSCALE))
(h, w) = img.shape
print(f"Dimensão da imagem: {h} linhas x {w} colunas")
print(f"Tipo da imagem: {img.dtype}")

# ============================================================================
# Q4: IMPLEMENTAÇÃO DO DETECTOR DE HARRIS
# ============================================================================
print("\nCálculo da função de interesse de Harris...")

# Parâmetros
W_size = 5  # Tamanho da janela de somação (recomendado 3-7)
alpha = 0.04  # Parâmetro k de Harris (~0.04-0.06)
threshold_ratio = 0.01  # Limiar relativo para detecção

# Passo 1: Calcular os gradientes (Ix, Iy)
Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32) / 8.0
Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32) / 8.0

Ix = cv2.filter2D(img, cv2.CV_64F, Kx)
Iy = cv2.filter2D(img, cv2.CV_64F, Ky)

print(f"  Gradientes calculados (Ix, Iy)")

# Passo 2: Calcular os produtos dos gradientes
Ix2 = Ix**2
Iy2 = Iy**2
Ixy = Ix * Iy

print(f"  Produtos dos gradientes calculados")

# Passo 3: Suavizar com Gaussiana
Ix2_smoothed = cv2.GaussianBlur(Ix2, (W_size, W_size), 2.0)
Iy2_smoothed = cv2.GaussianBlur(Iy2, (W_size, W_size), 2.0)
Ixy_smoothed = cv2.GaussianBlur(Ixy, (W_size, W_size), 2.0)

print(f"  Valores suavizados com Gaussiana (tamanho de janela: {W_size}x{W_size})")

# Passo 4: Calcular a função de interesse de Harris para cada pixel
# R = det(M) - alpha*trace(M)^2
# onde M = [[Ix2, Ixy], [Ixy, Iy2]] é a matriz de estrutura local

det_M = Ix2_smoothed * Iy2_smoothed - Ixy_smoothed**2
trace_M = Ix2_smoothed + Iy2_smoothed
harris_response = det_M - alpha * (trace_M**2)

print(f"  Resposta de Harris calculada")
print(f"    Min: {harris_response.min():.2f}, Max: {harris_response.max():.2f}")

# Passo 5: Não-máxima supressão (NMS)
# Encontrar máximos locais usando dilatação morfológica
se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
harris_dil = cv2.dilate(harris_response, se)

# Máximos locais: pixels onde a resposta é igual ao máximo local
corners_map = np.zeros_like(harris_response)
corners_map[(harris_response == harris_dil) & (harris_response > 0)] = harris_response[(harris_response == harris_dil) & (harris_response > 0)]

# Aplicar limiar relativo
threshold = threshold_ratio * harris_response.max()
corners_map[corners_map < threshold] = 0

# Encontrar coordenadas dos cantos
y_coords, x_coords = np.where(corners_map > 0)
corners_harris = list(zip(x_coords, y_coords, corners_map[y_coords, x_coords]))
corners_harris.sort(key=lambda x: x[2], reverse=True)  # Ordenar por resposta

print(f"  Máximos locais encontrados: {len(corners_harris)} pontos")
print(f"  Top 5 respostas: {[c[2] for c in corners_harris[:5]]}")

# Visualização
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Imagem original')
axes[0, 0].axis('off')

axes[0, 1].imshow(harris_response, cmap='hot')
axes[0, 1].set_title('Resposta de Harris (antes NMS)')
axes[0, 1].axis('off')

# Plotar os cantos detectados
axes[1, 0].imshow(img, cmap='gray')
for x, y, _ in corners_harris[:100]:  # Mostrar top 100
    axes[1, 0].plot(x, y, 'r+', markersize=10)
axes[1, 0].set_title(f'Detctor Harris ({len(corners_harris)} pontos)')
axes[1, 0].axis('off')

# Investigar o efeito do limiar
axes[1, 1].imshow(corners_map, cmap='hot')
axes[1, 1].set_title('Mapa de máximos locais (após NMS)')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('results/q4_harris_detection.png', dpi=100, bbox_inches='tight')
print("Imagem salva: results/q4_harris_detection.png\n")
plt.close()

# ============================================================================
# Q5: ANÁLISE DE PARÂMETROS E MULTESCALA
# ============================================================================
print("="*70)
print("Q5: Análise dos parâmetros e extensão multescala")
print("="*70)

# Testar diferentes tamanhos de janela
window_sizes = [3, 5, 7, 9]
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for idx, w_size in enumerate(window_sizes):
    # Recalcular com novo tamanho
    ix2_s = cv2.GaussianBlur(Ix2, (w_size, w_size), 2.0)
    iy2_s = cv2.GaussianBlur(Iy2, (w_size, w_size), 2.0)
    ixy_s = cv2.GaussianBlur(Ixy, (w_size, w_size), 2.0)
    
    dt_m = ix2_s * iy2_s - ixy_s**2
    tr_m = ix2_s + iy2_s
    harris_s = dt_m - alpha * (tr_m**2)
    
    # Encontrar pontos
    h_dil = cv2.dilate(harris_s, se)
    c_map = np.zeros_like(harris_s)
    c_map[(harris_s == h_dil) & (harris_s > 0)] = harris_s[(harris_s == h_dil) & (harris_s > 0)]
    c_map[c_map < threshold] = 0
    
    n_corners = np.sum(c_map > 0)
    
    axes[idx].imshow(img, cmap='gray')
    y_c, x_c = np.where(c_map > 0)
    axes[idx].plot(x_c, y_c, 'r.', markersize=4)
    axes[idx].set_title(f'Janela {w_size}x{w_size} ({n_corners} pontos)')
    axes[idx].axis('off')

print("\nEfeito do tamanho da janela de suavização:")
print("  - Janelas pequenas (3x3): Mais sensível a ruído, detecta mais detalhes")
print("  - Janelas médias (5x5, 7x7): Equilíbrio entre robustez e precisão")
print("  - Janelas grandes (9x9+): Menos sensível a ruído, detecta estruturas maiores")

plt.tight_layout()
plt.savefig('results/q5_window_sizes.png', dpi=100, bbox_inches='tight')
print("\nImagem salva: results/q5_window_sizes.png")
plt.close()

# Análise também com álpha
print("\nEfeito do parâmetro alpha (k de Harris):")
print("  - alpha ≈ 0.04-0.06: Intervalo típico")
print("  - alpha pequeno: Mais cantos detectados")
print("  - alpha grande: Menos cantos, mais seletivo")

# ============================================================================
# Q6: COMPARAÇÃO ORB vs KAZE
# ============================================================================
print("\n" + "="*70)
print("Q6: Comparação de detectores - ORB vs KAZE")
print("="*70)

# Ler imagem colorida para visualização
img_color = cv2.imread(img_path, cv2.IMREAD_COLOR)
img_gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Detector ORB
orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2, nlevels=8)
kp_orb, des_orb = orb.detectAndCompute(img_gray, None)
print(f"ORB: {len(kp_orb)} pontos detectados")
print(f"  Parâmetros: nfeatures=500, scaleFactor=1.2, nlevels=8")

# Detector KAZE
kaze = cv2.KAZE_create()
kp_kaze, des_kaze = kaze.detectAndCompute(img_gray, None)
print(f"KAZE: {len(kp_kaze)} pontos detectados")
print(f"  Parâmetros: padrão (detector automático de escala)")

# Visualizar
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# ORB
img_orb = cv2.drawKeypoints(img_gray, kp_orb, None, color=(0, 255, 0),
                            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
axes[0].imshow(cv2.cvtColor(img_orb, cv2.COLOR_BGR2RGB))
axes[0].set_title(f'ORB Detector ({len(kp_orb)} pontos)')
axes[0].axis('off')

# KAZE
img_kaze = cv2.drawKeypoints(img_gray, kp_kaze, None, color=(0, 255, 0),
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
axes[1].imshow(cv2.cvtColor(img_kaze, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'KAZE Detector ({len(kp_kaze)} pontos)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('results/q6_orb_kaze_comparison.png', dpi=100, bbox_inches='tight')
print("Imagem salva: results/q6_orb_kaze_comparison.png\n")
plt.close()

# Análise detalhada dos detectores
print("\nPRINCÍPIOS DOS DETECTORES:\n")
print("ORB (Oriented FAST and Rotated BRIEF):")
print("  - Detector: FAST (Features from Accelerated Segment Test)")
print("  - Baseado na comparação rápida de intensidades de pixéis")
print("  - Muito rápido, mas sensível a rotações")
print("  - Trabalha bem em múltiplas escalas com pirâmide de imagens")
print("\nKAZE:")
print("  - Detector baseado em morfologia matemática (não-linear diffusion)")
print("  - Detecta pontos de interesse em múltiplas escalas automaticamente")
print("  - Mais robusto a transformações afins")
print("  - Mais lento que ORB mas mais preciso")

print("\nPARÂMETROS ORB:")
print("  - nfeatures: número máximo de pontos (~500)")
print("  - scaleFactor: escala entre níveis da pirâmide (~1.2)")
print("  - nlevels: número de níveis da pirâmide (~8)")
print("  - fastThreshold: limiar FAST (~31)")

print("\nPARÂMETROS KAZE:")
print("  - threshold: limiar de resposta (~0.001)")
print("  - nOctaves: número de oitavas (~4)")
print("  - nOctaveLayers: camadas por oitava (~2)")
print("  - diffusivity: tipo de difusão (NORM_L1 ou NORM_L2)")

# ============================================================================
# Q6b: AVALIAÇÃO DE REPETIBILIDADE
# ============================================================================
print("\n" + "="*70)
print("Q6b: Avaliação visual de repetibilidade")
print("="*70)

# Testar com par de imagens
img1_path = 'TP1_Features/Image_Pairs/Graffiti0.png'
img2_path = 'TP1_Features/Image_Pairs/Graffiti1.png'

img1_gray = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2_gray = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# ORB
kp1_orb, des1_orb = orb.detectAndCompute(img1_gray, None)
kp2_orb, des2_orb = orb.detectAndCompute(img2_gray, None)

# KAZE
kp1_kaze, des1_kaze = kaze.detectAndCompute(img1_gray, None)
kp2_kaze, des2_kaze = kaze.detectAndCompute(img2_gray, None)

print(f"Imagem 1 - ORB: {len(kp1_orb)} pontos, KAZE: {len(kp1_kaze)} pontos")
print(f"Imagem 2 - ORB: {len(kp2_orb)} pontos, KAZE: {len(kp2_kaze)} pontos")

# Visualizar detecções lado a lado
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# ORB
img1_orb = cv2.drawKeypoints(img1_gray, kp1_orb, None, 
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_orb = cv2.drawKeypoints(img2_gray, kp2_orb, None,
                             flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
axes[0, 0].imshow(cv2.cvtColor(img1_orb, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f'ORB - Imagem 1 ({len(kp1_orb)} pontos)')
axes[0, 0].axis('off')

axes[0, 1].imshow(cv2.cvtColor(img2_orb, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f'ORB - Imagem 2 ({len(kp2_orb)} pontos)')
axes[0, 1].axis('off')

# KAZE
img1_kaze = cv2.drawKeypoints(img1_gray, kp1_kaze, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img2_kaze = cv2.drawKeypoints(img2_gray, kp2_kaze, None,
                              flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
axes[1, 0].imshow(cv2.cvtColor(img1_kaze, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f'KAZE - Imagem 1 ({len(kp1_kaze)} pontos)')
axes[1, 0].axis('off')

axes[1, 1].imshow(cv2.cvtColor(img2_kaze, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f'KAZE - Imagem 2 ({len(kp2_kaze)} pontos)')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('results/q6_repetibility_assessment.png', dpi=100, bbox_inches='tight')
print("Imagem salva: results/q6_repetibility_assessment.png\n")
plt.close()

print("="*70)
print("Processamento Q4-Q6 concluído!")
print("="*70)
