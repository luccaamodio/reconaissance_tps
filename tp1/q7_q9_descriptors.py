"""
Q7-Q9: Descritores e Appariement de Pontos
TP1 - Reconnaissances de Points Caractéristiques
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

print("="*70)
print("Q7: Princípios dos Descritores - ORB vs KAZE")
print("="*70)

# Carregar imagens
img_path = 'TP1_Features/Image_Pairs/Graffiti0.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# Detectores
orb = cv2.ORB_create(nfeatures=200)
kaze = cv2.KAZE_create()

kp_orb, des_orb = orb.detectAndCompute(img, None)
kp_kaze, des_kaze = kaze.detectAndCompute(img, None)

print(f"\nORB:")
print(f"  Pontos detectados: {len(kp_orb)}")
print(f"  Dimensão do descritor: {des_orb.shape}")
print(f"  Tipo: {des_orb.dtype}")

print(f"\nKAZE:")
print(f"  Pontos detectados: {len(kp_kaze)}")
print(f"  Dimensão do descritor: {des_kaze.shape}")
print(f"  Tipo: {des_kaze.dtype}")

print("""
PROPRIEDADES DOS DESCRITORES:

ORB (Rotated BRIEF):
  - Descritor BINÁRIO (bits) baseado em comparações de pixéis
  - Tamanho: 256 bits (32 bytes) por ponto
  - Invariância à ROTAÇÃO: Orientado pelo centroide do ponto
  - Invariância à ESCALA: Via pirâmide de imagens multiscala
  - Distância: Hamming (XOR + contagem de bits diferentes)
  - Muito rápido e compacto em memória
  
KAZE (Speeded-Up Robust Features):
  - Descritor REAL-VALUED (ponto flutuante)
  - Tamanho: 64 floats (256 bytes) por ponto
  - Invariância à ROTAÇÃO: Orientado via gradientes locais
  - Invariância à ESCALA: Escala automática via non-linear diffusion
  - Distância: Euclidiana ou cosseno
  - Mais descritivo mas requer mais espaço/computação
  
INVARIÂNCIA À ESCALA:
  - ORB: Pirâmide de imagens (múltiplas resoluções)
  - KAZE: Espaço de escala não-linear (evita suavização excessiva)
  
INVARIÂNCIA À ROTAÇÃO:
  - ORB: Calcula orientação via momentos do ponto-chave
  - KAZE: Utiliza gradientes orientados localmente
""")

# ============================================================================
# Q8: ESTRATÉGIAS DE APPARIEMENT
# ============================================================================
print("\n" + "="*70)
print("Q8: Comparação de estratégias de appariement")
print("="*70)

# Carregar par de imagens
img1_path = 'TP1_Features/Image_Pairs/Graffiti0.png'
img2_path = 'TP1_Features/Image_Pairs/Graffiti1.png'

img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

# Detectar e computar descritores
orb = cv2.ORB_create(nfeatures=300)
kaze = cv2.KAZE_create()

kp1_orb, des1_orb = orb.detectAndCompute(img1, None)
kp2_orb, des2_orb = orb.detectAndCompute(img2, None)

kp1_kaze, des1_kaze = kaze.detectAndCompute(img1, None)
kp2_kaze, des2_kaze = kaze.detectAndCompute(img2, None)

# ============================================================================
# Estratégia 1: CROSS-CHECK
# ============================================================================
print("\nESTRATÉGIA 1: Cross-Check (Matching Bidirecional)")
print("  - Descrição: Casamento é válido apenas se bidirecional")
print("  - Processo:")
print("    1. Encontrar NN de descritores de Img1 em Img2")
print("    2. Encontrar NN de descritores de Img2 em Img1")
print("    3. Manter apenas casamentos onde ambos confirmam")
print("  - Vantagem: Reduz falsos positivos")
print("  - Desvantagem: Mais conservador, pode perder casamentos válidos")

# Cross-Check para ORB (Hamming)
bf_orb = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_crosscheck_orb = bf_orb.match(des1_orb, des2_orb)
matches_crosscheck_orb = sorted(matches_crosscheck_orb, key=lambda x: x.distance)

print(f"  ORB CrossCheck: {len(matches_crosscheck_orb)} casamentos")

# Cross-Check para KAZE (Euclidiana)
bf_kaze = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
matches_crosscheck_kaze = bf_kaze.match(des1_kaze, des2_kaze)
matches_crosscheck_kaze = sorted(matches_crosscheck_kaze, key=lambda x: x.distance)

print(f"  KAZE CrossCheck: {len(matches_crosscheck_kaze)} casamentos")

# ============================================================================
# Estratégia 2: RATIO TEST (Lowe's Ratio Test)
# ============================================================================
print("\nESTRATÉGIA 2: Ratio Test (Lowe's Method)")
print("  - Descrição: Valida casamentos usando razão de distâncias")
print("  - Processo:")
print("    1. Para cada descritor, encontrar 2 vizinhos mais próximos")
print("    2. Calcular razão: distância_NN1 / distância_NN2")
print("    3. Manter apenas se razão < limiar (tipicamente 0.7-0.8)")
print("    4. Filtra casamentos ambíguos")
print("  - Vantagem: Mais sensível mas elimina casamentos ambíguos")
print("  - Desvantagem: Requer cálculo de 2 vizinhos")

# Ratio Test para ORB
bf_orb_knn = cv2.BFMatcher(cv2.NORM_HAMMING)
matches_ratiotest_orb = bf_orb_knn.knnMatch(des1_orb, des2_orb, k=2)

# Aplicar Lowe's ratio test
ratio_threshold = 0.75
good_matches_orb = []
for match_pair in matches_ratiotest_orb:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < ratio_threshold * n.distance:
            good_matches_orb.append(m)

print(f"  ORB Ratio Test: {len(good_matches_orb)} casamentos (threshold={ratio_threshold})")

# Ratio Test para KAZE
bf_kaze_knn = cv2.BFMatcher(cv2.NORM_L2)
matches_ratiotest_kaze = bf_kaze_knn.knnMatch(des1_kaze, des2_kaze, k=2)

good_matches_kaze = []
for match_pair in matches_ratiotest_kaze:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < ratio_threshold * n.distance:
            good_matches_kaze.append(m)

print(f"  KAZE Ratio Test: {len(good_matches_kaze)} casamentos (threshold={ratio_threshold})")

# ============================================================================
# Estratégia 3: FLANN (Fast Approximate Nearest Neighbors)
# ============================================================================
print("\nESTRATÉGIA 3: FLANN (Fast Approximate Nearest Neighbors)")
print("  - Descrição: Busca aproximada de vizinhos mais próximos")
print("  - Processo:")
print("    1. Constrói estrutura de dados (KDTree ou randomized trees)")
print("    2. Busca aproximada rápida em vez de exata")
print("    3. Aplicar Lowe's ratio test no resultado")
print("  - Vantagem: Muito mais rápido para grandes conjuntos")
print("  - Desvantagem: Resultado aproximado, pode perder casamentos")

# FLANN para ORB (binário - LSH)
FLANN_INDEX_LSH = 6
index_params_orb = dict(algorithm=FLANN_INDEX_LSH,
                        table_number=12,
                        key_size=20,
                        multi_probe_level=2)
search_params_orb = dict(checks=50)
flann_orb = cv2.FlannBasedMatcher(index_params_orb, search_params_orb)
matches_flann_orb = flann_orb.knnMatch(des1_orb, des2_orb, k=2)

good_matches_flann_orb = []
for match_pair in matches_flann_orb:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < ratio_threshold * n.distance:
            good_matches_flann_orb.append(m)

print(f"  ORB FLANN: {len(good_matches_flann_orb)} casamentos (LSH)")

# FLANN para KAZE (float - KDTree)
FLANN_INDEX_KDTREE = 1
index_params_kaze = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params_kaze = dict(checks=50)
flann_kaze = cv2.FlannBasedMatcher(index_params_kaze, search_params_kaze)
matches_flann_kaze = flann_kaze.knnMatch(des1_kaze, des2_kaze, k=2)

good_matches_flann_kaze = []
for match_pair in matches_flann_kaze:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < ratio_threshold * n.distance:
            good_matches_flann_kaze.append(m)

print(f"  KAZE FLANN: {len(good_matches_flann_kaze)} casamentos (KDTree)")

print("""
DISTÂNCIAS UTILIZADAS:

Por que diferentes distâncias para diferentes descritores:

ORB (descritor binário):
  - Usa NORM_HAMMING (Distância de Hamming)
  - Conta o número de bits diferentes entre dois descritores binários
  - Muito rápida (operações binárias simples)
  - Exemplo: 01010110 vs 01010010 → 1 bit diferente

KAZE (descritor real-valued):
  - Usa NORM_L2 (Distância Euclidiana)
  - ||d1 - d2|| = sqrt(sum((d1_i - d2_i)^2))
  - Mais intuitivo para valores contínuos
  - Pode usar NORM_L1 (Manhattan) também
""")

# Visualizar comparações
fig, axes = plt.subplots(3, 2, figsize=(16, 18))

# Visualizar ORB Cross-Check
img_cc_orb = cv2.drawMatches(img1, kp1_orb, img2, kp2_orb,
                             matches_crosscheck_orb[:20], None,
                             matchColor=(0, 255, 0), flags=2)
axes[0, 0].imshow(cv2.cvtColor(img_cc_orb, cv2.COLOR_BGR2RGB))
axes[0, 0].set_title(f'ORB Cross-Check ({len(matches_crosscheck_orb)} matches)')
axes[0, 0].axis('off')

# Visualizar KAZE Cross-Check
img_cc_kaze = cv2.drawMatches(img1, kp1_kaze, img2, kp2_kaze,
                              matches_crosscheck_kaze[:20], None,
                              matchColor=(0, 255, 0), flags=2)
axes[0, 1].imshow(cv2.cvtColor(img_cc_kaze, cv2.COLOR_BGR2RGB))
axes[0, 1].set_title(f'KAZE Cross-Check ({len(matches_crosscheck_kaze)} matches)')
axes[0, 1].axis('off')

# Visualizar ORB Ratio Test
img_rt_orb = cv2.drawMatches(img1, kp1_orb, img2, kp2_orb,
                             good_matches_orb[:20], None,
                             matchColor=(0, 255, 0), flags=2)
axes[1, 0].imshow(cv2.cvtColor(img_rt_orb, cv2.COLOR_BGR2RGB))
axes[1, 0].set_title(f'ORB Ratio Test ({len(good_matches_orb)} matches)')
axes[1, 0].axis('off')

# Visualizar KAZE Ratio Test
img_rt_kaze = cv2.drawMatches(img1, kp1_kaze, img2, kp2_kaze,
                              good_matches_kaze[:20], None,
                              matchColor=(0, 255, 0), flags=2)
axes[1, 1].imshow(cv2.cvtColor(img_rt_kaze, cv2.COLOR_BGR2RGB))
axes[1, 1].set_title(f'KAZE Ratio Test ({len(good_matches_kaze)} matches)')
axes[1, 1].axis('off')

# Visualizar ORB FLANN
img_fl_orb = cv2.drawMatches(img1, kp1_orb, img2, kp2_orb,
                             good_matches_flann_orb[:20], None,
                             matchColor=(0, 255, 0), flags=2)
axes[2, 0].imshow(cv2.cvtColor(img_fl_orb, cv2.COLOR_BGR2RGB))
axes[2, 0].set_title(f'ORB FLANN ({len(good_matches_flann_orb)} matches)')
axes[2, 0].axis('off')

# Visualizar KAZE FLANN
img_fl_kaze = cv2.drawMatches(img1, kp1_kaze, img2, kp2_kaze,
                              good_matches_flann_kaze[:20], None,
                              matchColor=(0, 255, 0), flags=2)
axes[2, 1].imshow(cv2.cvtColor(img_fl_kaze, cv2.COLOR_BGR2RGB))
axes[2, 1].set_title(f'KAZE FLANN ({len(good_matches_flann_kaze)} matches)')
axes[2, 1].axis('off')

plt.tight_layout()
plt.savefig('results/q8_matching_strategies.png', dpi=100, bbox_inches='tight')
print("\nImagem salva: results/q8_matching_strategies.png\n")
plt.close()

# ============================================================================
# Q9: AVALIAÇÃO QUANTITATIVA DA QUALIDADE USANDO WARP AFFINE
# ============================================================================
print("="*70)
print("Q9: Avaliação quantitativa de appariement")
print("="*70)

# Criar uma transformação afim conhecida
# Ler imagem em cores para referência
img_original = cv2.imread(img1_path, cv2.IMREAD_COLOR)
h, w = img_original.shape[:2]

# Definir pontos para transformação afim
pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])

# Matriz de transformação afim
M = cv2.getAffineTransform(pts1, pts2)

# Aplicar transformação
img_warped = cv2.warpAffine(img_original, M, (w, h))

# Detectar e descrever em ambas as imagens
kp1, des1 = orb.detectAndCompute(img_original, None)
kp2, des2 = orb.detectAndCompute(img_warped, None)

# Aplicar matching
bf = cv2.BFMatcher(cv2.NORM_HAMMING)
matches = bf.knnMatch(des1, des2, k=2)

# Ratio test
good_matches_test = []
for match_pair in matches:
    if len(match_pair) == 2:
        m, n = match_pair
        if m.distance < 0.75 * n.distance:
            good_matches_test.append(m)

print(f"\nTeste de transformação afim:")
print(f"  Pontos em Imagem 1: {len(kp1)}")
print(f"  Pontos em Imagem 2 (warped): {len(kp2)}")
print(f"  Casamentos válidos: {len(good_matches_test)}")

# Calcular métrica de qualidade
if len(good_matches_test) > 3:
    # Obter coordenadas dos pontos casados
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches_test]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches_test]).reshape(-1, 1, 2)
    
    # Estimar a transformação usando os casamentos
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    if H is not None:
        # Calcular erro de reprojeção
        src_pts_all = np.float32([kp1[m.queryIdx].pt for m in good_matches_test]).T
        src_pts_proj = cv2.perspectiveTransform(src_pts_all.reshape(-1, 1, 2), H)
        dst_pts_all = np.float32([kp2[m.trainIdx].pt for m in good_matches_test]).T
        
        errors = np.linalg.norm(src_pts_proj.reshape(-1, 2) - dst_pts_all.T, axis=1)
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        
        print(f"  Erro médio de reprojeção: {mean_error:.2f} pixels")
        print(f"  Erro mediano: {median_error:.2f} pixels")
        print(f"  Inliers (< 5px): {np.sum(errors < 5)} / {len(errors)}")

# Visualizar
img_result = cv2.drawMatches(img_original, kp1, img_warped, kp2,
                             good_matches_test[:30], None,
                             matchColor=(0, 255, 0), flags=2)

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
axes[0].imshow(cv2.cvtColor(img_original, cv2.COLOR_BGR2RGB))
axes[0].set_title('Imagem Original')
axes[0].axis('off')

axes[1].imshow(cv2.cvtColor(img_result, cv2.COLOR_BGR2RGB))
axes[1].set_title(f'Appariement com Transformação Afim ({len(good_matches_test)} matches)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('results/q9_quantitative_evaluation.png', dpi=100, bbox_inches='tight')
print("Imagem salva: results/q9_quantitative_evaluation.png\n")
plt.close()

print("""
CRITÉRIOS DE AVALIAÇÃO QUANTITATIVA:

1. Número de casamentos válidos
2. Razão inlier/outlier (após RANSAC)
3. Erro de reprojeção (distância entre pontos transformados esperados vs observados)
4. Precisão: % de verdadeiros positivos (TP) / (TP + FP)
5. Recall: % de TP / (TP + FN)
6. F-measure: Média harmônica de Precisão e Recall
""")

print("\n" + "="*70)
print("Processamento Q7-Q9 concluído!")
print("="*70)
