"""
Q1-Q3: Convolutions et Gradients
TP1 - Reconnaissances de Points Caractéristiques
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt

# ============================================================================
# Q1: Tester le code de convolution fourni
# ============================================================================
print("="*70)
print("Q1: Observation de la convolution - Méthode directe vs filter2D")
print("="*70)

# Lecture image en niveau de gris et conversion en float64
img_path = 'TP1_Features/Image_Pairs/FlowerGarden2.png'
img = np.float64(cv2.imread(img_path, 0))
(h, w) = img.shape
print(f"Dimension de l'image: {h} lignes x {w} colonnes")

# Méthode directe
print("\nMéthode 1: Balayage direct du tableau 2D")
t1 = cv2.getTickCount()
img2 = cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_REPLICATE)
for y in range(1, h-1):
    for x in range(1, w-1):
        val = 5*img[y, x] - img[y-1, x] - img[y, x-1] - img[y+1, x] - img[y, x+1]
        img2[y, x] = min(max(val, 0), 255)
t2 = cv2.getTickCount()
time_direct = (t2 - t1) / cv2.getTickFrequency()
print(f"  Temps : {time_direct:.4f} s")

# Méthode filter2D
print("\nMéthode 2: Utilisant cv2.filter2D")
t1 = cv2.getTickCount()
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
img3 = cv2.filter2D(img, -1, kernel)
t2 = cv2.getTickCount()
time_filter2d = (t2 - t1) / cv2.getTickFrequency()
print(f"  Temps : {time_filter2d:.4f} s")

print(f"\nSpeedup: {time_direct/time_filter2d:.1f}x plus rapide avec filter2D")

# Affichage et comparaison
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Image origiale')
axes[0].axis('off')

axes[1].imshow(img2, cmap='gray', vmin=0, vmax=255)
axes[1].set_title('Convolution - Méthode Directe')
axes[1].axis('off')

axes[2].imshow(img3, cmap='gray', vmin=0, vmax=255)
axes[2].set_title('Convolution - filter2D')
axes[2].axis('off')

plt.tight_layout()
plt.savefig('results/q1_convolution_comparison.png', dpi=100, bbox_inches='tight')
print("\nImage sauvegardée: results/q1_convolution_comparison.png")
plt.close()

# ============================================================================
# Q2: Explication du rehaussement de contraste
# ============================================================================
print("\n" + "="*70)
print("Q2: Rehaussement de contraste")
print("="*70)
print("""
EXPLICATION:
Le noyau de convolution [[0, -1, 0], [-1, 5, -1], [0, -1, 0]] effectue:
  A = 5*pixel_central - somme_des_4_voisins

Ce filtre amplifie les transitions d'intensité (arêtes). Pour un pixel:
- Si le pixel central est similaire à ses voisins, A ≈ 0 (région homogène)
- Si le pixel central est différent, A est grand (arête/contraste)

Donc le résultat rehausse les contours en augmentant les différences.
""")

# ============================================================================
# Q3: Calcul des composants du gradient et norme
# ============================================================================
print("\n" + "="*70)
print("Q3: Gradient (Ix, Iy et norme ||∇I||)")
print("="*70)
print("Calcul des gradients avec des opérateurs Sobel...")

# Utiliser les filtres Sobel (approximations du gradient)
# Filtre Sobel pour Ix (dérivée en x)
Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
Ix = cv2.filter2D(img, -1, Kx)

# Filtre Sobel pour Iy (dérivée en y)
Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)
Iy = cv2.filter2D(img, -1, Ky)

# Norme euclidienne du gradient
magnitude = np.sqrt(Ix**2 + Iy**2)

# Angle du gradient
# angle = np.arctan2(Iy, Ix)

print(f"Ix min: {Ix.min():.2f}, max: {Ix.max():.2f}")
print(f"Iy min: {Iy.min():.2f}, max: {Iy.max():.2f}")
print(f"||∇I|| min: {magnitude.min():.2f}, max: {magnitude.max():.2f}")

# PRÉCAUTIONS IMPORTANTES pour l'affichage:
# 1. Les gradients sont positifs et négatifs - normaliser pour affichage
# 2. Utiliser la valeur absolue pour visualiser les contours
# 3. Normaliser entre 0 et 255 pour affichage correct

# Normalisation pour affichage
Ix_norm = np.uint8(255.0 * (np.abs(Ix) / np.max(np.abs(Ix))))
Iy_norm = np.uint8(255.0 * (np.abs(Iy) / np.max(np.abs(Iy))))
magnitude_norm = np.uint8(255.0 * (magnitude / np.max(magnitude)))

# Affichage
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(img, cmap='gray')
axes[0, 0].set_title('Image originale')
axes[0, 0].axis('off')

axes[0, 1].imshow(Ix_norm, cmap='gray')
axes[0, 1].set_title('Gradient Ix (composante x)')
axes[0, 1].axis('off')

axes[1, 0].imshow(Iy_norm, cmap='gray')
axes[1, 0].set_title('Gradient Iy (composante y)')
axes[1, 0].axis('off')

axes[1, 1].imshow(magnitude_norm, cmap='gray')
axes[1, 1].set_title('Magnitude du gradient ||∇I||')
axes[1, 1].axis('off')

plt.tight_layout()
plt.savefig('results/q3_gradients.png', dpi=100, bbox_inches='tight')
print("Image sauvegardée: results/q3_gradients.png")
plt.close()

# Comparaison avec cv2.Sobel
print("\nComparaison avec cv2.Sobel:")
Ix_sobel = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
Iy_sobel = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
magnitude_sobel = np.sqrt(Ix_sobel**2 + Iy_sobel**2)

print(f"Gradient Sobel ||∇I|| min: {magnitude_sobel.min():.2f}, max: {magnitude_sobel.max():.2f}")

# Affichage
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

magnitude_norm_manual = np.uint8(255.0 * (magnitude / np.max(magnitude)))
magnitude_norm_sobel = np.uint8(255.0 * (magnitude_sobel / np.max(magnitude_sobel)))

axes[0].imshow(magnitude_norm_manual, cmap='gray')
axes[0].set_title('Magnitude (Filtre custom)')
axes[0].axis('off')

axes[1].imshow(magnitude_norm_sobel, cmap='gray')
axes[1].set_title('Magnitude (cv2.Sobel)')
axes[1].axis('off')

plt.tight_layout()
plt.savefig('results/q3_gradients_sobel_comparison.png', dpi=100, bbox_inches='tight')
print("Image sauvegardée: results/q3_gradients_sobel_comparison.png")
plt.close()

print("\n" + "="*70)
print("Traitement Q1-Q3 terminé!")
print("="*70)
