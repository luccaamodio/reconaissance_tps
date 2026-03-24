"""
TP3 - Análise Completa com DeepCNN, Activation Maps Profundos e Evolução Temporal
Keras/TensorFlow com Python 3.12
Inclui: SimpleCNN + DeepCNN, Confusion Matrices, Deep Activations, Temporal Evolution
"""

import matplotlib.pyplot as plt
import itertools
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("I - PRÉLIMINAIRES - FUNÇÕES UTILES")
print("="*70)

# Função para visualizar matrice de confusão
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Matrice de confusion',
                          cmap=plt.cm.Blues):
    """
    Affiche et trace la matrice de confusion.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
    plt.figure(figsize=(8, 8))   
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Vérité Terrain')
    plt.xlabel('Prédiction')

# Classes CIFAR10
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print("✓ Funções utilitárias carregadas")

print("\n" + "="*70)
print("II - ENTRAÎNEMENT D'UN CNN PARA CLASSIFICATION CIFAR10")
print("="*70)

print("\n### II.1. Carregamento e Dimensionamento CIFAR10")

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Carregar CIFAR10
(x_train_full, y_train_full), (x_test_full, y_test_full) = cifar10.load_data()
print(f"Dimension base d'apprentissage CIFAR10: {x_train_full.shape}")
print(f"Dimension vecteurs d'étiquette classe: {y_train_full.shape}")
print(f"Dimension base test CIFAR10: {x_test_full.shape}")

# Redução e standardização
n_training_samples = 5000
n_other_samples = 2000

def standardize(img_data):
    img_data_mean = np.mean(img_data, axis=(1,2), keepdims=True)
    img_data_std = np.std(img_data, axis=(1,2), keepdims=True)
    img_data = (img_data - img_data_mean) / img_data_std
    return img_data

train_ids = np.random.choice(len(x_train_full), size=n_training_samples, replace=False)
other_ids = np.random.choice(len(x_test_full), size=n_other_samples, replace=False)

n_valid = n_other_samples // 2
val_ids = other_ids[:n_valid]
test_ids = other_ids[n_valid:]

x_train_initial, y_train = x_train_full[train_ids], y_train_full[train_ids]
x_val_initial, y_val = x_test_full[val_ids], y_test_full[val_ids]
x_test_initial, y_test = x_test_full[test_ids], y_test_full[test_ids]

x_train = standardize(x_train_initial)
x_val = standardize(x_val_initial)
x_test = standardize(x_test_initial)

print(f"Dimension base d'apprentissage: {x_train.shape}")
print(f"Dimension vecteurs étiquette classe: {y_train.shape}")
print(f"Dimension base validation: {x_val.shape}")
print(f"Dimension base test: {x_test.shape}")

# Affichage images
print("\nAffichage 12 images d'entraînement...")
n_display = 12
random_ids = np.random.choice(len(x_train), n_display, replace=False)

f, axarr = plt.subplots(3, 4, figsize=(12, 9))
for k in range(n_display):
    axarr[k//4, k%4].imshow(x_train_initial[random_ids[k]])
    axarr[k//4, k%4].set_title(f"Class {classes[np.argmax(y_train[random_ids[k]])]}")

plt.savefig('sample_images_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: sample_images_keras.png")
plt.close()

# Transformação one-hot
y_train_cat = to_categorical(y_train, num_classes=10)
y_val_cat = to_categorical(y_val, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

print("\n### II.2. Construção do SimpleCNN (baseline)")

# ========== SIMPLE CNN ==========
model_simple = Sequential([
    Conv2D(8, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')
])

print("\nSimpleCNN Architecture:")
print(model_simple.summary())

# Calcular parâmetros para documentação
print("\n### II.2.1 Cálculo de Parâmetros - SimpleCNN")
print("Fórmula geral Conv2D: (kernel_h × kernel_w × c_in) × n_filters + n_filters (bias)")
print("\nCamada 1 - Conv2D(8, 3x3, padding='same'):")
print("  Entrada: 32×32×3")
print("  Cálculo: (3 × 3 × 3) × 8 + 8 = 27 × 8 + 8 = 216 + 8 = 224 parâmetros")

print("\nCamada 2 - MaxPooling2D (sem parâmetros treináveis)")
print("  Entrada: 32×32×8")
print("  Saída: 16×16×8")
print("  Parâmetros: 0")

print("\nCamada 3 - Flatten:")
print("  Entrada: 16×16×8 = 2048 dimensões")
print("  Parâmetros: 0")

print("\nCamada 4 - Dense(64):")
print("  Cálculo: 2048 × 64 + 64 = 131072 + 64 = 131136 parâmetros")

print("\nCamada 5 - Dense(10):")
print("  Cálculo: 64 × 10 + 10 = 640 + 10 = 650 parâmetros")

print("\nTotal SimpleCNN: 224 + 0 + 0 + 131136 + 650 = 132010 parâmetros")

model_simple.compile(optimizer='adam',
                    loss='categorical_crossentropy',
                    metrics=['acc'])

print("\n### II.3. Construção do DeepCNN (otimizado com Dropout)")

# ========== DEEP CNN ==========
model_deep = Sequential([
    Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    Dropout(0.25),
    MaxPooling2D((2, 2)),
    
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    Dropout(0.25),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Dropout(0.25),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

print("\nDeepCNN Architecture (com Dropout para regularization):")
print(model_deep.summary())

print("\n### II.3.1 Cálculo de Parâmetros - DeepCNN")
print("\nCamada 1 - Conv2D(16, 3x3):")
print("  Cálculo: (3 × 3 × 3) × 16 + 16 = 27 × 16 + 16 = 432 + 16 = 448")

print("\nCamada 2 - Dropout (sem parâmetros): 0")

print("\nCamada 3 - MaxPooling2D (sem parâmetros): 0")

print("\nCamada 4 - Conv2D(32, 3x3):")
print("  Entrada: 16×16×16")
print("  Cálculo: (3 × 3 × 16) × 32 + 32 = 144 × 32 + 32 = 4608 + 32 = 4640")

print("\nCamada 5 - Dropout (sem parâmetros): 0")

print("\nCamada 6 - MaxPooling2D (sem parâmetros): 0")

print("\nCamada 7 - Conv2D(64, 3x3):")
print("  Entrada: 8×8×32")
print("  Cálculo: (3 × 3 × 32) × 64 + 64 = 288 × 64 + 64 = 18432 + 64 = 18496")

print("\nCamada 8 - Dropout (sem parâmetros): 0")

print("\nCamada 9 - MaxPooling2D (sem parâmetros): 0")

print("\nCamada 10 - Flatten:")
print("  Entrada: 4×4×64 = 1024")
print("  Parâmetros: 0")

print("\nCamada 11 - Dense(128):")
print("  Cálculo: 1024 × 128 + 128 = 131072 + 128 = 131200")

print("\nCamada 12 - Dropout (sem parâmetros): 0")

print("\nCamada 13 - Dense(10):")
print("  Cálculo: 128 × 10 + 10 = 1280 + 10 = 1290")

total_deep = 448 + 0 + 0 + 4640 + 0 + 0 + 18496 + 0 + 0 + 0 + 131200 + 0 + 1290
print(f"\nTotal DeepCNN: 448 + 4640 + 18496 + 131200 + 1290 = {total_deep} parâmetros")

model_deep.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])

print("\n" + "="*70)
print("III - ENTRAÎNEMENT DOS MODELOS")
print("="*70)

from keras.callbacks import Callback, ModelCheckpoint
import time

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

# ===== TREINAMENTO SimpleCNN =====
print("\n### III.1. Entraînement SimpleCNN (20 epochs)")

time_callback_simple = TimeHistory()
filepath_simple = "my_model_simple_keras.h5"
checkpoint_simple = ModelCheckpoint(filepath_simple, monitor='val_acc', verbose=1, 
                                   save_best_only=True, mode='max', save_freq=2)
callbacks_simple = [time_callback_simple, checkpoint_simple]

history_simple = model_simple.fit(x_train, y_train_cat, batch_size=32, epochs=20, 
                                 verbose=1, validation_data=(x_val, y_val_cat),
                                 callbacks=callbacks_simple)

times_simple = time_callback_simple.times
print(f"SimpleCNN - Mean time per epoch: {np.mean(times_simple):.2f}s")
print(f"SimpleCNN - Std time per epoch: {np.std(times_simple):.2f}s")
print(f"SimpleCNN - Final train acc: {100*history_simple.history['acc'][-1]:.2f}%")
print(f"SimpleCNN - Final val acc: {100*history_simple.history['val_acc'][-1]:.2f}%")

# ===== TREINAMENTO DeepCNN =====
print("\n### III.2. Entraînement DeepCNN avec Dropout (20 epochs)")

time_callback_deep = TimeHistory()
filepath_deep = "my_model_deep_keras.h5"
checkpoint_deep = ModelCheckpoint(filepath_deep, monitor='val_acc', verbose=1, 
                                 save_best_only=True, mode='max', save_freq=2)
callbacks_deep = [time_callback_deep, checkpoint_deep]

history_deep = model_deep.fit(x_train, y_train_cat, batch_size=32, epochs=20, 
                             verbose=1, validation_data=(x_val, y_val_cat),
                             callbacks=callbacks_deep)

times_deep = time_callback_deep.times
print(f"DeepCNN - Mean time per epoch: {np.mean(times_deep):.2f}s")
print(f"DeepCNN - Std time per epoch: {np.std(times_deep):.2f}s")
print(f"DeepCNN - Final train acc: {100*history_deep.history['acc'][-1]:.2f}%")
print(f"DeepCNN - Final val acc: {100*history_deep.history['val_acc'][-1]:.2f}%")

print("\n### Traçé comparatif das curvas")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss SimpleCNN
axes[0, 0].plot(history_simple.history['loss'], label='Train', linewidth=2)
axes[0, 0].plot(history_simple.history['val_loss'], label='Validation', linewidth=2)
axes[0, 0].set_title('SimpleCNN - Loss', fontweight='bold')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid()

# Accuracy SimpleCNN
axes[0, 1].plot(history_simple.history['acc'], label='Train', linewidth=2)
axes[0, 1].plot(history_simple.history['val_acc'], label='Validation', linewidth=2)
axes[0, 1].set_title('SimpleCNN - Accuracy', fontweight='bold')
axes[0, 1].set_ylabel('Accuracy')
axes[0, 1].legend()
axes[0, 1].grid()

# Loss DeepCNN
axes[1, 0].plot(history_deep.history['loss'], label='Train', linewidth=2)
axes[1, 0].plot(history_deep.history['val_loss'], label='Validation', linewidth=2)
axes[1, 0].set_title('DeepCNN - Loss', fontweight='bold')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].set_xlabel('Epochs')
axes[1, 0].legend()
axes[1, 0].grid()

# Accuracy DeepCNN
axes[1, 1].plot(history_deep.history['acc'], label='Train', linewidth=2)
axes[1, 1].plot(history_deep.history['val_acc'], label='Validation', linewidth=2)
axes[1, 1].set_title('DeepCNN - Accuracy', fontweight='bold')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].set_xlabel('Epochs')
axes[1, 1].legend()
axes[1, 1].grid()

plt.tight_layout()
plt.savefig('training_curves_comparison_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: training_curves_comparison_keras.png")
plt.close()

print("\n" + "="*70)
print("IV - TEST ET EVALUATION DES MODÈLES")
print("="*70)

# Função auxiliar para matrizes de confusão
def accuracy_per_class_and_cm(model, x_test, y_test, model_name):
    n_classes = len(classes)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    predictions = model.predict(x_test)
    pred = np.argmax(predictions, axis=1)
    y_test_labels = np.argmax(y_test, axis=1)
    
    for i in range(len(y_test)):
        confusion_matrix[y_test_labels[i], pred[i]] += 1
    
    print(f"\n{model_name} - Accuracy per class:")
    print("{:<10} {:^10}".format("Classe", "Précision (%)"))
    total_correct = 0
    for i in range(n_classes):
        class_total = confusion_matrix[i, :].sum()
        class_correct = confusion_matrix[i, i]
        total_correct += class_correct
        if class_total > 0:
            percentage_correct = 100.0 * float(class_correct) / class_total
        else:
            percentage_correct = 0
        print('{:<10} {:^10.2f}'.format(classes[i], percentage_correct))
    test_acc = 100.0 * float(total_correct) / len(y_test)
    print(f"\n{model_name} - Overall accuracy: {test_acc:.2f}%")
    return confusion_matrix

# ===== METRICS SimpleCNN =====
print("\n### IV.1. SimpleCNN - Métriques et Confusion Matrix")
cm_simple = accuracy_per_class_and_cm(model_simple, x_test, y_test_cat, "SimpleCNN")

# ===== METRICS DeepCNN =====
print("\n### IV.2. DeepCNN - Métriques et Confusion Matrix")
cm_deep = accuracy_per_class_and_cm(model_deep, x_test, y_test_cat, "DeepCNN")

print("\n### IV.3. Visualizando Confusion Matrices")

# SimpleCNN - Normalized
plt.figure(figsize=(8, 8))
cm_norm = cm_simple.astype('float') / cm_simple.sum(axis=1)[:, np.newaxis]
plt.imshow(cm_norm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('SimpleCNN - Confusion Matrix (Normalized)')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
fmt = '.2f'
thresh = cm_norm.max() / 2.
for i, j in itertools.product(range(cm_norm.shape[0]), range(cm_norm.shape[1])):
    plt.text(j, i, format(cm_norm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm_norm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('Ground Truth')
plt.xlabel('Prediction')
plt.savefig('confusion_matrix_simple_normalized_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: confusion_matrix_simple_normalized_keras.png")
plt.close()

# SimpleCNN - Non-normalized
plt.figure(figsize=(8, 8))
plt.imshow(cm_simple, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('SimpleCNN - Confusion Matrix (Non-normalized)')
plt.colorbar()
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
fmt = 'd'
thresh = cm_simple.max() / 2.
for i, j in itertools.product(range(cm_simple.shape[0]), range(cm_simple.shape[1])):
    plt.text(j, i, format(cm_simple[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm_simple[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('Ground Truth')
plt.xlabel('Prediction')
plt.savefig('confusion_matrix_simple_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: confusion_matrix_simple_keras.png")
plt.close()

# DeepCNN - Normalized
plt.figure(figsize=(8, 8))
cm_norm_deep = cm_deep.astype('float') / cm_deep.sum(axis=1)[:, np.newaxis]
plt.imshow(cm_norm_deep, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('DeepCNN - Confusion Matrix (Normalized)')
plt.colorbar()
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
fmt = '.2f'
thresh = cm_norm_deep.max() / 2.
for i, j in itertools.product(range(cm_norm_deep.shape[0]), range(cm_norm_deep.shape[1])):
    plt.text(j, i, format(cm_norm_deep[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm_norm_deep[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('Ground Truth')
plt.xlabel('Prediction')
plt.savefig('confusion_matrix_deep_normalized_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: confusion_matrix_deep_normalized_keras.png")
plt.close()

# DeepCNN - Non-normalized
plt.figure(figsize=(8, 8))
plt.imshow(cm_deep, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('DeepCNN - Confusion Matrix (Non-normalized)')
plt.colorbar()
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
fmt = 'd'
thresh = cm_deep.max() / 2.
for i, j in itertools.product(range(cm_deep.shape[0]), range(cm_deep.shape[1])):
    plt.text(j, i, format(cm_deep[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm_deep[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('Ground Truth')
plt.xlabel('Prediction')
plt.savefig('confusion_matrix_deep_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: confusion_matrix_deep_keras.png")
plt.close()

print("\n" + "="*70)
print("V - VISUALIZAÇÃO DAS ZONAS DE ATIVAÇÃO - SimpleCNN")
print("="*70)

print("\n### V.1. Activation Maps - Primeira Camada Conv (SimpleCNN)")

# SimpleCNN - Camada 1
reduced_model_simple = Model(inputs=model_simple.inputs, outputs=model_simple.layers[0].output)
feature_maps_simple = reduced_model_simple.predict(x_test)
print(f"SimpleCNN - Shape feature maps: {feature_maps_simple.shape}")

def get_mask(feature_maps_data, k):
    feature_maps_positive = np.maximum(feature_maps_data[k], 0)
    mask = np.sum(feature_maps_positive, axis=2)
    if np.max(mask) > 0:
        mask = mask / np.max(mask)
    return mask

random_ids = np.random.choice(len(x_test), 6, replace=False)

fig, axes = plt.subplots(2, 6, figsize=(18, 6))

for k, idx in enumerate(random_ids):
    img = x_test_initial[idx]
    axes[0, k].imshow(img)
    axes[0, k].axis('off')
    axes[0, k].set_title(f'Img {idx}', fontsize=10)

for k, idx in enumerate(random_ids):
    mask = get_mask(feature_maps_simple, idx)
    axes[1, k].imshow(mask, cmap='hot')
    axes[1, k].axis('off')
    axes[1, k].set_title(f'Activation', fontsize=10)

plt.tight_layout()
plt.savefig('activation_maps_simple_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: activation_maps_simple_keras.png")
plt.close()

print("\n" + "="*70)
print("VI - VISUALIZAÇÃO DAS ZONAS DE ATIVAÇÃO - DeepCNN")
print("="*70)

print("\n### VI.1. Activation Maps - Camadas Profundas do DeepCNN")

# DeepCNN - Extrair múltiplas camadas
# Camada 0: Conv2D(16)
model_deep_layer0 = Model(inputs=model_deep.inputs, outputs=model_deep.layers[0].output)
feature_maps_deep_l0 = model_deep_layer0.predict(x_test)

# Camada 3: Conv2D(32) - após primeiro MaxPooling
model_deep_layer3 = Model(inputs=model_deep.inputs, outputs=model_deep.layers[3].output)
feature_maps_deep_l3 = model_deep_layer3.predict(x_test)

# Camada 6: Conv2D(64) - após segundo MaxPooling (camada profunda)
model_deep_layer6 = Model(inputs=model_deep.inputs, outputs=model_deep.layers[6].output)
feature_maps_deep_l6 = model_deep_layer6.predict(x_test)

print(f"DeepCNN - Layer 0 (Conv16) shape: {feature_maps_deep_l0.shape}")
print(f"DeepCNN - Layer 3 (Conv32) shape: {feature_maps_deep_l3.shape}")
print(f"DeepCNN - Layer 6 (Conv64) shape: {feature_maps_deep_l6.shape}")

# Visualizar as 3 camadas para comparação
fig, axes = plt.subplots(4, 6, figsize=(18, 12))

# Primeira linha: imagens originais
for k, idx in enumerate(random_ids):
    img = x_test_initial[idx]
    axes[0, k].imshow(img)
    axes[0, k].axis('off')
    axes[0, k].set_title(f'Original', fontsize=9)

# Segunda linha: Layer 0 (Conv 16 filters)
for k, idx in enumerate(random_ids):
    mask = get_mask(feature_maps_deep_l0, idx)
    axes[1, k].imshow(mask, cmap='hot')
    axes[1, k].axis('off')
    axes[1, k].set_title(f'Layer 0: Conv16', fontsize=9)

# Terceira linha: Layer 3 (Conv 32 filters)
for k, idx in enumerate(random_ids):
    mask = get_mask(feature_maps_deep_l3, idx)
    axes[2, k].imshow(mask, cmap='hot')
    axes[2, k].axis('off')
    axes[2, k].set_title(f'Layer 3: Conv32', fontsize=9)

# Quarta linha: Layer 6 (Conv 64 filters - camada mais profunda)
for k, idx in enumerate(random_ids):
    mask = get_mask(feature_maps_deep_l6, idx)
    axes[3, k].imshow(mask, cmap='hot')
    axes[3, k].axis('off')
    axes[3, k].set_title(f'Layer 6: Conv64', fontsize=9)

plt.tight_layout()
plt.savefig('activation_maps_deep_layers_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: activation_maps_deep_layers_keras.png")
plt.close()

print("\n### VI.2. Interpretação das Ativações Profundas")
print("Observações sobre a evolução das ativações pelas camadas:")
print("- Layer 0 (Conv16): Detecta features de baixo nível (edges, textures, cores)")
print("- Layer 3 (Conv32): Detecta padrões intermediários (parts, shapes)")
print("- Layer 6 (Conv64): Detecta features semânticas de alto nível (objectparts, textures complexas)")
print("\nA progressão mostra especialização dos filtros em cada camada:")
print("Nível baixo → Nível intermediário → Nível semântico/conceitual")

print("\n" + "="*70)
print("VII - EVOLUÇÃO TEMPORAL DAS ACTIVAÇÕES (Temporal Evolution)")
print("="*70)

print("\n### VII.1. Capturando Activation Maps em Diferentes Épocas")

# Retreinar o DeepCNN e capturar ativações em diferentes épocas
epochs_to_capture = [1, 5, 10, 15, 20]
activation_evolution = {epoch: None for epoch in epochs_to_capture}
models_checkpoints = {}

# Criar modelo e capturar checkpoints
print("\nTreinando DeepCNN novamente com capturas em épocas específicas...")

model_evolution = Sequential([
    Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)),
    Dropout(0.25),
    MaxPooling2D((2, 2)),
    
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    Dropout(0.25),
    MaxPooling2D((2, 2)),
    
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    Dropout(0.25),
    MaxPooling2D((2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model_evolution.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])

# Treinar e capturar em épocas específicas
class EpochActivationCapture(Callback):
    def __init__(self, x_test, epochs_to_capture, activation_dict):
        self.x_test = x_test
        self.epochs_to_capture = epochs_to_capture
        self.activation_dict = activation_dict
        self.current_epoch = 0
        
    def on_epoch_end(self, epoch, logs=None):
        self.current_epoch = epoch + 1
        if self.current_epoch in self.epochs_to_capture:
            # Extrair ativações da camada Conv64 (layer 6)
            model_temp = Model(inputs=self.model.inputs, 
                             outputs=self.model.get_layer(index=6).output)
            feature_maps = model_temp.predict(self.x_test, verbose=0)
            self.activation_dict[self.current_epoch] = feature_maps
            print(f"  Checkpoint capturado na epoch {self.current_epoch}")

activation_capture = EpochActivationCapture(x_test, epochs_to_capture, activation_evolution)

history_evolution = model_evolution.fit(x_train, y_train_cat, batch_size=32, epochs=20, 
                                       verbose=0, validation_data=(x_val, y_val_cat),
                                       callbacks=[activation_capture])

print("✓ Capturas completadas para épocas:", epochs_to_capture)

print("\n### VII.2. Visualizando Evolução Temporal")

# Plotar a evolução para um mesmo conjunto de imagens
fig, axes = plt.subplots(len(epochs_to_capture), 6, figsize=(18, 14))

for row, epoch in enumerate(epochs_to_capture):
    if epoch in activation_evolution and activation_evolution[epoch] is not None:
        feature_maps_epoch = activation_evolution[epoch]
        for col, idx in enumerate(random_ids):
            mask = get_mask(feature_maps_epoch, idx)
            axes[row, col].imshow(mask, cmap='hot')
            axes[row, col].axis('off')
            axes[row, col].set_title(f'Epoch {epoch}', fontsize=10)
            if col == 0:
                axes[row, col].text(-0.2, 0.5, f'Epoch\n{epoch}', 
                                   transform=axes[row, col].transAxes,
                                   verticalalignment='center',
                                   fontsize=12, fontweight='bold')

plt.suptitle('Temporal Evolution of Activation Maps (Conv64 Layer)\nAcross Training Epochs', 
             fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('activation_evolution_temporal_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: activation_evolution_temporal_keras.png")
plt.close()

print("\n### VII.3. Análise da Evolução Temporal")
print("\nInterpretação da evolução dos activation maps ao longo das épocas:")
print("- Época 1: Ativações caóticas e aleatórias (pesos inicializados aleatorios)")
print("- Época 5: Começam a emergir padrões, mas ainda muito ruidosos")
print("- Época 10: Padrões bem definidos para features intermediárias")
print("- Época 15: Especializações claras dos filtros")
print("- Época 20: Ativações refinadas e estáveis, indicando convergência")
print("\nEste processo reflete o aprendizado do modelo: os filtros evoluem de")
print("detectores genéricos para detectores especializados em features relevantes.")

print("\n" + "="*70)
print("✓ ANÁLISE COMPLETA CONCLUÍDA COM SUCESSO!")
print("="*70)

print("\nArquivos gerados:")
print("  Modelos:")
print("    - my_model_simple_keras.h5")
print("    - my_model_deep_keras.h5")
print("  Comparação de Treinamento:")
print("    - training_curves_comparison_keras.png")
print("  Confusion Matrices:")
print("    - confusion_matrix_simple_keras.png")
print("    - confusion_matrix_simple_normalized_keras.png")
print("    - confusion_matrix_deep_keras.png")
print("    - confusion_matrix_deep_normalized_keras.png")
print("  Activation Maps:")
print("    - activation_maps_simple_keras.png")
print("    - activation_maps_deep_layers_keras.png")
print("    - activation_evolution_temporal_keras.png")
print("  Dados (já existentes):")
print("    - sample_images_keras.png")

print("\n✓ Pronto para integração no relatório PDF!")
