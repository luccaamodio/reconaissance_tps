"""
TP3 - Replicação exata do notebook do professor
Usando Keras/TensorFlow com Python 3.12
"""

import matplotlib.pyplot as plt
import itertools
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("I - PRÉLIMINAIRES - FUNÇÕES UTEIS")
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
print("II - ENTRAÎNEMENT D'UN CNN POUR CLASSIFICATION CIFAR10")
print("="*70)

print("\n### II.1. Carregamento e Dimensionamento CIFAR10")

from keras.datasets import cifar10
from keras.utils import to_categorical

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
f, axarr = plt.subplots(1, n_display, figsize=(16, 16))
for k in range(n_display):
    axarr[k].imshow(x_train_initial[random_ids[k]])
    axarr[k].title.set_text(classes[y_train[random_ids[k]][0]])
plt.savefig('sample_images_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: sample_images_keras.png")
plt.close()

# Converter labels para one-hot
y_train = to_categorical(y_train)
y_val = to_categorical(y_val)
y_test = to_categorical(y_test)

print(f"Dimension matrices étiquette classe (train): {y_train.shape}")
print(f"Dimension matrices étiquette classe (val): {y_val.shape}")
print(f"Dimension matrices étiquette classe (test): {y_test.shape}")

print("\n### II.2. Definição da arquitetura do CNN")

from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, Activation, Input
from keras.regularizers import l2

model = Sequential()
model.add(Input(shape=(32, 32, 3)))
model.add(Conv2D(filters=8, 
                kernel_size=(3, 3),
                activation='relu',
                padding='same',
                kernel_regularizer=l2(0.00)))
model.add(Dropout(0.0))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu', kernel_regularizer=l2(0.00)))
model.add(Dropout(0.0))
model.add(Dense(10, activation='softmax', kernel_regularizer=l2(0.00)))
model.add(Dropout(0.0))

print("✓ Architecture définie")

# Enregistrer poids iniciais
weights_init = model.get_weights()

print("\n### II.3. Definição da função de coût et optimisation")

from keras.optimizers import SGD

opt = SGD(learning_rate=0.01, momentum=0.0)
model.compile(optimizer=opt,
             loss='categorical_crossentropy',
             metrics=['acc'])

print(model.summary())

print("\n### II.4. Entraînement du CNN")

from keras.callbacks import Callback, ModelCheckpoint
import time

class TimeHistory(Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)

time_callback = TimeHistory()
filepath = "my_model_keras.h5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_freq=2)
callbacks = [time_callback, checkpoint]

print("\nEntraînement en cours... (20 epochs)")
history = model.fit(x_train, y_train, batch_size=32, epochs=20, verbose=1, validation_data=(x_val, y_val),
                    callbacks=callbacks)

# Statistiques temps
times = time_callback.times
print(f"\nMean time per epoch: {np.mean(times):.2f}s")
print(f"Std time per epoch: {np.std(times):.2f}s")

print("\n### Traçé des courbes de coût et précision")

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

epochs = range(1, len(history_dict['acc']) + 1)

fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.plot(epochs, loss_values, color='b', label='Training loss')
ax1.plot(epochs, val_loss_values, color='g', label='Validation loss')
ax1.tick_params(axis='y')
plt.legend()

ax2 = ax1.twinx()
ax2.set_ylabel('Accuracy')
ax2.plot(epochs, acc_values, color='c', label='Training accuracy')
ax2.plot(epochs, val_acc_values, color='r', label='Validation accuracy')
ax2.tick_params(axis='y')

fig.tight_layout()
plt.legend(loc=1)
plt.savefig('training_curves_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: training_curves_keras.png")
plt.close()

print("\n" + "="*70)
print("III - TEST ET EVALUATION DU MODÈLE EN PRÉDICTION")
print("="*70)

print("\n### III.1. Test du modèle sur données test")

random_ids = np.random.choice(len(x_test), n_display, replace=False)
pred = np.argmax(model.predict(x_test[random_ids]), axis=1)
f, axarr = plt.subplots(1, n_display, figsize=(16, 16))
for k in range(n_display):
    axarr[k].imshow(x_test_initial[random_ids[k]])
    axarr[k].title.set_text(classes[pred[k]])
plt.savefig('predictions_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: predictions_keras.png")
plt.close()

print("\n### Précision sur données")

print(f"Précision réseau sur {n_training_samples} images d'entraînement: {100 * history_dict['acc'][-1]:.2f}%")
print(f"Précision réseau sur {n_valid} images validation: {100 * history_dict['val_acc'][-1]:.2f}%")

def accuracy_per_class(model):
    n_classes = len(classes)
    confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    
    pred = np.argmax(model.predict(x_test), axis=1)
    for i in range(len(y_test)):
        confusion_matrix[np.argmax(y_test[i]), pred[i]] += 1
    
    print("\n{:<10} {:^10}".format("Classe", "Précision (%)"))
    total_correct = 0
    for i in range(n_classes):
        class_total = confusion_matrix[i, :].sum()
        class_correct = confusion_matrix[i, i]
        total_correct += class_correct
        percentage_correct = 100.0 * float(class_correct) / class_total
        print('{:<10} {:^10.2f}'.format(classes[i], percentage_correct))
    test_acc = 100.0 * float(total_correct) / len(y_test)
    print(f"\nPrécision réseau sur {len(y_test)} images test: {test_acc:.2f}%")
    return confusion_matrix

confusion_matrix = accuracy_per_class(model)

print("\n### III.2. Matrices de Confusion")

# Matrice normalisée
plt.figure(figsize=(8, 8))
cm = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de confusion normalisée')
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
fmt = '.2f'
thresh = cm.max() / 2.
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
    plt.text(j, i, format(cm[i, j], fmt),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('Vérité Terrain')
plt.xlabel('Prédiction')
plt.savefig('confusion_matrix_normalized_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: confusion_matrix_normalized_keras.png")
plt.close()

# Matriz não normalizada
plt.figure(figsize=(8, 8))
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('Matrice de confusion non normalisée')
plt.colorbar()
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
fmt = 'd'
thresh = confusion_matrix.max() / 2.
for i, j in itertools.product(range(confusion_matrix.shape[0]), range(confusion_matrix.shape[1])):
    plt.text(j, i, format(confusion_matrix[i, j], fmt),
             horizontalalignment="center",
             color="white" if confusion_matrix[i, j] > thresh else "black")
plt.tight_layout()
plt.ylabel('Vérité Terrain')
plt.xlabel('Prédiction')
plt.savefig('confusion_matrix_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: confusion_matrix_keras.png")
plt.close()

print("\n" + "="*70)
print("IV - VISUALIZAÇÃO DAS ZONAS DE ATIVAÇÃO")
print("="*70)

from keras.models import Model

# Extrair features da primeira camada conv
reduced_model = Model(inputs=model.inputs, outputs=model.layers[1].output)
print(reduced_model.summary())

feature_maps = reduced_model.predict(x_test)
print(f"Shape feature maps: {feature_maps.shape}")

def get_mask(k):
    feature_maps_positive = np.maximum(feature_maps[k], 0)
    mask = np.sum(feature_maps_positive, axis=2)
    mask = mask / np.max(mask)
    return mask

# Visualizar imagens e suas mascaras de ativação
random_ids = np.random.choice(len(x_test), 6, replace=False)

fig, axes = plt.subplots(2, 6, figsize=(18, 6))

# Primeira linha: imagens
for k, idx in enumerate(random_ids):
    img = x_test_initial[idx]
    axes[0, k].imshow(img)
    axes[0, k].axis('off')
    axes[0, k].set_title(f'Img {idx}')

# Segunda linha: mascaras
for k, idx in enumerate(random_ids):
    mask = get_mask(idx)
    axes[1, k].imshow(mask, cmap='hot')
    axes[1, k].axis('off')
    axes[1, k].set_title(f'Mask')

plt.tight_layout()
plt.savefig('activation_maps_keras.png', dpi=100, bbox_inches='tight')
print("✓ Figura salva: activation_maps_keras.png")
plt.close()

print("\n" + "="*70)
print("✓ TP3 COM KERAS COMPLETADO COM SUCESSO!")
print("="*70)

print("\nArquivos gerados:")
print("  - my_model_keras.h5 (modelo treinado)")
print("  - sample_images_keras.png")
print("  - training_curves_keras.png")
print("  - predictions_keras.png")
print("  - confusion_matrix_normalized_keras.png")
print("  - confusion_matrix_keras.png")
print("  - activation_maps_keras.png")
