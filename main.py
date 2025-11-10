import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

#Se inicializa la ruta de la carpeta de training y de validacion
#Train tiene 80% de los datos
TRAIN_DIR = "assets/train"
#Val tiene 20% de los datos
#Test tiene 10% de los datos
VAL_DIR = "assets/val"

#Tamaño y propiedades de las imagenes durante el testeo
IMG_SIZE = (128, 128)
# batch_size  = la cantidad de ejemplos de entrenamiento
# en un ciclo. Cuánto mayor sea el tamaño del lote, más espacio de memoria necesitará.
BATCH_SIZE = 32
SEED = 123

#Crear dataset de training
train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED
)

#Crear dataset de validacion
val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=SEED
)

#Info clases
class_names = train_ds.class_names
num_classes = len(class_names)

#Comprobacion de estar leyendo todas las clases (debe ser 20)
print("Clases detectadas:", num_classes)

#Impresion titulos clases
print(class_names[:10], "...")

# Mostrar un ejemplo de cada clase en grafico
for images, labels in train_ds.take(1):
    plt.figure(figsize=(10,8))
    for i in range(9):
        ax = plt.subplot(3,3,i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        label_index = tf.argmax(labels[i]).numpy()
        plt.title(class_names[label_index])
        plt.axis("off")
    plt.show()
    break

#Prefetch para rendimiento
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

#Modelo transfer learning
base = EfficientNetB0(
    include_top=False,
    input_shape=(*IMG_SIZE, 3),
    weights='imagenet'
)
base.trainable = False  # Congelar capa base

x = layers.GlobalAveragePooling2D()(base.output)
x = layers.Dropout(0.3)(x)
x = layers.Dense(256, activation='relu')(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)

model = models.Model(inputs=base.input, outputs=outputs)

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()
history_ft = None #inicializo var

#Muestra la tabla con cada parte del modelo
#Layer(type): el nombre y tipo de capa
#output shape: el tamaño de tensor de la capa. None es el tamaño del batch
#Param #: Cuántos parámetros (pesos + sesgos) tiene esa capa. La red ajusta estos numeros en entrenamiento
#Connected to: a que capa anterior esta conectada. Muestra flujo de datos entre capas

#Total params: Parametros entrenables y no entrenables del modelo
#Trainable params: Parametros entrenables que el modelo optimiza
#non-trainable params: parametros congelados que no cambian

#Callback (control del entrenamiento en tiempo de ejecucion)
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
]

#Registrar con history las perdidas y metricas del rendimiento del modelo
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5,
    callbacks=callbacks
)

#Accuracy: precision obtenida sobre el conjunto de entrenamiento
#Val_accuracy: precision sobre la validacion
#Los y val_loss: errores respecto a la prediccion


#curvas de entrenamiento, overfitting
def plot_history(h, title=""):
    plt.figure(figsize=(6,3))
    plt.plot(h.history["accuracy"], label="train_acc")
    plt.plot(h.history["val_accuracy"], label="val_acc")
    plt.title("Accuracy " + title); plt.legend(); plt.tight_layout(); plt.show()

    plt.figure(figsize=(6,3))
    plt.plot(h.history["loss"], label="train_loss")
    plt.plot(h.history["val_loss"], label="val_loss")
    plt.title("Loss " + title); plt.legend(); plt.tight_layout(); plt.show()

plot_history(history, "(head)")
try: plot_history(history_ft, "(fine-tune)")
except: pass

# matriz de confusión
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

def ytrue_ypred(ds, model):
    yt, yp = [], []
    for x, y in ds:
        p = model.predict(x, verbose=0)
        yp.extend(np.argmax(p, 1))
        yt.extend(np.argmax(y.numpy(), 1))
    return np.array(yt), np.array(yp)

# Evaluar dataset de validación o test
eval_ds = globals().get('test_ds', val_ds)
yt, yp = ytrue_ypred(eval_ds, model)

# Reporte completo por clase
print("\nclasificacion x clase")
print(classification_report(yt, yp, target_names=class_names, digits=4))

# Matriz normalizada (valores de 0 a 1)
cm = confusion_matrix(yt, yp, labels = np.arange(len(class_names)), normalize="true")

plt.figure(figsize=(12,10))
im = plt.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect='auto')
plt.title("Matriz de confusión (normalizada)")
plt.colorbar(im, fraction=0.046, pad=0.04)
ticks = np.arange(len(class_names))
plt.xticks(ticks, class_names, rotation=90, fontsize=7)
plt.yticks(ticks, class_names, fontsize=7)

m = cm.max() if cm.size else 1
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                 color="white" if cm[i,j] > m/2 else "black", fontsize=6)

plt.xlabel("Predicho"); plt.ylabel("Real")
plt.tight_layout()
print("Si no se puede mostrar el grafico se mostrara por defecto la imagen png de la matriz")
plt.savefig("confusion_matrix_full.png", dpi=200)
plt.show()
print("Guardado: confusion_matrix_full.png")

# Top-3 / Top-5
top3 = tf.keras.metrics.TopKCategoricalAccuracy(k=3)
top5 = tf.keras.metrics.TopKCategoricalAccuracy(k=5)
for x, y in eval_ds:
    p = model.predict(x, verbose=0)
    top3.update_state(y, p); top5.update_state(y, p)
print("Top-3 acc:", float(top3.result().numpy()))
print("Top-5 acc:", float(top5.result().numpy()))

#accuracy y loss en funcion de las epocas
if 'accuracy' in history.history and 'val_accuracy' in history.history:
    plt.figure(figsize=(8,5))
    plt.plot(history.history['accuracy'], label='Entrenamiento', color='blue')
    plt.plot(history.history['val_accuracy'], label='Validación', color='orange')
    plt.plot(history.history['loss'], label='Pérdida entrenamiento', linestyle='--', color='blue')
    plt.plot(history.history['val_loss'], label='Pérdida validación', linestyle='--', color='orange')
    plt.title('Curvas de Accuracy y Loss')
    plt.xlabel('Épocas')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("No hay datos de accuracy/loss disponibles para graficar.")

#curva de aprendizaje suavizada
import pandas as pd

acc = pd.Series(history.history['accuracy']).rolling(2).mean()
val_acc = pd.Series(history.history['val_accuracy']).rolling(2).mean()

plt.figure(figsize=(8,4))
plt.plot(acc, label='Entrenamiento')
plt.plot(val_acc, label='Validación')
plt.title('Curva de aprendizaje suavizada')
plt.xlabel('Épocas')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

# toma 9 imágenes del eval_ds y muestra true vs pred
cnt = 0
print("9 Casos de prediccion:")
plt.figure(figsize=(10,10))
for x_batch, y_batch in eval_ds.take(3):  # toma algunos batches
    preds = model.predict(x_batch, verbose=0)
    for i in range(len(x_batch)):
        if cnt >= 9: break
        plt.subplot(3,3,cnt+1)
        img = x_batch[i].numpy().astype("uint8")
        plt.imshow(img)
        true = np.argmax(y_batch[i].numpy())
        pred = np.argmax(preds[i])
        plt.title(f"Real: {class_names[true]}\nPred: {class_names[pred]}")
        plt.axis("off")
        cnt += 1
    if cnt >= 9: break
plt.tight_layout()
plt.show()

#resultados
final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]
print(f"\n Precisión final entrenamiento: {final_train_acc:.4f}")
print(f" Precisión final validación: {final_val_acc:.4f}")
print(f" Diferencia (posible overfitting): {abs(final_train_acc - final_val_acc):.4f}")
