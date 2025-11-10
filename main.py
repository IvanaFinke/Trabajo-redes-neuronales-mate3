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
