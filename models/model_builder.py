import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50V2

def build_resnet50v2_model(num_classes=4, input_shape=(224, 224, 3)):
    """
    Construye la arquitectura ResNet50V2 optimizada para clasificacion de
    patologias oculares en imagenes de fondo de ojo.
    
    Arquitectura ganadora (89.01% test accuracy):
    - Base: ResNet50V2 preentrenada en ImageNet (congelada inicialmente)
    - GlobalAveragePooling2D
    - BatchNormalization
    - Dense(512, activation='relu') + Dropout(0.5)
    - Dense(256, activation='relu') + Dropout(0.3)
    - Dense(num_classes, activation='softmax')
    """
    base_model = ResNet50V2(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="batch_norm")(x)
    x = layers.Dense(512, activation="relu", name="dense_512")(x)
    x = layers.Dropout(0.5, name="dropout_512")(x)
    x = layers.Dense(256, activation="relu", name="dense_256")(x)
    x = layers.Dropout(0.3, name="dropout_256")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = models.Model(inputs=inputs, outputs=outputs, name="ResNet50V2_Eye_Disease")
    return model, base_model

def compile_phase1(model, learning_rate=1e-3):
    """Fase 1: Entrenamiento de la cabeza clasificadora"""
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

def compile_phase2(model, base_model, unfreeze_layers=40, learning_rate=1e-4):
    """
    Fase 2: Fine-tuning descongelando las ultimas capas de la base convolucional
    """
    base_model.trainable = True
    for layer in base_model.layers[:-unfreeze_layers]:
        layer.trainable = False
        
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model
