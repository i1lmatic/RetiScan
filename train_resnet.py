import os
import argparse
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Generar graficos sin necesidad de GUI interactiva
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix

from models.model_builder import build_resnet50v2_model, compile_phase1, compile_phase2

# Configuracion de directorios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_split")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 4
CLASSES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

def get_data_generators(data_dir=DATA_DIR, img_size=IMG_SIZE, batch_size=BATCH_SIZE):
    """
    Configura los generadores de datos con el esquema exacto de Data Augmentation
    del proyecto.
    """
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        width_shift_range=0.1,
        height_shift_range=0.1
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        os.path.join(data_dir, "train"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=True
    )

    val_gen = val_datagen.flow_from_directory(
        os.path.join(data_dir, "val"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    test_gen = test_datagen.flow_from_directory(
        os.path.join(data_dir, "test"),
        target_size=img_size,
        batch_size=batch_size,
        class_mode="categorical",
        shuffle=False
    )

    return train_gen, val_gen, test_gen

def plot_and_save_curves(h_p1, h_p2, save_path):
    """Grafica y guarda las curvas de entrenamiento bifasicas (fase 1 + fase 2)"""
    acc_all = h_p1.history.get("accuracy", []) + h_p2.history.get("accuracy", [])
    val_all = h_p1.history.get("val_accuracy", []) + h_p2.history.get("val_accuracy", [])
    loss_all = h_p1.history.get("loss", []) + h_p2.history.get("loss", [])
    vloss_all = h_p1.history.get("val_loss", []) + h_p2.history.get("val_loss", [])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Precision
    ax1.plot(acc_all, label="Train Accuracy", color="royalblue", lw=2)
    ax1.plot(val_all, label="Val Accuracy", color="darkblue", lw=2)
    split_epoch = len(h_p1.history.get("accuracy", [])) - 1
    if split_epoch >= 0:
        ax1.axvline(x=split_epoch, color="gray", linestyle="--", label="Fine-tune start")
    ax1.set_title("ResNet50V2 - Accuracy", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Perdida
    ax2.plot(loss_all, label="Train Loss", color="indianred", lw=2)
    ax2.plot(vloss_all, label="Val Loss", color="darkred", lw=2)
    if split_epoch >= 0:
        ax2.axvline(x=split_epoch, color="gray", linestyle="--", label="Fine-tune start")
    ax2.set_title("ResNet50V2 - Loss", fontsize=13, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Curvas de entrenamiento guardadas en: {save_path}")

def plot_and_save_confusion_matrix(y_true, y_pred, classes, save_path):
    """Grafica y guarda la matriz de confusion con mapa de calor"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar=True
    )
    plt.title("ResNet50V2 - Confusion Matrix (Test Set)", fontsize=13, fontweight="bold")
    plt.ylabel("Clase Real (Actual)", fontsize=11)
    plt.xlabel("Clase Predicha (Predicted)", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[OK] Matriz de confusion guardada en: {save_path}")

def train_and_evaluate(epochs_p1=15, epochs_p2=30):
    print("=" * 60)
    print("  INICIANDO ENTRENAMIENTO DE RESNET50V2 (MODELO GANADOR)")
    print("=" * 60)

    train_gen, val_gen, test_gen = get_data_generators()

    model, base_model = build_resnet50v2_model(num_classes=NUM_CLASSES, input_shape=(224, 224, 3))
    model = compile_phase1(model, learning_rate=1e-3)

    best_model_path = os.path.join(MODELS_DIR, "best_resnet50.h5")
    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(best_model_path, monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=3, min_lr=1e-7, verbose=1)
    ]

    print("\n" + "=" * 50)
    print(f"  ResNet50V2 - Fase 1: Entrenamiento de Cabeza ({epochs_p1} epocas)")
    print("=" * 50)
    h_p1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_p1,
        callbacks=callbacks
    )

    print("\n" + "=" * 50)
    print(f"  ResNet50V2 - Fase 2: Fine-Tuning ultimas 40 capas ({epochs_p2} epocas)")
    print("=" * 50)
    model = compile_phase2(model, base_model, unfreeze_layers=40, learning_rate=1e-4)
    h_p2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs_p2,
        callbacks=callbacks
    )

    print("\n" + "=" * 50)
    print("  EVALUACION EN CONJUNTO DE PRUEBA (TEST SET)")
    print("=" * 50)
    test_loss, test_acc = model.evaluate(test_gen)
    print(f"\n[METRICA] ResNet50V2 Test Accuracy: {test_acc * 100:.2f}% (Loss: {test_loss:.4f})")

    y_pred_probs = model.predict(test_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes

    print("\n--- Reporte de Clasificacion Detallado ---")
    report = classification_report(y_true, y_pred, target_names=CLASSES)
    print(report)

    # Guardar graficos tanto en static/ como en models/
    curves_path = os.path.join(STATIC_DIR, "resnet50_training_curves.png")
    cm_path = os.path.join(STATIC_DIR, "resnet50_confusion_matrix.png")
    plot_and_save_curves(h_p1, h_p2, curves_path)
    plot_and_save_confusion_matrix(y_true, y_pred, CLASSES, cm_path)

    final_model_keras = os.path.join(MODELS_DIR, "resnet50_eye_disease.keras")
    model.save(final_model_keras)
    print(f"[OK] Modelo final guardado en formato Keras 3: {final_model_keras}")
    print("\n[PROCESO COMPLETADO EXITOSAMENTE]")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Entrenamiento de ResNet50V2")
    parser.add_argument("--p1", type=int, default=15, help="Epocas Fase 1")
    parser.add_argument("--p2", type=int, default=30, help="Epocas Fase 2")
    args = parser.parse_args()

    train_and_evaluate(epochs_p1=args.p1, epochs_p2=args.p2)
