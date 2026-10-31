# 👁️ RetinaScan AI: Detección de Enfermedades Oculares con Deep Learning

Sistema integral de visión por computadora y aprendizaje profundo para la clasificación automatizada de patologías oculares a partir de imágenes de fondo de ojo (retinografías), desplegado como una aplicación web interactiva desarrollada en Flask.

---

## 📌 1. Visión General del Proyecto

El objetivo es asistir en el tamizaje oftalmológico temprano detectando con alta precisión cuatro categorías clínicas críticas:
1. **Catarata (Cataract)**
2. **Retinopatía Diabética (Diabetic Retinopathy)**
3. **Glaucoma**
4. **Retina Normal / Sana (Normal)**

---

## 📊 2. Dataset y División Estratificada

El proyecto utiliza el conjunto de datos público de imágenes retinianas de Kaggle (4,217 imágenes en total), normalizadas a resolución **224 × 224 píxeles**:

| Clase Patológica | Entrenamiento (70%) | Validación (15%) | Prueba (15%) | Total Imágenes |
| :--- | :---: | :---: | :---: | :---: |
| **Cataract** | 726 | 155 | 157 | 1,038 |
| **Diabetic Retinopathy** | 768 | 164 | 166 | 1,098 |
| **Glaucoma** | 704 | 151 | 152 | 1,007 |
| **Normal** | 751 | 161 | 162 | 1,074 |
| **Total** | **2,949** | **631** | **637** | **4,217** |

*División generada de forma reproducible con semilla `42` mediante [`split_dataset.py`](split_dataset.py).*

---

## 🧠 3. Comparativa de Modelos (Benchmark de Arquitecturas)

Se evaluaron tres arquitecturas convolucionales bajo un esquema de **Transfer Learning** en 2 fases (Extracción de características + Fine-tuning):

| Modelo | Arquitectura Clave | Precisión Test | Diagnóstico de Desempeño |
| :--- | :--- | :---: | :--- |
| ⭐ **ResNet50V2 (Ganador)** | **Conexiones Residuales (Skip Connections)** | **89.01%** | **Óptimo:** Captó los micro-detalles vasculares y daño en fibras nerviosas sin degradación de gradiente. Curvas de Loss/Accuracy equilibradas. |
| ⚡ **MobileNetV2** | Convolución Separable en Profundidad | 81.20% | **Ligero:** Diseñado para baja potencia computacional/móvil, pero sacrificó agudeza diagnóstica en lesiones sutiles. |
| ❌ **EfficientNetB0** | Escalado Compuesto (Compound Scaling) | 25.43% | **Descartado:** Conflicto de normalización en el pipeline que impidió la convergencia en este conjunto de datos. |

### Arquitectura de la Cabeza Clasificadora (ResNet50V2):
```
Input(224, 224, 3) 
  ↳ ResNet50V2 Base (ImageNet)
  ↳ GlobalAveragePooling2D
  ↳ BatchNormalization
  ↳ Dense(512, activation='relu') + Dropout(0.5)
  ↳ Dense(256, activation='relu') + Dropout(0.3)
  ↳ Dense(4, activation='softmax')
```

---

## 📁 4. Estructura del Repositorio

```text
METODOLOGIA/
│
├── dataset/                     # Imágenes originales organizadas por clase
├── data_split/                  # Dataset particionado (train, val, test)
│   ├── train/
│   ├── val/
│   └── test/
├── models/
│   ├── model_builder.py         # Definición modular de ResNet50V2 y 2 fases
│   └── best_resnet50.h5         # Pesos entrenados del modelo
├── templates/
│   └── index.html               # Panel clínico de IA interactivo (HTML5/CSS3/JS)
├── split_dataset.py             # Script de división estratificada del dataset
├── train_resnet.py              # Script completo de entrenamiento y métricas
├── utils.py                     # Motor de inferencia y base de conocimiento médica
├── app.py                       # Servidor web Flask
├── requirements.txt             # Librerías necesarias
└── README.md                    # Documentación del proyecto
```

---

## 🚀 5. Instalación y Ejecución Local

### 1. Clonar el repositorio y acceder a la carpeta:
```bash
git clone https://github.com/tu-usuario/ocular-disease-detection.git
cd ocular-disease-detection
```

### 2. Instalar dependencias:
```bash
pip install -r requirements.txt
```

### 3. (Opcional) Separar dataset y entrenar el modelo:
```bash
# Partición del dataset
python split_dataset.py

# Entrenamiento con ResNet50V2
python train_resnet.py --p1 15 --p2 30
```

### 4. Iniciar la aplicación web Flask:
```bash
python app.py
```
Abre en tu navegador: **`http://localhost:5000`**

---

## 🌐 6. Despliegue en Hugging Face Spaces

La aplicación está preparada para su despliegue continuo en **Hugging Face Spaces** (usando el SDK de Docker o Streamlit/Gradio/Flask):
1. Crear un nuevo Space en [Hugging Face Spaces](https://huggingface.co/spaces).
2. Seleccionar entorno Docker o Python.
3. Subir `app.py`, `utils.py`, `models/`, `templates/` y `requirements.txt`.
4. El modelo se ejecuta de forma completamente funcional en la nube.
