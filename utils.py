import os
import numpy as np
from PIL import Image
import tensorflow as tf

from models.model_builder import build_resnet50v2_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASSES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]

DISEASE_INFO = {
    "cataract": {
        "name_es": "Catarata",
        "name_en": "Cataract",
        "badge_color": "#E67E22",
        "severity": "Moderada a Severa",
        "description": "Opacificación progresiva del cristalino natural del ojo. Impide el paso adecuado de la luz hacia la retina, provocando visión borrosa, pérdida de contraste y deslumbramiento con luces intensas.",
        "symptoms": [
            "Visión nublada, borrosa o tenue",
            "Dificultad creciente con la visión nocturna",
            "Sensibilidad a la luz y al resplandor (halos en faros de noche)",
            "Los colores se ven desvanecidos o amarillentos"
        ],
        "recommendations": [
            "Programar evaluación oftalmológica completa con lámpara de hendidura.",
            "Considerar cirugía de extracción de catarata con implante de lente intraocular (LIO) si afecta la calidad de vida.",
            "Uso de gafas de sol con filtro UV de alta protección para prevenir progresión.",
            "Evitar la conducción nocturna si experimenta halos o pérdida de contraste."
        ]
    },
    "diabetic_retinopathy": {
        "name_es": "Retinopatía Diabética",
        "name_en": "Diabetic Retinopathy",
        "badge_color": "#E74C3C",
        "severity": "Alta / Crítica",
        "description": "Complicación microvascular de la diabetes mellitus causada por daño crónico a los vasos sanguíneos de la retina. Puede evolucionar de no proliferativa a proliferativa con riesgo de hemorragia vítrea y desprendimiento traccional de retina.",
        "symptoms": [
            "Manchas flotantes o hilos oscuros en el campo visual (moscas volantes)",
            "Visión fluctuante y pérdida gradual de agudeza visual",
            "Áreas oscuras o vacías en el centro de la visión",
            "Dificultad para percibir colores adecuadamente"
        ],
        "recommendations": [
            "Control glucémico estricto (mantener HbA1c dentro del rango objetivo) y control de presión arterial.",
            "Examen de fondo de ojo con pupila dilatada y angiografía fluoresceínica urgente.",
            "Consulta con especialista en retina médica/quirúrgica para valorar fotocoagulación con láser o inyecciones intravítreas anti-VEGF.",
            "Monitoreo oftalmológico periódico al menos cada 6 a 12 meses."
        ]
    },
    "glaucoma": {
        "name_es": "Glaucoma",
        "name_en": "Glaucoma",
        "badge_color": "#9B59B6",
        "severity": "Muy Alta / Silenciosa",
        "description": "Neuropatía óptica progresiva caracterizada por la pérdida gradual de fibras nerviosas de la retina y excavación del nervio óptico, comúnmente ligada a presión intraocular (PIO) elevada. Es una de las principales causas de ceguera irreversible en el mundo.",
        "symptoms": [
            "Pérdida lenta e indolora de la visión periférica o lateral (visión en túnel)",
            "Puntos ciegos dispersos en la visión lateral o central",
            "En glaucoma de ángulo cerrado: dolor ocular intenso, cefalea y halos de arcoíris",
            "Náuseas o vómitos asociados a elevación súbita de PIO"
        ],
        "recommendations": [
            "Consulta oftalmológica urgente para tonometría (medición de PIO) y gonioscopia.",
            "Realizar campimetría visual computarizada y Tomografía de Coherencia Óptica (OCT) de nervio óptico.",
            "Adherencia estricta al tratamiento farmacológico hipotensor (gotas oculares diarias).",
            "Valoración de trabeculoplastia láser selectiva (SLT) o cirugía filtrante si la PIO no cede."
        ]
    },
    "normal": {
        "name_es": "Retina Normal (Sana)",
        "name_en": "Normal",
        "badge_color": "#2ECC71",
        "severity": "Saludable",
        "description": "El fondo de ojo presenta estructuras anatómicas íntegras. El disco óptico tiene márgenes nítidos, la mácula y fóvea no evidencian edema ni exudados, y la vasculatura retiniana mantiene calibre y trayectoria uniformes.",
        "symptoms": [
            "Agudeza visual normal conservada",
            "Sin defectos o escotomas en campo visual",
            "Discriminación cromática adecuada",
            "Sin fotopsias ni cuerpos flotantes anómalos"
        ],
        "recommendations": [
            "Continuar con exámenes oftalmológicos preventivos de rutina cada 1 o 2 años.",
            "Mantener hábitos de vida saludables y dieta rica en antioxidantes (luteína y zeaxantina).",
            "Usar protección ocular adecuada contra radiación ultravioleta (UV400).",
            "Descansar la vista periódicamente aplicando la regla 20-20-20 durante el uso de pantallas digitales."
        ]
    }
}

# Variable global para cachear el modelo en memoria
_LOADED_MODEL = None

def load_my_model():
    """
    Carga el modelo ResNet50V2.
    Prioridad:
    1. Archivo entrenado de pesos .keras o .h5 en la carpeta models/
    2. Si aún no existe el archivo entrenado, carga la arquitectura ResNet50V2
       con pesos ImageNet preentrenados para permitir inferencia inmediata.
    """
    global _LOADED_MODEL
    if _LOADED_MODEL is not None:
        return _LOADED_MODEL

    candidate_files = [
        os.path.join(MODELS_DIR, "best_resnet50.h5"),
        os.path.join(MODELS_DIR, "best_resnet50.keras"),
        os.path.join(MODELS_DIR, "resnet50_eye_disease.keras"),
        os.path.join(BASE_DIR, "best_resnet50.h5"),
        os.path.join(BASE_DIR, "best_model.h5")
    ]

    for model_path in candidate_files:
        if os.path.exists(model_path):
            try:
                print(f"[INFO] Cargando modelo guardado desde: {model_path}")
                _LOADED_MODEL = tf.keras.models.load_model(model_path)
                return _LOADED_MODEL
            except Exception as e:
                print(f"[WARN] Error al cargar {model_path}: {e}")

    # Fallback: Construir arquitectura ResNet50V2 lista para usar
    print("[INFO] Inicializando arquitectura ResNet50V2 con ImageNet preentrenado...")
    model, _ = build_resnet50v2_model(num_classes=len(CLASSES), input_shape=(224, 224, 3))
    _LOADED_MODEL = model
    return _LOADED_MODEL

def preprocess_image(image_input):
    """
    Preprocesa una imagen (PIL Image, bytes o ruta) para ResNet50V2 (224x224, [0, 1]).
    """
    if isinstance(image_input, str):
        image = Image.open(image_input)
    elif hasattr(image_input, 'read'):
        image = Image.open(image_input)
    else:
        image = image_input

    # Asegurar formato RGB
    image = image.convert("RGB")
    # Redimensionar exactamente a 224x224
    image_resized = image.resize((224, 224), Image.Resampling.BILINEAR)
    # Convertir a array numpy y normalizar a [0, 1]
    img_array = np.array(image_resized, dtype=np.float32) / 255.0
    # Añadir dimensión de lote (1, 224, 224, 3)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_retinal_image(image_input):
    """
    Realiza inferencia sobre una imagen retiniana devolviendo clase,
    confianza en porcentaje, desglose de probabilidades y ficha medica.
    """
    model = load_my_model()
    img_array = preprocess_image(image_input)

    # Inferencia
    preds = model.predict(img_array, verbose=0)
    
    # Manejar si el output es diccionario (ej. TFSMLayer)
    if isinstance(preds, dict):
        key = list(preds.keys())[0]
        preds = preds[key].numpy()

    preds = preds[0]  # Tomar el primer elemento del lote
    class_idx = int(np.argmax(preds))
    confidence_val = float(preds[class_idx]) * 100

    predicted_key = CLASSES[class_idx]
    info = DISEASE_INFO.get(predicted_key, {})

    # Diccionario con el desglose de todas las probabilidades
    probabilities = {
        cls: round(float(preds[i]) * 100, 2)
        for i, cls in enumerate(CLASSES)
    }

    return {
        "class_key": predicted_key,
        "name_es": info.get("name_es", predicted_key),
        "name_en": info.get("name_en", predicted_key),
        "confidence": round(confidence_val, 2),
        "badge_color": info.get("badge_color", "#3498DB"),
        "severity": info.get("severity", "Desconocida"),
        "description": info.get("description", ""),
        "symptoms": info.get("symptoms", []),
        "recommendations": info.get("recommendations", []),
        "probabilities": probabilities
    }
