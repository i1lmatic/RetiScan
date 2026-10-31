import os
import io
import json
from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
from PIL import Image

import utils

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data_split")
STATIC_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")

app = Flask(__name__, static_folder=STATIC_DIR, template_folder=TEMPLATES_DIR)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Max 16MB

# Precarga del modelo ResNet50V2
print("[STARTUP] Inicializando motor de inferencia ResNet50V2...")
utils.load_my_model()
print("[STARTUP] Motor listo para recibir peticiones.")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint para clasificacion de imagenes retinianas.
    Acepta subida de archivo multipart/form-data o llamada JSON con sample_class.
    """
    try:
        sample_class = request.form.get('sample_class')
        file = request.files.get('file')

        if sample_class and sample_class in utils.CLASSES:
            # Usar imagen de muestra del dataset de test
            sample_folder = os.path.join(DATA_DIR, "test", sample_class)
            if os.path.exists(sample_folder):
                files = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                if files:
                    img_path = os.path.join(sample_folder, files[0])
                    image = Image.open(img_path)
                else:
                    return jsonify({"error": "No hay imagenes de muestra en esa categoria"}), 404
            else:
                return jsonify({"error": "Carpeta de muestra no encontrada"}), 404
        elif file and file.filename != '':
            image = Image.open(file.stream)
        else:
            return jsonify({"error": "No se proporciono ninguna imagen valida"}), 400

        # Ejecutar inferencia a traves de utils
        result = utils.predict_retinal_image(image)
        return jsonify({"success": True, "result": result})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/sample/<category>', methods=['GET'])
def get_sample_image(category):
    """Devuelve el archivo de imagen de muestra del conjunto de prueba"""
    if category not in utils.CLASSES:
        return "Categoria invalida", 400
    
    sample_folder = os.path.join(DATA_DIR, "test", category)
    if not os.path.exists(sample_folder):
        return "Carpeta no encontrada", 404
        
    files = [f for f in os.listdir(sample_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if not files:
        return "No hay imagenes disponibles", 404
        
    return send_from_directory(sample_folder, files[0])

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Estadisticas del dataset y rendimiento de los modelos comparados"""
    stats = {
        "dataset": {
            "total_images": 4217,
            "train": 2949,
            "val": 631,
            "test": 637,
            "classes": ["Catarata", "Retinopatía Diabética", "Glaucoma", "Normal"]
        },
        "models_benchmark": [
            {
                "name": "ResNet50V2 (Ganador)",
                "type": "Residual Deep Learning",
                "accuracy": 89.01,
                "status": "Modelo en Producción",
                "highlight": True,
                "badge": "Especialista Profundo (Winner)"
            },
            {
                "name": "MobileNetV2",
                "type": "Depthwise Separable Conv",
                "accuracy": 81.20,
                "status": "Evaluado",
                "highlight": False,
                "badge": "El Velocista (Mobile)"
            },
            {
                "name": "EfficientNetB0",
                "type": "Compound Scaling",
                "accuracy": 25.43,
                "status": "Descartado",
                "highlight": False,
                "badge": "Baja Convergencia"
            }
        ]
    }
    return jsonify(stats)

if __name__ == '__main__':
    # Ejecucion local
    app.run(host='0.0.0.0', port=5000, debug=False)
