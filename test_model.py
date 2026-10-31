import os
import utils

def test_inference_all_classes():
    print("=" * 65)
    print("  VERIFICACIÓN DEL MODELO ENTRENADO: best_resnet50.h5")
    print("=" * 65)

    model = utils.load_my_model()
    print(f"[OK] Modelo cargado en memoria exitosamente: {type(model)}\n")

    for cls in utils.CLASSES:
        folder = os.path.join("data_split", "test", cls)
        if not os.path.exists(folder):
            continue
        imgs = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if not imgs:
            continue
            
        test_img_path = os.path.join(folder, imgs[0])
        res = utils.predict_retinal_image(test_img_path)

        match_str = "CORRECTO [OK]" if res["class_key"] == cls else "DIFERENTE"
        print(f"Clase real:   {cls.upper()}")
        print(f"Predicción:   {res['name_es']} ({res['class_key']}) - {match_str}")
        print(f"Confianza:    {res['confidence']}%")
        print(f"Probabilidades: {res['probabilities']}")
        print("-" * 65)

if __name__ == "__main__":
    test_inference_all_classes()
