import os
import shutil
import random

# Directorios de origen y destino
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "data_split")

# Clases y proporciones de partición
CLASSES = ["cataract", "diabetic_retinopathy", "glaucoma", "normal"]
SPLITS = {"train": 0.70, "val": 0.15, "test": 0.15}

# Semilla para reproducibilidad exacta
random.seed(42)

def split_dataset():
    print(f"Iniciando división del dataset desde: {SOURCE_DIR}")
    print(f"Destino: {OUTPUT_DIR}\n")

    for cls in CLASSES:
        src = os.path.join(SOURCE_DIR, cls)
        if not os.path.exists(src):
            print(f"[!] Carpeta no encontrada: {src}")
            continue

        images = [
            f for f in os.listdir(src)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ]
        random.shuffle(images)

        n = len(images)
        n_train = int(n * SPLITS["train"])
        n_val = int(n * SPLITS["val"])

        split_map = {
            "train": images[:n_train],
            "val": images[n_train:n_train + n_val],
            "test": images[n_train + n_val:]
        }

        for split, imgs in split_map.items():
            dest = os.path.join(OUTPUT_DIR, split, cls)
            os.makedirs(dest, exist_ok=True)
            for img in imgs:
                shutil.copy(os.path.join(src, img), os.path.join(dest, img))

        print(
            f"{cls:30s} -> train:{len(split_map['train'])} "
            f"val:{len(split_map['val'])} test:{len(split_map['test'])}"
        )

    print("\n[OK] Dataset split complete!")

if __name__ == "__main__":
    split_dataset()
