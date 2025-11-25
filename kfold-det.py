from pathlib import Path
import yaml
import pandas as pd
from collections import Counter
from sklearn.model_selection import KFold
import datetime
import shutil
from ultralytics import YOLO
import os

def realizar_kfold_validacion(dataset_path, yaml_file, weights_path, ksplit=5, batch=16, epochs=100, project="kfold_validacion"):
    # 1. Carga del archivo YAML al inicio
    yaml_file = Path(yaml_file)
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    # 2. Carga de rutas de imágenes y etiquetas
    dataset_path = Path(dataset_path)
    img_path = Path(data["train"])
    lbl_path = Path(data["train"]).parent / "labels"
    images = sorted(list(img_path.glob("*.jpg")))
    labels = sorted(list(lbl_path.glob("*.txt")))

    print(f"Ruta de las imágenes: {img_path}")
    print(f"Ruta de las etiquetas: {lbl_path}")
    print(f"Número de imágenes encontradas: {len(images)}")
    print(f"Número de etiquetas encontradas: {len(labels)}")

    # 3. Creación del DataFrame con la distribución de los folds
    labels_df = pd.DataFrame(index=[img.stem for img in images])
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=42)
    for i, (train, val) in enumerate(kf.split(labels_df)):
        labels_df[f"split_{i+1}"] = "NaN"
        labels_df[f"split_{i+1}"].loc[labels_df.iloc[train].index] = "train"
        labels_df[f"split_{i+1}"].loc[labels_df.iloc[val].index] = "val"

    folds_df = labels_df.copy()

    # 4. Creación de los directorios para los folds y copia de imágenes/etiquetas
    save_path = dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val"
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = {}
    for k in range(1, ksplit + 1):
        (save_path / f"split_{k}/train/images").mkdir(parents=True, exist_ok=True)
        (save_path / f"split_{k}/train/labels").mkdir(parents=True, exist_ok=True)
        (save_path / f"split_{k}/val/images").mkdir(parents=True, exist_ok=True)
        (save_path / f"split_{k}/val/labels").mkdir(parents=True, exist_ok=True)

        # Copia de imágenes y etiquetas a los folds
        train_images = labels_df[labels_df[f"split_{k}"] == "train"].index
        val_images = labels_df[labels_df[f"split_{k}"] == "val"].index
        for img_name in train_images:
            shutil.copy(img_path / f"{img_name}.jpg", save_path / f"split_{k}/train/images/{img_name}.jpg")
            shutil.copy(lbl_path / f"{img_name}.txt", save_path / f"split_{k}/train/labels/{img_name}.txt")
        for img_name in val_images:
            shutil.copy(img_path / f"{img_name}.jpg", save_path / f"split_{k}/val/images/{img_name}.jpg")
            shutil.copy(lbl_path / f"{img_name}.txt", save_path / f"split_{k}/val/labels/{img_name}.txt")

        # Creación del archivo YAML para cada fold
        ds_yamls[k] = save_path / f"split_{k}/split_{k}_dataset.yaml"
        with open(ds_yamls[k], "w") as f:
            yaml.dump({
                "path": str(save_path / f"split_{k}"),
                "train": "train/images",
                "val": "val/images",
                "names": data["names"]
            }, f)
        print(f"Archivo YAML creado: {ds_yamls[k]}")

    # 5. Entrenamiento del modelo para cada fold
    results = {}
    for k in range(1, ksplit + 1):
        dataset_yaml = ds_yamls[k]
        # Crear un nuevo yaml con la ruta absoluta.
        absolute_dataset_yaml = save_path / f"split_{k}/split_{k}_dataset_absolute.yaml"
        with open(dataset_yaml, 'r') as file:
            data_yaml = yaml.safe_load(file)
        with open(absolute_dataset_yaml,'w') as file:
            yaml.dump(data_yaml, file)

        print(f"Entrenando en fold {k} con archivo YAML: {absolute_dataset_yaml}")

        model = YOLO(weights_path, task="detect")
        model.train(data=str(absolute_dataset_yaml), epochs=epochs, batch=batch, project=project, name=f"train{k}")
        results[k] = model.metrics

    print("Validación cruzada k-fold completada.")

# Ejemplo de uso
dataset_path = "/home/berto/Documents/CIRCAROCK/detection/datasets"
yaml_file = "label1.yaml"
weights_path = "runs/detect/train24/weights/best.pt"
realizar_kfold_validacion(dataset_path, yaml_file, weights_path)