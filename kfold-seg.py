from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import KFold
import datetime
import shutil
from ultralytics import YOLO
import os

def realizar_kfold_validacion_segmentacion(dataset_path, yaml_file, weights_path, ksplit=5, batch=16, epochs=100, project="kfold_validacion_segmentacion"):
    """
    Realiza la validación cruzada K-Fold para un modelo de segmentación YOLOv8.

    Args:
        dataset_path (str): Ruta al directorio principal del dataset.
        yaml_file (str): Ruta al archivo YAML de configuración del dataset.
        weights_path (str): Ruta a los pesos pre-entrenados del modelo (opcional, puede ser un modelo .pt o 'yolov8n-seg.pt', por ejemplo).
        ksplit (int): Número de folds (K).
        batch (int): Tamaño del lote.
        epochs (int): Número de épocas de entrenamiento.
        project (str): Nombre del proyecto para guardar los resultados.

    Returns:
        None (Guarda los resultados de cada fold y los archivos de configuración).
    """

    # 1. Carga del archivo YAML
    yaml_file = Path(yaml_file)
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    # 2. Carga de rutas de imágenes y etiquetas
    dataset_path = Path(dataset_path)
    img_path = Path(data["train"]).parent  # Usamos el directorio padre de 'train'
    # La ruta de etiquetas ahora probablemente esté dentro de un subdirectorio 'labels' en el mismo nivel que 'images'
    lbl_path = img_path / "labels"
    img_path = img_path / "images" # La ruta a la carpeta de imagenes sera la definida con "images"

    images = sorted(list(img_path.glob("*.jpg")))
    labels = sorted(list(lbl_path.glob("*.txt")))


    print(f"Ruta de las imágenes: {img_path}")
    print(f"Ruta de las etiquetas: {lbl_path}")
    print(f"Número de imágenes encontradas: {len(images)}")
    print(f"Número de etiquetas encontradas: {len(labels)}")

    if not images:
        raise ValueError("No se encontraron imágenes en la ruta especificada.")
    if not labels:
        raise ValueError("No se encontraron etiquetas en la ruta especificada.")
    if len(images) != len(labels):
        raise ValueError("El número de imágenes y etiquetas no coincide.")


    # 3. Creación del DataFrame (igual que antes)
    labels_df = pd.DataFrame(index=[img.stem for img in images])
    kf = KFold(n_splits=ksplit, shuffle=True, random_state=42)
    for i, (train, val) in enumerate(kf.split(labels_df)):
        labels_df[f"split_{i+1}"] = "NaN"
        labels_df[f"split_{i+1}"].loc[labels_df.iloc[train].index] = "train"
        labels_df[f"split_{i+1}"].loc[labels_df.iloc[val].index] = "val"

    folds_df = labels_df.copy() # Copia para evitar modificaciones futuras no deseadas

    # 4. Creación de directorios y copia (igual que antes, pero ahora dentro de 'segmentation')
    save_path = dataset_path / f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val_segmentation"
    save_path.mkdir(parents=True, exist_ok=True)
    ds_yamls = {}
    for k in range(1, ksplit + 1):
        (save_path / f"split_{k}/train/images").mkdir(parents=True, exist_ok=True)
        (save_path / f"split_{k}/train/labels").mkdir(parents=True, exist_ok=True)
        (save_path / f"split_{k}/val/images").mkdir(parents=True, exist_ok=True)
        (save_path / f"split_{k}/val/labels").mkdir(parents=True, exist_ok=True)

        # Copia de imágenes y etiquetas
        train_images = labels_df[labels_df[f"split_{k}"] == "train"].index
        val_images = labels_df[labels_df[f"split_{k}"] == "val"].index
        for img_name in train_images:
            shutil.copy(img_path / f"{img_name}.jpg", save_path / f"split_{k}/train/images/{img_name}.jpg")
            shutil.copy(lbl_path / f"{img_name}.txt", save_path / f"split_{k}/train/labels/{img_name}.txt")
        for img_name in val_images:
            shutil.copy(img_path / f"{img_name}.jpg", save_path / f"split_{k}/val/images/{img_name}.jpg")
            shutil.copy(lbl_path / f"{img_name}.txt", save_path / f"split_{k}/val/labels/{img_name}.txt")

        # Creación de YAMLs para cada fold (rutas RELATIVAS)
        ds_yamls[k] = save_path / f"split_{k}/split_{k}_dataset.yaml"
        with open(ds_yamls[k], "w") as f:
            yaml.dump({
                "path": str(save_path / f"split_{k}"), # Ruta base ABSOLUTA del fold
                "train": "train/images",  # Rutas RELATIVAS a 'path'
                "val": "val/images",      # Rutas RELATIVAS a 'path'
                "names": data["names"]
            }, f)
        print(f"Archivo YAML creado: {ds_yamls[k]}")


    # 5. Entrenamiento del modelo (especificando task='segment')
    results = {}
    for k in range(1, ksplit + 1):
        dataset_yaml = ds_yamls[k]
        #Crear un nuevo yaml con la ruta absoluta.
        absolute_dataset_yaml = save_path / f"split_{k}/split_{k}_dataset_absolute.yaml"
        with open(dataset_yaml, 'r') as file:
            data_yaml = yaml.safe_load(file)
            data_yaml['path'] = os.path.abspath(data_yaml['path']) # Obtener ruta absoluta
        
        with open(absolute_dataset_yaml,'w') as file:
            yaml.dump(data_yaml, file)

        print(f"Entrenando en fold {k} con archivo YAML: {absolute_dataset_yaml}")

        # Inicializar el modelo para SEGMENTACIÓN.  Importante usar el sufijo '-seg'.
        if weights_path.endswith(".pt"):
            model = YOLO(weights_path, task="segment")
        else: #Si no es una ruta, que busque el modelo, por ejemplo, 'yolov8n-seg.pt'
             model = YOLO(weights_path, task="segment")

        model.train(data=str(absolute_dataset_yaml), epochs=epochs, batch=batch, project=project, name=f"train_seg{k}")
        results[k] = model.metrics  # Guardar métricas (opcional, para análisis posterior)

    print("Validación cruzada k-fold para segmentación completada.")

dataset_path = "datasets"  # Ruta a tu dataset
yaml_file = "label.yaml"  # Tu archivo YAML
weights_path = "runs/segment/train15/weights/best.pt"  # O la ruta a tus pesos pre-entrenados, o un modelo base como 'yolov8n-seg.pt'
realizar_kfold_validacion_segmentacion(dataset_path, yaml_file, weights_path, ksplit=5, epochs=100) # Puedes ajustar ksplit y epochs