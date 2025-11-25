!pip install -U ultralytics

from pathlib import Path
import yaml
import pandas as pd
from sklearn.model_selection import KFold
import datetime
import shutil
from ultralytics import YOLO
import os
import random # Para barajar las imágenes

def realizar_kfold_validacion_clasificacion(dataset_source_path, yaml_file, weights_path, ksplit=5, batch=16, epochs=100, project="kfold_validacion_clasificacion"):

    # 1. Carga del archivo YAML original para obtener 'names'
    yaml_file_path = Path(yaml_file)
    with open(yaml_file_path, "r") as f:
        data_original = yaml.safe_load(f)
        class_names = data_original['names']
        print(f"Nombres de clases cargados: {class_names}")

    # 2. Carga de rutas de imágenes y sus clases
    source_path = Path(dataset_source_path)
    image_paths = []
    # Asumimos que dataset_source_path contiene subdirectorios con los nombres de las clases
    for class_dir in source_path.iterdir():
        if class_dir.is_dir():
            class_name = class_dir.name
            if class_name in class_names.values(): # Asegurarse que el directorio es una clase válida
                for img_file in class_dir.glob('*.*'): # Aceptar varios formatos de imagen
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.gif']:
                        image_paths.append((img_file, class_name))
            else:
                print(f"Advertencia: El directorio '{class_name}' no está en 'names' del YAML y será ignorado.")


    if not image_paths:
        raise ValueError(f"No se encontraron imágenes en las subcarpetas de clases dentro de: {source_path}")

    print(f"Número total de imágenes encontradas: {len(image_paths)}")

    # Barajar las imágenes antes de dividir en folds
    random.seed(42) # Para reproducibilidad
    random.shuffle(image_paths)

    # Convertir a DataFrame para facilitar el manejo con KFold (opcional pero cómodo)
    # Usaremos los índices del DataFrame para KFold
    images_df = pd.DataFrame(image_paths, columns=['path', 'class'])


    # 3. Creación de Folds con KFold
    kf = KFold(n_splits=ksplit, shuffle=False, random_state=None) # Ya barajamos antes

    # 4. Creación de directorios y copia de archivos para cada fold
    # El directorio base donde se guardarán los folds
    save_base_path = Path(f"{datetime.date.today().isoformat()}_{ksplit}-Fold_Cross-val_clasificacion")
    save_base_path.mkdir(parents=True, exist_ok=True)
    print(f"Directorio base para los folds: {save_base_path}")

    ds_yamls = {} # Diccionario para guardar las rutas a los YAML de cada fold

    for i, (train_indices, val_indices) in enumerate(kf.split(images_df)):
        k = i + 1
        fold_path = save_base_path / f"split_{k}"
        train_path = fold_path / "train"
        val_path = fold_path / "val"

        print(f"\nProcesando Fold {k}/{ksplit}...")
        print(f"  Directorio del fold: {fold_path}")

        # Crear directorios para train/val dentro del fold actual
        train_path.mkdir(parents=True, exist_ok=True)
        val_path.mkdir(parents=True, exist_ok=True)

        # Copiar imágenes de ENTRENAMIENTO
        print(f"  Copiando {len(train_indices)} imágenes de entrenamiento...")
        for idx in train_indices:
            original_img_path, class_name = images_df.iloc[idx]
            target_class_dir = train_path / class_name
            target_class_dir.mkdir(exist_ok=True) # Crear directorio de clase si no existe
            shutil.copy(original_img_path, target_class_dir / original_img_path.name)

        # Copiar imágenes de VALIDACIÓN
        print(f"  Copiando {len(val_indices)} imágenes de validación...")
        for idx in val_indices:
            original_img_path, class_name = images_df.iloc[idx]
            target_class_dir = val_path / class_name
            target_class_dir.mkdir(exist_ok=True) # Crear directorio de clase si no existe
            shutil.copy(original_img_path, target_class_dir / original_img_path.name)

        # 5. Creación de YAML para el fold actual
        yaml_fold_path = fold_path / f"split_{k}_dataset.yaml"
        ds_yamls[k] = yaml_fold_path

        with open(yaml_fold_path, "w") as f:
            yaml.dump({
                "train": str(train_path),  # Ruta absoluta al directorio de entrenamiento
                "val": str(val_path),    # Ruta absoluta al directorio de validación
                "names": class_names
            }, f, default_flow_style=False)
        print(f"  Archivo YAML creado: {yaml_fold_path}")

    # 6. Entrenamiento del modelo para cada fold
    results = {}
    for k in range(1, ksplit + 1):
        fold_path = save_base_path / f"split_{k}"
        # dataset_yaml = ds_yamls[k] # Ya no lo usaremos directamente aquí

        print(f"\n--- Entrenando Fold {k}/{ksplit} ---")
        print(f"Usando directorio de datos: {fold_path}") # Ahora mostramos la ruta del directorio

        # Inicializar el modelo para CLASIFICACIÓN.
        model = YOLO(weights_path, task='classify')

        model.train(
            data=str(fold_path), # Pasamos la ruta al directorio del fold
            epochs=epochs,
            batch=batch,
            project=project, # Guarda en project/train_cls{k}
            name=f"train_cls{k}",
            # Puedes añadir otros hiperparámetros aquí: patience, optimizer, lr0, lrf, etc.
            # cache=True # Puede acelerar la carga de datos en épocas subsiguientes
        )
        results[k] = model.metrics # Guardar métricas (opcional, para análisis posterior)
        print(f"--- Entrenamiento Fold {k} completado ---")

    print("\nValidación cruzada k-fold para clasificación completada.")
    print(f"Resultados guardados en el directorio del proyecto: '{project}'")

# Ruta al directorio que contiene las carpetas de clase (tu 'train' original)
dataset_source_path = "datasets/train"

# Ruta al archivo YAML original (para leer 'names')
yaml_file = "label1.yaml"

# Modelo base de clasificación o ruta a tus pesos (asegúrate que sea -cls.pt)
weights_path = "runs/classify/train10/weights/best.pt" # O "yolo11l-cls.pt" o una ruta a pesos tuyos

# Nombre para la carpeta donde se guardarán los resultados de los 5 entrenamientos
project_name = "resultados_kfold_class" # ¡Elige un nombre descriptivo!

realizar_kfold_validacion_clasificacion(
    dataset_source_path=dataset_source_path,
    yaml_file=yaml_file,
    weights_path=weights_path,
    ksplit=5,        # O el número de folds que quieras
    epochs=100,       # Ajusta las épocas
    batch=16,        # Ajusta el batch según tu GPU
    project=project_name # Nombre de la carpeta de resultados
)