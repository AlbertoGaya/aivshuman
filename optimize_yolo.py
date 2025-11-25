import optuna
from ultralytics import YOLO
import torch # Recomendable para verificar la disponibilidad de la GPU

# --- PASO 1: Definir la Función Objetivo ---
# Esta función es el corazón del proceso. Optuna la llamará en cada "trial" (intento)
# para entrenar el modelo con una nueva combinación de hiperparámetros.

def objective(trial):
    """
    Entrena un modelo YOLO con hiperparámetros sugeridos por Optuna
    y devuelve la métrica de rendimiento que queremos maximizar (mAP50-95).
    """
    try:
        # Verifica si hay GPU disponible para el entrenamiento
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Iniciando trial #{trial.number} en el dispositivo: {device}")

        # Carga el modelo base en cada trial para empezar desde cero
        model = YOLO('yolov8m.pt') # Puedes cambiarlo a 'yolov8s.pt' o el que uses

        # --- PASO 2: Definir el Espacio de Búsqueda de Hiperparámetros ---
        # Aquí le decimos a Optuna en qué rangos debe buscar los valores.
        
        # Tasa de aprendizaje inicial. Escala logarítmica es mejor para lr.
        lr0 = trial.suggest_float('lr0', 1e-5, 1e-1, log=True)
        
        # Tasa de aprendizaje final (lrf = lr0 * lrf_ratio)
        lrf_ratio = trial.suggest_float('lrf_ratio', 0.01, 1.0)
        lrf = lr0 * lrf_ratio
        
        # Momento para optimizadores como SGD o Adam
        momentum = trial.suggest_float('momentum', 0.85, 0.95)
        
        # Aumentos de datos
        mixup = trial.suggest_float('mixup', 0.0, 0.5)
        translate = trial.suggest_float('translate', 0.0, 0.3)
        scale = trial.suggest_float('scale', 0.4, 0.8)
        erasing = trial.suggest_float('erasing', 0.2, 0.7)
        
        # Regularización (weight decay)
        weight_decay = trial.suggest_float('weight_decay', 0.0, 0.001)

        # --- PASO 3: Entrenar el Modelo ---
        # Se ejecuta el entrenamiento con la combinación de hiperparámetros del trial actual.
        
        results = model.train(
            data='label1.yaml',
            # **RECOMENDACIÓN**: Para la búsqueda, reduce las épocas (ej. 100)
            # para que cada trial sea más rápido. Luego, entrena el modelo final
            # con las 600 épocas completas.
            epochs=100,
            patience=25, # Reducir patience junto con epochs
            batch=16,
            imgsz=640,
            single_cls=True,
            optimizer='AdamW',
            device=device,
            
            # --- Hiperparámetros a optimizar ---
            lr0=lr0,
            lrf=lrf,
            momentum=momentum,
            weight_decay=weight_decay,
            mixup=mixup,
            translate=translate,
            scale=scale,
            erasing=erasing,

            # Desactiva logs detallados de Ultralytics para mantener la salida limpia
            verbose=False 
        )

        # --- Devolver la Métrica a Maximizar ---
        # El valor de retorno es lo que Optuna intentará maximizar.
        # `results.box.map` es el mAP50-95, la métrica estándar.
        return results.box.map

    except Exception as e:
        print(f"Trial #{trial.number} falló con el error: {e}")
        # Si un trial falla, devuelve un valor muy bajo para que Optuna lo descarte.
        return 0.0

# --- PASO 4: Crear y Ejecutar el Estudio de Optimización ---

if __name__ == '__main__':
    # Creamos el estudio. 'direction="maximize"' porque queremos el mAP más alto.
    # El sampler por defecto es TPESampler, así que no es necesario especificarlo.
    study = optuna.create_study(direction='maximize')

    # Iniciamos la optimización. Optuna llamará a la función 'objective' 50 veces.
    # Puedes cambiar 'n_trials' según el tiempo que tengas disponible.
    try:
        study.optimize(objective, n_trials=50)
    except KeyboardInterrupt:
        print("Optimización interrumpida por el usuario.")

    # --- PASO 5: Mostrar los Resultados ---
    
    print("\n" + "="*50)
    print("🚀 OPTIMIZACIÓN COMPLETADA 🚀")
    print("="*50)

    print(f"Número de trials finalizados: {len(study.trials)}")

    print("\n🏆 MEJOR TRIAL ENCONTRADO 🏆")
    best_trial = study.best_trial
    print(f"  📈 Valor (mAP50-95): {best_trial.value:.4f}")

    print("\n📋 MEJORES HIPERPARÁMETROS:")
    for key, value in best_trial.params.items():
        print(f"  - {key}: {value:.6f}")
    
    print("\n" + "="*50)
    print("\nAhora puedes usar estos hiperparámetros para realizar el entrenamiento final")
    print("con el número completo de épocas (600) para obtener el mejor modelo.")