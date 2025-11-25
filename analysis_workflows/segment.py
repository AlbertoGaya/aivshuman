import cv2
import pandas as pd
import math
import os
import traceback
from ultralytics import YOLO
import numpy as np

# --- Funciones de Telemetría y Extracción de Frames (sin cambios) ---

def haversine(lat1, lon1, lat2, lon2):
    R = 6371000
    lat1_rad, lon1_rad, lat2_rad, lon2_rad = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def filter_coordinates(df):
    selected_rows = []
    group_counter = 1
    group_size = 10
    current_group = []

    if df.empty:
        return pd.DataFrame()

    lat1, lon1 = float(df.iloc[0]['Lat_Rov']), float(df.iloc[0]['Lon_Rov'])
    current_group.append(df.index[0])

    for index, row in df.iloc[1:].iterrows():
        lat2, lon2 = float(row['Lat_Rov']), float(row['Lon_Rov'])
        distance = haversine(lat1, lon1, lat2, lon2)

        if distance >= 1:
            if len(current_group) >= group_size:
                for i in current_group[:group_size]:
                    selected_rows.append((i, group_counter))
                group_counter += 1
            current_group = [index]
            lat1, lon1 = lat2, lon2
        else:
            current_group.append(index)

    if len(current_group) >= group_size:
        for i in current_group[:group_size]:
            selected_rows.append((i, group_counter))

    if not selected_rows:
        print("No se seleccionaron filas basadas en el filtro de distancia.")
        return pd.DataFrame()

    indices = [index for index, group in selected_rows]
    grupos = [group for index, group in selected_rows]
    df_filtered = df.loc[indices].copy()
    df_filtered['Grupo'] = grupos
    return df_filtered

def add_frame_column(df):
    df['Frame'] = range(1, len(df) * 25 + 1, 25)
    print("Columna 'Frame' agregada con éxito al DataFrame.")
    return df

def extract_frames(video_path, frame_indices, group_number):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error al abrir el video: {video_path}")
        return []

    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_folder_group = os.path.join(f"{video_name}_frames", f"grupo_{group_number}")
    os.makedirs(output_folder_group, exist_ok=True)

    extracted_paths = []
    for frame_index in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index - 1)
        ret, frame = cap.read()
        if ret:
            file_path = os.path.join(output_folder_group, f"{frame_index}.jpg")
            cv2.imwrite(file_path, frame)
            extracted_paths.append(file_path)
    cap.release()
    return extracted_paths

# --- NUEVA SECCIÓN DE DETECCIÓN Y ESCALADO ---

def detectar_y_escalar_por_pixeles(image_path, params):
    """
    Detecta un par de láseres basado en reglas de píxeles, calcula la escala y el área total.
    """
    img_cv = cv2.imread(image_path)
    if img_cv is None:
        print(f"Error: No se pudo cargar la imagen {image_path}")
        return None, None, None

    # Crea una máscara con los píxeles que están en el rango de color.
    mask = cv2.inRange(img_cv, params['lower_bgr'], params['upper_bgr'])
    coords_yx = np.argwhere(mask == 255)
    coords_xy = [(int(coord[1]), int(coord[0])) for coord in coords_yx]

    print(f"  - Píxeles de color láser encontrados: {len(coords_xy)}")

    if len(coords_xy) < 2:
        return img_cv, None, None

    # Busca el primer par que cumpla todas las reglas
    for i in range(len(coords_xy)):
        for j in range(i + 1, len(coords_xy)):
            p1, p2 = coords_xy[i], coords_xy[j]
            
            if abs(p1[1] - p2[1]) <= params['y_tolerance']:
                dist_x = abs(p1[0] - p2[0])
                if params['min_x_dist'] <= dist_x <= params['max_x_dist']:
                    print(f"  - ✅ ¡Par válido encontrado! Distancia: {dist_x:.1f} píxeles.")
                    
                    # Calcular escala
                    cm_per_pixel = params['real_dist_cm'] / dist_x
                    
                    # Calcular área total
                    alto, ancho, _ = img_cv.shape
                    area_total_m2 = (alto * ancho * cm_per_pixel**2) / 10000
                    
                    print(f"  - Escala: {cm_per_pixel:.4f} cm/píxel. Área Total: {area_total_m2:.3f} m².")
                    return img_cv, cm_per_pixel, area_total_m2

    print("  - ❌ No se encontró un par de píxeles que cumpla los criterios.")
    return img_cv, None, None


def apply_segmentation_mask(image_array, segmentation_model):
    """
    Aplica las máscaras de segmentación y calcula el área total cubierta por ellas
    sin contar dos veces las áreas solapadas.
    """
    img_with_mask = image_array.copy()
    alto, ancho, _ = image_array.shape
    
    # 1. Crea una máscara en blanco del mismo tamaño que la imagen original.
    union_mask = np.zeros((alto, ancho), dtype=np.uint8)

    try:
        results = segmentation_model.predict(image_array, verbose=False)
        if results and results[0].masks:
            # 2. Dibuja cada máscara detectada sobre la máscara en blanco.
            #    Las áreas solapadas simplemente se sobreescribirán, sin sumarse.
            for mask in results[0].masks:
                contour = mask.xy[0].astype(np.int32)
                cv2.fillPoly(img_with_mask, [contour], (0, 0, 0)) # Mantiene la visualización
                cv2.drawContours(union_mask, [contour], -1, 255, thickness=cv2.FILLED)
    
    except Exception as e:
        print(f"Error en segmentación: {e}")
        # Si hay un error, el área de la máscara es 0 y se devuelve la imagen original.
        return image_array, 0

    # 3. Calcula el área total de la máscara de unión contando los píxeles no nulos.
    mask_pixel_area = np.count_nonzero(union_mask)
    
    return img_with_mask, mask_pixel_area


def main():
    # --- PARÁMETROS DE CONFIGURACIÓN ---
    filename = input("Introduce el nombre del archivo de telemetria (ej: telemetria.csv): ")
    video_path = input("Introduce la ruta completa del video (ej: /ruta/a/video.mp4): ")
    
    # Rutas de los modelos YOLO
    segmentation_model_path = 'segment.pt'
    classification_model_path = 'classify.pt'
    detection_model_path = 'detection.pt'

    # Parámetros para la nueva detección de láser
    laser_params = {
        'lower_bgr': np.array([175, 240, 180]),
        'upper_bgr': np.array([220, 255, 220]),
        'y_tolerance': 50,
        'min_x_dist': 50,
        'max_x_dist': 500,
        'real_dist_cm': 20.0
    }

    # --- 1. Cargar y procesar telemetría ---
    try:
        df = pd.read_csv(filename)
        df_with_frame = add_frame_column(df)
        df_filtered = filter_coordinates(df_with_frame)
        if df_filtered.empty: return
    except Exception as e:
        print(f"Error al procesar la telemetría: {e}")
        return

    # --- 2. Cargar Modelos YOLO ---
    try:
        segmentation_model = YOLO(segmentation_model_path)
        classification_model = YOLO(classification_model_path)
        detection_model = YOLO(detection_model_path)
    except Exception as e:
        print(f"Error al cargar modelos YOLO: {e}")
        return

    # --- 3. Procesamiento principal por grupos ---
    output_images_folder = "imagenes_procesadas_good"
    os.makedirs(output_images_folder, exist_ok=True)
    
    best_images_data = []
    for group_number, group_df in df_filtered.groupby('Grupo'):
        print(f"\n--- Procesando Grupo {group_number} ---")
        
        frame_indices = group_df['Frame'].tolist()
        extracted_paths = extract_frames(video_path, frame_indices, group_number)
        if not extracted_paths: continue

        laser_detected_candidates = []
        for image_path in extracted_paths:
            # -- Se utiliza la nueva función de detección y escalado --
            original_img, cm_px, area_m2 = detectar_y_escalar_por_pixeles(image_path, laser_params)
            
            if cm_px is not None:
                frame_index = int(os.path.splitext(os.path.basename(image_path))[0])
                laser_detected_candidates.append({
                    'frame_index': frame_index, 'original_img': original_img,
                    'cm_per_pixel': cm_px, 'total_area_m2': area_m2
                })

        if not laser_detected_candidates: continue

        # --- Clasificación y selección de la mejor imagen (sin cambios) ---
        good_candidates = []
        for candidate in laser_detected_candidates:
            results = classification_model.predict(candidate['original_img'], verbose=False)
            if classification_model.names[results[0].probs.top1] == 'good':
                candidate['confidence'] = results[0].probs.top1conf
                good_candidates.append(candidate)
        
        if not good_candidates: continue
        best_candidate = max(good_candidates, key=lambda x: x['confidence'])
        
        # --- Procesamiento final de la mejor imagen ---
        bc = best_candidate
        img_with_mask, mask_area_px = apply_segmentation_mask(bc['original_img'], segmentation_model)
        
        detection_results = detection_model.predict(img_with_mask, verbose=False)
        num_detections = len(detection_results[0].boxes)

        # Cálculo de densidad con área visible
        alto, ancho, _ = bc['original_img'].shape
        visible_area_px = (alto * ancho) - mask_area_px
        visible_area_m2 = (visible_area_px * bc['cm_per_pixel']**2) / 10000
        density = num_detections / visible_area_m2 if visible_area_m2 > 0 else 0

        # Almacenar resultados
        original_row = df[df['Frame'] == bc['frame_index']].iloc[0]
        best_images_data.append({
            'Frame': bc['frame_index'], 'Confidence': bc['confidence'],
            'NumDetections': num_detections, 'Density': density,
            'Visible_Area_m2': visible_area_m2, 'Total_Area_m2': bc['total_area_m2'],
            'Cm_Por_Pixel': bc['cm_per_pixel'], **original_row
        })
        
        # Guardar imagen procesada
        output_image_path = os.path.join(output_images_folder, f"grupo_{group_number}_frame_{bc['frame_index']}.jpg")
        cv2.imwrite(output_image_path, bc['original_img'])

    # --- 4. Guardar resultados finales ---
    if best_images_data:
        df_final = pd.DataFrame(best_images_data)
        df_final.to_csv("results_final.csv", index=False)
        print("\nProcesamiento completo. Resultados guardados en 'results_final.csv'")
    else:
        print("\nNo se procesó ninguna imagen con éxito.")


if __name__ == "__main__":
    main()