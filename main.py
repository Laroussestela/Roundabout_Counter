import cv2
import time
import numpy as np
import os
from datetime import datetime
from ultralytics import YOLO
from sort import Sort

# Áreas de color (rectángulos)
RED_AREA = np.array([[650, 210], [710, 210], [710, 270], [650, 270]])
YELLOW_AREA = np.array([[500, 430], [570, 430], [570, 490], [500, 490]])
BLUE_AREA = np.array([[820, 420], [870, 420], [870, 490], [820, 490]])
GREEN_AREA = np.array([[690, 560], [780, 560], [780, 595], [690, 595]])

# Diccionarios para hacer seguimiento
cross_areas = {
    'red': {},
    'yellow': {},
    'blue': {},
    'green': {}
}

# Contadores de vehículos por zona
vehicle_counts = {
    'red': 0,
    'yellow': 0,
    'blue': 0,
    'green': 0
}

# Colores para dibujar
area_colors = {
    'red': (0, 0, 255),
    'yellow': (0, 255, 255),
    'blue': (255, 0, 0),
    'green': (0, 255, 0)
}

# Polígonos para detección
areas = {
    'red': RED_AREA,
    'yellow': YELLOW_AREA,
    'blue': BLUE_AREA,
    'green': GREEN_AREA
}

# Crear la carpeta para guardar los frames si no existe
output_folder = 'final_frames'
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

if __name__ == '__main__':
    cap = cv2.VideoCapture('rotonda.mp4')
    VIDEO_FPS = cap.get(cv2.CAP_PROP_FPS)

    model = YOLO('E:/00. Kaggle/21. Yolo rotonda/train_yolo/runs_yolov11/car_detector5/weights/best.pt')
    tracker = Sort()

    frame_count = 0  # Contador para el nombre de archivo de los frames

    while cap.isOpened():
        status, frame = cap.read()
        if not status:
            break

        frame = cv2.resize(frame, (1280, 720))
        results = model(frame, stream=True)

        for res in results:
            filtered_indices = np.where((np.isin(res.boxes.cls.cpu().numpy(), [0])))[0]
            boxes = res.boxes.xyxy.cpu().numpy()[filtered_indices].astype(int)
            tracks = tracker.update(boxes).astype(int)

            for xmin, ymin, xmax, ymax, track_id in tracks:
                xc, yc = int((xmin + xmax) / 2), int((ymin + ymax) / 2)
                shown = False

                for area_name, polygon in areas.items():
                    if track_id not in cross_areas[area_name] and cv2.pointPolygonTest(polygon, (xc, yc), False) >= 0:
                        vehicle_counts[area_name] += 1
                        cross_areas[area_name][track_id] = vehicle_counts[area_name]

                    if track_id in cross_areas[area_name]:
                        shown = True
                        color = area_colors[area_name]
                        order_number = cross_areas[area_name][track_id]
                        # Dibuja el recuadro y el número de orden
                        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
                        cv2.putText(frame, str(order_number), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                if shown:
                    cv2.circle(frame, (xc, yc), 5, (255, 255, 255), -1)

        # Dibuja las áreas de conteo
        for area_name, polygon in areas.items():
            cv2.polylines(frame, [polygon], True, area_colors[area_name], 2)

        # Guardar el frame con un nombre único basado en el contador
        frame_filename = os.path.join(output_folder, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
