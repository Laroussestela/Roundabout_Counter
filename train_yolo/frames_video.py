import cv2
import os

video_path = 'rotonda.mp4'
output_folder = 'frames'

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_num = 0
while True:
    success, frame = cap.read()
    if not success:
        break
    frame_filename = os.path.join(output_folder, f'frame_{frame_num:05d}.jpg')
    cv2.imwrite(frame_filename, frame)
    frame_num += 1

cap.release()
print(f'Total de frames extraídos: {frame_num}')
