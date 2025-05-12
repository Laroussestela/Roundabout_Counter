from ultralytics import YOLO

data_yaml = 'data.yaml'

model = YOLO('yolo11n.pt')

results = model.train(
    data=data_yaml,
    epochs=500,
    patience=15,
    project='runs_yolov11',
    name='car_detector'
)