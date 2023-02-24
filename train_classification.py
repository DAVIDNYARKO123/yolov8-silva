# Load YOLOv8n-cls, train it on mnist160 for 3 epochs and predict an image with it
from ultralytics import YOLO

model = YOLO('weights/yolov8n-cls.pt')  # load a pretrained YOLOv8n classification model
model.train(data='/Users/davidnyarko/Documents/KOBY/KobyGitHub/yolov8-silva/datasets/animals', epochs=100)  # train the model
model('inference/images/bird.jpeg')  # predict on an image