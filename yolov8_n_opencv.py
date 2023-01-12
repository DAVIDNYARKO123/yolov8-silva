import numpy as np
import cv2
from ultralytics import YOLO
import random

# opening the file in read mode
my_file = open("utils/coco.txt", "r")
# reading the file
data = my_file.read()
# replacing end splitting the text | when newline ('\n') is seen.
class_list = data.split("\n")
my_file.close()

# print(class_list)

# Generate random colors for class list
detection_colors = []
for i in range(len(class_list)):
    r = random.randint(0,255)
    g = random.randint(0,255)
    b = random.randint(0,255)
    detection_colors.append((b,g,r))

# load a pretrained YOLOv8n model
model = YOLO("weights/yolov8n.pt", "v8") 

# Vals to resize video frames | small frame optimise the run 
frame_wid = 640
frame_hyt = 480

# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture("inference/videos/afriq0.MP4")

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    #  resize the frame | small frame optimise the run 
    # frame = cv2.resize(frame, (frame_wid, frame_hyt))

    # write frame to be loaded by the prediction module
    cv2.imwrite("inference/images/frame.png",frame)

    # Predict on image 
    detect_params = model.predict(source="inference/images/frame.png", conf=0.45,save=False)

    # Convert tensor array to numpy
    print(detect_params[0].numpy())
    detect_params = detect_params[0].numpy()

    if len(detect_params) !=0:

        # Loop through all detections in current frame
        for param in detect_params:
            # print(param[1])

            # Draw BBox around detection
            cv2.rectangle(frame, (int(param[0]),int(param[1])), (int(param[2]), int(param[3])), detection_colors[int(param[5])], 3)

            # Display class name and confidence
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(frame,class_list[int(param[5])]+" "+str(round(param[4],3))+"%",(int(param[0]),int(param[1])-10),font,1,(255,255,255),2)

    # Display the resulting frame
    cv2.imshow('ObjectDetection', frame)

    # Terminate run when "Q" pressed
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()