from ultralytics import YOLO
import os
import cv2

model = YOLO('../Models/Shapes_segmentation_YOLO.pt')

# IMG_PATH = '../Experiment/Try1/savedImages/0.8667388684528475.png'
# image = cv2.imread(IMG_PATH, cv2.COLOR_BGR2RGB)
# image = cv2.resize(image, (128,128))
#
# result = model.predict(source= image, show=True, conf = 0.20,save = True, iou = 0.6)
# print(result)

print (model.names)
