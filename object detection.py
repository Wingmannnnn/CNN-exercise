import cv2
import torch
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot as plt
import uuid
import os
import time


model = torch.hub.load("ultralytics/yolov5", "yolov5s")
# img = "https://tse2.mm.bing.net/th?id=OIP.7Cv0U_4Wp3-yuj9XPm12qwHaFj&pid=Api&P=0"
# result = model(img)
# print(result)

# plt.imshow(np.squeeze(result.render()))
# plt.show()

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    if ret:

        result = model(frame)
        cv2.imshow("frame", np.squeeze(result.render()))

    if cv2.waitKey(1) == 27:
        break
cap.release()
cv2.destroyAllWindows