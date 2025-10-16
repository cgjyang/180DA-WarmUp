# code adapted from: https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
# code adapted from: https://docs.opencv.org/ and its various pages
# memory leak issue avoided using assistance of chatGPT

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import os
os.environ["OMP_NUM_THREADS"] = "1"


def find_histogram(kmeans):

    num_labels = np.arange(0, len(np.unique(kmeans.labels_)) + 1)
    hist, _ = np.histogram(kmeans.labels_, bins=num_labels)

    # normalize to percentages
    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors(hist, centers):

    # make blank image
    bar = np.zeros((50, 300, 3), dtype = "uint8")
    start_x = 0

    for percent, color in zip(hist, centers):
        # draw color as a rect bsed on percentage present
        end_x = start_x + int(percent * 300)
        cv2.rectangle(bar, (start_x, 0), (end_x, 50), color.astype("uint8").tolist(), -1)
        start_x = end_x
    
    return bar

cap = cv2.VideoCapture(0)

# HSV color range for blue
lower = np.array([100, 150, 0])
upper = np.array([140, 255, 255])

while(True):

    ret, frame = cap.read()

    # no frame found exit
    if not ret: 
        break
    
    # so green bounding box color isn't included in dominant colors
    frame_copy = frame.copy()

    # BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # make a mask with HSV range
    mask = cv2.inRange(hsv, lower, upper)

    # find contours in mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # draw bounding box if largest contour found
    if contours:

        c = max(contours, key = cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)

        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

        # isolate bounding box
        roi = frame_copy[y:y+h, x:x+w]

        # change BGR to RGB for processing
        if roi.size > 0:
            try:
                roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                pixels = roi_rgb.reshape((-1, 3))

                clt = KMeans(n_clusters = 3, n_init = 10)
                clt.fit(pixels)

                hist = find_histogram(clt)
                bar = plot_colors(hist, clt.cluster_centers_)

                bar_bgr = cv2.cvtColor(bar, cv2.COLOR_RGB2BGR)
                cv2.imshow("dominant colors", bar_bgr)

            except:
                pass


    cv2.imshow('camera', frame)
    cv2.imshow('mask', mask)

    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

# 4. The object seems to be more robust to brightness because there was glare appearing on the 
# phone, which affected how well the detection was able to work.