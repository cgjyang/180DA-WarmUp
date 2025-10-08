import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# HSV color range for blue
lower = np.array([100, 150, 0])
upper = np.array([140, 255, 255])

while(True):

    ret, frame = cap.read()
    if not ret: 
        break

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

    cv2.imshow('camera', frame)
    cv2.imshow('mask', mask)

    # q to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()