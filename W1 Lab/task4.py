# citing https://docs.opencv.org/ and its various pages

import numpy as np
import cv2

cap = cv2.VideoCapture(0)

# HSV color range for blue
lower = np.array([100, 150, 0])
upper = np.array([140, 255, 255])

while(True):

    ret, frame = cap.read()

    # no frame found exit
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

# 1. HSV seems to have better masking / better object isolation, 
# whereas BGR is more sensitive but detects the object more readily (along with other objects tho)
# For HSV, I tested H = 90-130, S = 100-255, and V = 100-255, with range sizes of 40, 155, 155 respectively.
# For BGR, I tested B = 100-255, G = 0-150, and R = 0-130 using the color wheel with range sizes of 
# 155, 150, and 130 respectively.

# 2. I tested with HSV under low light, and it tracked poorly, unable to isolate the object fully but 
# still detecting it.

# 3.  Testing with varying the brightness on my phone and seeing if the color was able to still be 
# detected, it worked well with my brightness turned down all the way, but could not isolate the color 
# at all with a high brightness level. This most likely has to do with how HSV uses shadows and hue to 
# color detect, with brightness changing its ability to see the color. One factor that could have affected
# results is the natural lighting in my room, which could cause glare. 

# see 4 code & answer in new file 