import sys
import numpy as np
import cv2

img = cv2.imread('./image/parking04.png')
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imwrite('./image/parking04_hsv.png', hsv)