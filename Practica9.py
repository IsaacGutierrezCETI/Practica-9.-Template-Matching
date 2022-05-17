#Isaac Alejandro Gutiérrez Huerta 19110198 7E1
#Sistemas de Visión Artificial

import cv2
import numpy as np

img_rgb = cv2.imread('Pastillas.jpg')
img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)

roi = cv2.selectROI('ROI',img_rgb)
recorteROI = img_rgb[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
cv2.imwrite("ROI.jpg",recorteROI)

template = cv2.imread('ROI.jpg',0)
w, h = template.shape[::-1]

res = cv2.matchTemplate(img_gray,template,cv2.TM_CCOEFF_NORMED)
threshold = 0.85
loc = np.where( res >= threshold)

for pt in zip(*loc[::-1]):
    cv2.rectangle(img_rgb, pt, (pt[0] + w, pt[1] + h), (0,255,255), 2)

cv2.imshow('Detected',img_rgb)

cv2.waitKey(0)
cv2.destroyAllWindows()
