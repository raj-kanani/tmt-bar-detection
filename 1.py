import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread('1.jpg')
blur_hor = cv2.filter2D(img[:, :, 0], cv2.CV_32F, kernel=np.ones((11,1,1), np.float32)/11.0, borderType=cv2.BORDER_CONSTANT)
blur_vert = cv2.filter2D(img[:, :, 0], cv2.CV_32F, kernel=np.ones((1,11,1), np.float32)/11.0, borderType=cv2.BORDER_CONSTANT)
mask = ((img[:,:,0]>blur_hor*1.2) | (img[:,:,0]>blur_vert*1.2)).astype(np.uint8)*255

plt.imshow(mask)

circles = cv2.HoughCircles(mask,
                           cv2.HOUGH_GRADIENT,
                           minDist=8,
                           dp=1,
                           param1=150,
                           param2=12,
                           minRadius=10,
                           maxRadius=15)
output = img.copy()
for (x, y, r) in circles[0, :, :]:
  cv2.circle(output, (x, y), r, (0, 255, 0), 4)

