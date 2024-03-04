import numpy as np
import cv2
from matplotlib import pyplot as plt
img = cv2.imread("C:/Users/VIJITHA REDDY/OneDrive/Pictures/Screenshots/tinytan.jpg")
b, g, r = cv2.split(img)
rgb_img = cv2.merge([r, g, b])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
kernel = np.ones((5, 5), np.uint8)
closing = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
sure_bg = cv2.dilate(closing, kernel, iterations=3)
plt.subplot(121), plt.imshow(closing, 'gray')
plt.title("morphologyEx: Closing"), plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(sure_bg, 'gray')
plt.title("Dilation"), plt.xticks([]), plt.yticks([])
plt.tight_layout()
plt.show()
