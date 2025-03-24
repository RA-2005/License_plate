import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread("car.jpg");

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY);

plt.imshow(gray,cmap ='gray')
plt.show()