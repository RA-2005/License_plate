import cv2 as cv
import matplotlib.pyplot as plt

image = cv.imread("car.jpg")

gray = cv.cvtColor(image,cv.COLOR_BGR2GRAY)

plt.imshow(gray,cmap ='gray')
plt.title("Grayscale")
plt.show()