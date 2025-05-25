import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
from Reading_image import gray,image

blur = cv.GaussianBlur(gray,(5,5),0)

plt.imshow(blur, cmap='gray')
plt.title("Blurred Image")
plt.show()

edges = cv.Canny(blur,50,150)

plt.imshow(edges, cmap='gray')
plt.title("Edge Detection")
plt.show()


#Finding Contours
contours, _ = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)


#Dividing contours
contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]


image_copy = image.copy()
cv.drawContours(image_copy, contours, -1, (0, 255, 0), 2)

license_plate = None

for contour in contours:
   
    approx = cv.approxPolyDP(contour, 0.02 * cv.arcLength(contour, True), True)
    
    
    if len(approx) == 4:
        x, y, w, h = cv.boundingRect(approx)
        aspect_ratio = w / h
        
       
        if 2 < aspect_ratio < 5:
            license_plate = image[y:y+h, x:x+w]
            break


if license_plate is not None:
    
    plt.imshow(cv.cvtColor(license_plate, cv.COLOR_BGR2RGB))
    plt.title("License Plate")
    plt.show()
else:
    print("License plate not found")


gray_plate = cv.cvtColor(license_plate, cv.COLOR_BGR2GRAY)

# Apply edge detection
edges_plate = cv.Canny(gray_plate, 50, 150)

# Display the edges of the license plate
plt.imshow(edges_plate, cmap='gray')
plt.title("Edges of License Plate")
plt.show()