import cv2
import pytesseract

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from Plate_Detection import license_plate
# Load image


# Preprocessing: Convert to grayscale
gray = cv2.cvtColor(license_plate, cv2.COLOR_BGR2GRAY)

# Optional: Thresholding to increase contrast
_, thresh = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY)

# OCR
custom_config = r'--oem 3 --psm 8'  # Assume a single word or line
text = pytesseract.image_to_string(thresh, config=custom_config)

print("Detected Text:", text.strip())
