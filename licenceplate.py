import cv2
import numpy
import pytesseract
import immutils

pytesseract.pytesseract.tesseract_cmd = r'Downloads/tesseract-ocr-w32-setup-v5.0.0-alpha.20201127.exe'

car = cv2.imread('C://car1.jpeg', cv2.IMREAD_COLOR)
car = cv2.resize(car, (620, 480))

bnw = cv2.cvtColor(car, cv2.COLOR_BGR2GRAY)
bnw = cv2.bilateralFilter(bnw, 13, 15, 15)

edgedetect = cv2.Canny(bnw, 30, 200)
contourdetect = cv2.findContours(edgedetect.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contourdetect = imutils.grab_contours(contourdetect)
contourdetect = sorted(contourdetect, key=cv2.contourArea, reverse=True)[:10]
ctr = None

for c in contourdetect:
   # The contour is approximated
  
   perimeter = cv2.arcLength(c, True)
   epsilon = 0.018 * perimeter
   approx = cv2.approxPolyDP(c, epsilon, True)

   # 	If the approximated contour has four points then 
   # we have found our screen

   if len(approx) == 4:
       ctr = approx
   break

if ctr is None:
   flag = 0
   print("Contour is not detected")
else:
   flag = 1

if flag == 1:
   cv2.drawContours(car, [ctr], -1, (0, 0, 255), 3)

# Now we mask the other part of the image 

mask = numpy.zeros(bnw.shape, numpy.uint8)
new_image = cv2.drawContours(mask, [ctr], 0, 255, -1, )
new_image = cv2.bitwise_and(car, car, mask=mask)

# Here we crop the main image

(x, y) = numpy.where(mask == 255)
(left, high) = (numpy.min(x), numpy.min(y))
(right, low) = (numpy.max(x), numpy.max(y))
Crop = bnw[left:right + 1, high:low + 1]

#Reading of the number plate is done

licenceno = pytesseract.image_to_string(Crop, config='--psm 11')

print("The Licence Plate Number in the given image is :", licenceno)
car = cv2.resize(car, (500, 300))
Crop = cv2.resize(Crop, (400, 200))
cv2.imshow('Car Image ', car)
cv2.imshow('Cropped Image', Crop)

cv2.waitKey(0)
cv2.destroyAllWindows()
