import numpy
import cv2

img = numpy.zeros([255,255,3])

img[:,:,0] = numpy.ones([255,255])*64/255.0
img[:,:,1] = numpy.ones([255,255])*128/255.0
img[:,:,2] = numpy.ones([255,255])*192/255.0

cv2.imwrite('color_img.jpg', img)
cv2.imshow("image", img)
cv2.waitKey()