
import cv2
 
img = cv2.imread('no_signal_train_1-0001.jpg', cv2.IMREAD_UNCHANGED)

print(type(img))

exit(0)

print('Original Dimensions : ',img.shape)

scale_percent = 60 # percent of original size
width = 224
height = 224
dim = (width, height)

resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)

print('Resized Dimensions : ',resized.shape)

cv2.imwrite('output.jpg', resized)
