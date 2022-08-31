import cv2
from util import load_image, letterbox

img = cv2.imread('img.png')

img_size = 640

img, (h0, w0), (h, w) = load_image(img, img_size)
img, ratio, pad, cratio = letterbox(img, (img_size, img_size), auto=False, scaleup=False)

cv2.imwrite('img1.png', img)