from sdc_main import compute_binary_image, compute_perspective_transform
import cv2
import numpy as np
from matplotlib import pyplot as plt

refPt = []
cropping = False
 
 # get coordinates of double click on image
mouse_click_coords = []
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        print("MOUSE POS---------->",x, y)
        mouse_click_coords.append([x, y])
        print(mouse_click_coords)


cv2.namedWindow("image")
cv2.setMouseCallback("image", draw_circle)
img = cv2.imread('gta.png')

binary = compute_binary_image(img)

pres, _ = compute_perspective_transform(binary)

histogram = np.sum(pres[int(pres.shape[0]/2):, :], axis=0)
print(histogram.shape)
for i in range(800):
    if np.sum(pres[int(pres.shape[0]):,i], axis=0) != 0:
        print(np.sum(pres[int(pres.shape[0]):,i], axis=0))
        break
plt.plot(histogram)
plt.show()

cv2.imshow('image', pres)

k = cv2.waitKey(0)
if k == 27:         # wait for ESC key to exit
    cv2.destroyAllWindows()
elif k == ord('s'): # wait for 's' key to save and exit
    cv2.destroyAllWindows()