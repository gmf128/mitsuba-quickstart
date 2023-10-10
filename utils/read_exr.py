import os

os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import cv2
import numpy as np
import matplotlib.pyplot as plt



def imcrop(a, crop = {}):
    x = 0 if "x" not in crop.keys() else crop["x"]
    y = 0 if "y" not in crop.keys() else crop["y"]
    w = a.shape[1] if "w" not in crop.keys() else crop["w"]
    h = a.shape[0] if "h" not in crop.keys() else crop["h"]
    return a[y:y+h, x:x+w, :]

def readexr(a, crop = {}):

    imga = cv2.imread(a, cv2.IMREAD_UNCHANGED)
    if len(imga.shape) == 2: # copy 1 channel to 3
        imga = cv2.merge((imga, imga, imga))
    imga = imga.astype("float")[:, :, [2, 1, 0]]
    return imcrop(imga, crop)

# Test

a = "refs/cbox_ppg.exr"
img = readexr(a)
img = np.power(img, 1.0 / 2.2)
plt.imshow(img)
plt.pause(10)
