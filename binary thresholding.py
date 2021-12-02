import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# read RGB image as gray image
img = Image.open('images/3.jpg').convert('L')
img = np.array(img)

# binary thresholding
def binary_thresholding(image, threshold):
    temp = image.flatten()
    for i in range(len(temp)):
        if temp[i] >= threshold:
            temp[i] = 255
        else:
            temp[i] = 0
    temp = temp.reshape(image.shape)
    return temp

# histogram
histogram, bin_edges = np.histogram(img, bins=256)
plt.plot(bin_edges[:-1], histogram)
plt.xlabel("Pixel Value")
plt.ylabel("Count")
plt.xlim([0, 255])
plt.show()


# segmentation
segmented_image = binary_thresholding(img, 150)
Image.fromarray(segmented_image).show()
