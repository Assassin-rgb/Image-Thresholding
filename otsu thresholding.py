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
        if temp[i] > threshold:
            temp[i] = 255
        else:
            temp[i] = 0
    temp = temp.reshape(image.shape)
    return temp

# get probability
def get_probability(image, threshold):
    total = np.sum(image)
    temp = 0
    for i in range(0, threshold+1):
        temp += image[i]
    return temp/total

# get means
def get_mean(image, threshold, probability):
    p1 = probability
    p2 = 1 - p1
    mean1 = 0
    mean2 = 0
    for i in range(threshold+1):
        mean1 += i * image[i]
    for i in range(threshold+1, 256):
        mean2 += i * image[i]
    return mean1/p1, mean2/p2

# otsu thresholding
def otsu_thresholding(image):
    g_max = 200
    s_max = 0
    u = 50
    threshold = 50
    histogram, bin_edges = np.histogram(img, bins=256)
    plt.plot(bin_edges[:-1], histogram)
    plt.xlabel("Pixel Value")
    plt.ylabel("Count")
    plt.xlim([0, 255])
    plt.show()
    while (u < g_max):
        prob_p1 = get_probability(histogram, u)
        mean_c1, mean_c2 = get_mean(histogram, u, prob_p1)
        variance = prob_p1 * (1 - prob_p1) * (mean_c1 - mean_c2)**2
        if variance > s_max:
            s_max = variance
            threshold = u
        u += 1
    print('Threshold = ',threshold)
    return binary_thresholding(image, threshold)

# segmentation
segmented_image = otsu_thresholding(img)
Image.fromarray(segmented_image).show()
