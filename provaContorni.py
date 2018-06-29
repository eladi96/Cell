# ---------------------------------------------------
# Tentativo di estrazione utilizzando la funzione
# watershed di OpenCv
# ---------------------------------------------------

# import the necessary packages
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
import os
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

import numpy as np
from skimage import io
from skimage.measure import regionprops
import cv2

image = cv2.imread('IMG00022.JPG')
shifted = cv2.pyrMeanShiftFiltering(image, 21, 51)

# convert the mean shift image to grayscale, then apply
# Otsu's thresholding
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255,
                       cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
cv2.imshow("Thresh", thresh)

# compute the exact Euclidean distance from every binary
# pixel to the nearest zero pixel, then find peaks in this
# distance map
D = ndimage.distance_transform_edt(thresh)
localMax = peak_local_max(D, indices=False, min_distance=20,
                          labels=thresh)

# perform a connected component analysis on the local peaks,
# using 8-connectivity, then appy the Watershed algorithm
markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
labels = watershed(-D, markers, mask=thresh)

# Remove lables too small
filtered_labels = np.copy(labels)
component_sizes = np.bincount(labels.ravel())
too_small = component_sizes < 10000
too_small_mask = too_small[labels]
filtered_labels[too_small_mask] = 1

# # loop over the unique labels returned by the Watershed
# # algorithm
# for label in np.unique(filtered_labels):
#     # if the label is zero, we are examining the 'background'
#     # so simply ignore it
#     if label == 0:
#         continue
#
#     # otherwise, allocate memory for the label region and draw
#     # it on the mask
#     mask = np.zeros(gray.shape, dtype="uint8")
#     mask[filtered_labels == label] = 255
#
#     # detect contours in the mask and grab the largest one
#     cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
#                             cv2.CHAIN_APPROX_SIMPLE)[-2]
#     c = max(cnts, key=cv2.contourArea)
#
#     # draw a circle enclosing the object
#     ((x, y), r) = cv2.minEnclosingCircle(c)
#     cv2.circle(image, (int(x), int(y)), int(r), (0, 255, 0), 2)
#     cv2.putText(image, "#{}".format(label), (int(x) - 10, int(y)),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

i = 0

# if the directory doesn't exist then create a new one
in_path = os.getcwd() + "/"
directory = in_path + "/" + "cellule/agglomerati/"
if not os.path.exists(directory):
    os.makedirs("cellule/agglomerati/")

fig, ax = plt.subplots(figsize=(10, 6))

i = 0
c = 0

for region in regionprops(filtered_labels):

    # draw circle around cells
    minr, minc, maxr, maxc = region.bbox
    x, y = region.centroid
    diam = region.equivalent_diameter
    circle = mpatches.Circle((y, x), radius=diam,
                             fill=False, edgecolor='red', linewidth=2)
    ax.add_patch(circle)

    cell = image[minr:maxr + 10, minc:maxc + 10]

    if i != 0:
        io.imsave("cellule/agglomerati/image" + str(c) + "_cell" + str(i) + ".png", cell)

    i = i + 1

ax.set_axis_off()
io.imshow(image)
plt.show()

# # show the output image
# cv2.imshow("Output", image)
# cv2.waitKey(0)
