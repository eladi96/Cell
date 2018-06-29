# ---------------------------------------------------
# Cells extraction using the watershed method from
# OpenCv library
# ---------------------------------------------------

# import the necessary packages
import glob
import os
import time
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import cv2

from PIL import ImageEnhance, Image
from skimage import io
from skimage.measure import regionprops
from skimage.feature import peak_local_max
from skimage.morphology import watershed
from scipy import ndimage
from datetime import timedelta


def detections_cells(image):

    # perform pyramid mean shift filtering
    # to aid the thresholding step
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
    d = ndimage.distance_transform_edt(thresh)
    local_max = peak_local_max(d, indices=False, min_distance=20,
                               labels=thresh)

    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(local_max, structure=np.ones((3, 3)))[0]
    labels = watershed(-d, markers, mask=thresh)

    # Remove lables too small
    filtered_labels = np.copy(labels)
    component_sizes = np.bincount(labels.ravel())
    too_small = component_sizes < 10000
    too_small_mask = too_small[labels]
    filtered_labels[too_small_mask] = 1

    return filtered_labels


def extraction_cells(image, c):

    # if the directory doesn't exist then create a new one
    in_path = os.getcwd() + "/"
    directory = in_path + "/" + "cellule/"
    if not os.path.exists(directory):
        os.makedirs("cellule/agglomerati/")

    filtered_labels = detections_cells(image)

    fig, ax = plt.subplots(figsize=(10, 6))

    i = 0
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


# Main execution
if __name__ == "__main__":

    start_time = time.monotonic()
    c = 0
    path = os.getcwd() + "/" + "inputImages/"
    for infile in glob.glob(os.path.join(path, '*.JPG')):
        print(infile)
        img_or = Image.open(infile)
        img_en = ImageEnhance.Color(img_or)
        img_en = img_en.enhance(2)

        img_contr = ImageEnhance.Contrast(img_en)
        img_final = img_contr.enhance(1.5)

        img_final.save(path + "enhanced" + str(c) + ".png")
        img_opened = io.imread(path + "enhanced" + str(c) + ".png")

        try:
            extraction_cells(img_opened, c)
        except ValueError:
            continue
        c = c + 1

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
