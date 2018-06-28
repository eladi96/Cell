import glob
import os
import time
from datetime import timedelta
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from PIL import ImageEnhance, Image

import numpy as np
from skimage import filters
from skimage import io
from skimage import morphology
from skimage.color import rgb2gray
from skimage.measure import label, regionprops


def detectionCells(img_or):
    # selection of RGB's red channel
    channel_r = img_or[..., 0]

    # thresholding otsu to separate cells from background
    thresh = filters.threshold_otsu(channel_r)  # thresh = 131
    img_thresh = channel_r < thresh  # img_thresh è una immagine binaria con i pixel <131 bianchi il resto neri

    # closing
    selem = morphology.disk(5)
    img_thresh = morphology.erosion(img_thresh, selem)
    io.imshow(img_thresh)

    # label image regions
    label_image = label(img_thresh)

    # delete labels too big
    filtered_image = np.copy(label_image)
    component_sizes = np.bincount(label_image.ravel())
    too_big = component_sizes > 100000
    too_big_mask = too_big[label_image]
    filtered_image[too_big_mask] = 1

    # delete labels too small
    too_small = component_sizes < 4000
    too_small_mask = too_small[label_image]
    filtered_image[too_small_mask] = 1

    return filtered_image


def extractionCells(img_or, c):
    # if the directory doesn't exist then create a new one
    in_path = os.getcwd() + "/"
    directory = in_path + "/" + "cellule/"
    if not os.path.exists(directory):
        os.makedirs("cellule/")
    img = detectionCells(img_or)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(img_or)

    i = 0
    j = 0
    n_labels = 0
    for region in regionprops(img):

        # take regions with large enough areas
        if region.area >= 200:
            # get pixels belonging to the bounding box
            minr, minc, maxr, maxc = region.bbox

            # draw circle around cells
            minr, minc, maxr, maxc = region.bbox
            x, y = region.centroid
            diam = region.equivalent_diameter
            circle = mpatches.Circle((y, x), radius=diam,
                                     fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(circle)

            cell = img_or[minr:maxr + 10, minc:maxc + 10]
            # transform image from rgb to gray
            img_grey = rgb2gray(cell)

            # thresholding otsu to separate cells from background
            thresh = filters.threshold_otsu(img_grey)
            img_thresh = img_grey < thresh

            # opening
            selem = morphology.disk(7)
            img_thresh = morphology.opening(img_thresh, selem)

            # label image regions
            label_image = label(img_thresh)

            for r in regionprops(label_image):
                n_labels = n_labels + 1

            # if labels number is > 1 then it means there are more than one cell in the image
            if (n_labels > 1 and i != 0):
                # delete labels too small
                filtered_subimage = morphology.remove_small_objects(label_image, 300)

                # extract cells from the subimage
                for region2 in regionprops(filtered_subimage):
                    # take regions with large enough areas

                    # get pixels belonging to the bounding box
                    minr, minc, maxr, maxc = region2.bbox

                    sub_cell = cell[minr:maxr + 10, minc:maxc + 10]

                    # save cell subimage
                    io.imsave("cellule/cellula" + str(i) + "_tile" + str(c) + str(j) + ".png", sub_cell)
                    j = j + 1
            else:
                # save cell image
                io.imsave("cellule/cellula" + str(i) + "_tile" + str(c) + ".png", cell)

            if (i == 0):
                # remove tile image
                os.remove("cellule/cellula" + str(i) + "_tile" + str(c) + ".png")
            i = i + 1

    ax.set_axis_off()
    io.imshow(img_or)
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
            extractionCells(img_opened, c)
        except ValueError:
            continue
        c = c + 1
end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
