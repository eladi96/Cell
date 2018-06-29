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
from skimage.measure import label, regionprops


def detection_cells(img_or):
    # selection of RGB's red channel
    channel_r = img_or[..., 0]

    # thresholding otsu to separate cells from background
    thresh = filters.threshold_otsu(channel_r)  # thresh = 131
    # img_thresh è una immagine binaria con i pixel <131 bianchi il resto neri
    img_thresh = channel_r < thresh

    # closing
    selem = morphology.disk(5)
    img_thresh = morphology.erosion(img_thresh, selem)
    io.imshow(img_thresh)

    # label image regions
    label_image = label(img_thresh)

    # delete labels too big
    filtered_image = np.copy(label_image)
    component_sizes = np.bincount(label_image.ravel())
    # too_big = component_sizes > 100000
    # too_big_mask = too_big[label_image]
    # filtered_image[too_big_mask] = 1

    # delete labels too small
    too_small = component_sizes < 4000
    too_small_mask = too_small[label_image]
    filtered_image[too_small_mask] = 1

    return filtered_image


def extraction_cells(img_or, c):
    # if the directory doesn't exist then create a new one
    in_path = os.getcwd() + "/"
    directory = in_path + "/" + "cellule/"
    if not os.path.exists(directory):
        os.makedirs("cellule/")
    img = detection_cells(img_or)

    fig, ax = plt.subplots(figsize=(10, 6))

    i = 0

    for region in regionprops(img):

        # draw circle around cells
        minr, minc, maxr, maxc = region.bbox
        x, y = region.centroid
        diam = region.equivalent_diameter
        circle = mpatches.Circle((y, x), radius=diam,
                                 fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(circle)

        cell = img_or[minr:maxr + 10, minc:maxc + 10]

        if i != 0:
            io.imsave("cellule/image" + str(c) + "_cell" + str(i) + ".png", cell)

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
            extraction_cells(img_opened, c)
        except ValueError:
            continue
        c = c + 1

end_time = time.monotonic()
print(timedelta(seconds=end_time - start_time))
