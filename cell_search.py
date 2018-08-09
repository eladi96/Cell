import math
import operator
import cv2
import os
from PIL import ImageChops as imgch, Image
from functools import reduce
import matplotlib.pyplot as plt


def cell_rms(img_rgb, img_gray, template_gray, template_rgb):
    # serve a trovarne solo uno e non molti
    img2 = img_gray.copy()

    # serve a trovarne solo uno e non molti
    img = img2.copy()

    # Apply template Matching using SQDIFF method and save image coordinates
    res = cv2.matchTemplate(img, template_gray, cv2.TM_SQDIFF)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    top_left = min_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)
    match = img_rgb[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    # Calculate the difference between images
    diff = imgch.difference(Image.fromarray(template_rgb), Image.fromarray(match)).histogram()

    # calculate rms
    rms = math.sqrt(reduce(operator.add, map(lambda diff, i: diff * (i ** 2), diff, range(256)))
                    / (float(Image.fromarray(template_rgb).size[0]) * Image.fromarray(template_rgb).size[1]))

    return rms, top_left


# Main exectution

if __name__ == "__main__":

    template_rgb = cv2.cvtColor(cv2.imread('cellulina.png'), cv2.COLOR_BGR2RGB)
    # Opens the template to research in grayscale (flag = 0)
    template_gray = cv2.imread('cellulina.png', 0)
    # width and height of template
    w, h = template_gray.shape[::-1]

    min_rms = 100

    for file in os.listdir(os.getcwd() + "/inputImages"):
        img_rgb = cv2.cvtColor(cv2.imread(os.getcwd() + "/inputImages/" + file), cv2.COLOR_BGR2RGB)
        img_gray = cv2.imread(os.getcwd() + "/inputImages/" + file, 0)
        rms, tl = cell_rms(img_rgb, img_gray, template_gray, template_rgb)

        if rms < min_rms:
            min_rms = rms
            match = img_rgb
            top_left = tl

    # Draws circle around the found match on RGB image
    x = top_left[0] + w // 2
    y = top_left[1] + h // 2
    cv2.circle(match, (x, y), (w // 2), (255, 0, 0), 3)
    plt.imshow(match)
    plt.title('Detected Point'), plt.xticks([]), plt.yticks([])
    plt.show()
