import os
import random

import numpy as np
from PIL import Image


def get_image_crops(image, crop_height, crop_width):
    max_x = int(image.shape[0] / 8) * 8
    max_y = int(image.shape[1] / 8) * 8

    # Crop Dimension Validation
    if (max_y % crop_width) or (max_x % crop_width):
        raise ValueError(
            f'Given crop dimensions ({crop_height}, {crop_width}) are not compatible with the given Image Dimensions')

    max_x = image.shape[1] - crop_width
    max_y = image.shape[0] - crop_height

    crops = []

    for i in range(0, max_x, crop_width):
        for j in range(0, max_y, crop_height):
            # Take the image as crop_height*crop_width matrices
            sub_im = image[j:j + crop_height, i: i + crop_width]
            crops.append(sub_im)

    return crops


def upsample_images(path, num_images, crop_height=160, crop_width=160):
    toupsample = [image_path for image_path in os.listdir(path) if image_path[-4:] in {'.jpg','.png','.bmp'}]
    image_compare = []
    for test_im in random.sample(toupsample, num_images):
        img_path = path + test_im
        original_image = np.array(Image.open(img_path))
        dim0 = original_image.shape[0]
        dim1 = original_image.shape[1]

        crops = get_image_crops(original_image, crop_height, crop_width)
        upsampled_images = [np.array(Image.fromarray(crop, mode='YCbCr').resize((dim1, dim0), Image.BICUBIC)) for crop
                            in
                            crops]
        image_compare.append([original_image, crops, upsampled_images])

    return image_compare
