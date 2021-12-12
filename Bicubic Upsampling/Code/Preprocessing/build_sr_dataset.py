"""
    Creates HDF5 dataset file for super resolution task.
    note: only the luminance channel of the YCbCr image is saved.
"""

# Import the necessary libraries
import os
import sys
import numpy as np
import h5py

from tqdm import tqdm
from PIL import Image

# Import the necessary source codes
from Preprocessing.distort_images import distort_image

# We need 4 arguments, so less than that will lead to a system fault
if len(sys.argv) < 4:
    print("Usage:", sys.argv[0], "<path_to_images/>", "<train_or_test_or_val/file_name.hdf5>",
          "<super_resolution_factor>")
    sys.exit()

# Dimension of square patch size to be used for training
PATCH_SIZE = 32

# Standard initialized constants
SR_FACTOR = int(sys.argv[3])
SIGMA = 1.0
PATH = sys.argv[1]

# Add support for images as BMP, JPG, PNG files
img_names_list = [f for f in os.listdir(PATH) if f[-4:] in {'.bmp', '.jpg', '.png'}]

# Initialise a file to write the output images
hdf5_file = h5py.File(sys.argv[2], "w")

# To store the intermediate data
training_patches = []
target_patches = []

# As we are creating image and target pairs
print("Creating image/target pairs...")

# Iterate through all the selected images
for idx in tqdm(range(len(img_names_list))):
    img_name = img_names_list[idx]

    # Load the luminance channel
    # Store the images as a file with the name
    ImageFile = Image.open(PATH + img_name)

    # Convert the image from the RGB to the YCbCr color coding with float data-type to give an enhanced color contrast
    im = np.array(ImageFile.convert('YCbCr'), dtype=np.float)

    # Crop image to be multiple of 8 in both dimension
    max_x = int(im.shape[0] / 8) * 8
    max_y = int(im.shape[1] / 8) * 8

    # As the image is a square any length can be taken as a dimension
    square_dim = min(max_x, max_y)

    # Breaking the image based on the YCbCr color space

    # First dimension of the image corresponds to the Y Color space
    im_Y = im[0:square_dim, 0:square_dim, 0]

    # Second dimension of the image corresponds to the Cb Color space
    im_Cb = im[0:square_dim, 0:square_dim, 1]

    # Third dimension of the image corresponds to the Cr Color space
    im_Cr = im[0:square_dim, 0:square_dim, 2]

    # Distort the image with blur, downsize and upsize bicubic interpolation

    # Refer to the documentation of the function in the source file

    # First dimension of the image is blurred with the necessary factors as parameters
    im_Y_blur = distort_image(path=PATH + img_name,
                              factor=SR_FACTOR,
                              sigma=SIGMA)[0:square_dim, 0:square_dim, 0].astype(np.float)

    # Need to extract to patches so that all images are the same size

    # Iterate through every patch vertically with stride 13
    for i in range(0, im_Y_blur.shape[1] - PATCH_SIZE, 13):
        for j in range(0, im_Y_blur.shape[0] - PATCH_SIZE, 13):
            # Take the image as 32 * 32 matrices
            sub_im_blur = im_Y_blur[j:j + PATCH_SIZE, i:i + PATCH_SIZE]
            sub_im = im_Y[j:j + PATCH_SIZE, i:i + PATCH_SIZE]

            # Completion of the pre-processing
            # Store the pre-processed images
            training_patches.append(sub_im_blur)
            target_patches.append(sub_im)

# Make a 3D shape out of the image matrices
data_shape = (len(training_patches), PATCH_SIZE, PATCH_SIZE)

# Create a dataset supporting hdf5 file system from the blurred and target images
hdf5_file.create_dataset("blurred_img", data_shape, np.float)
hdf5_file.create_dataset("target_img", data_shape, np.float)

# Build the hdf5 dataset
print("Building HDF5 dataset...")

# Iterate over each image
for i in tqdm(range(len(training_patches))):
    # Add each image to the build
    hdf5_file["blurred_img"][i, ...] = training_patches[i]
    hdf5_file["target_img"][i, ...] = target_patches[i]

# As we have built the dataset we can close the file
hdf5_file.close()

# Completion of execution
print("Done creating %d training pairs from %d original images" %
      (len(target_patches), len(img_names_list)))
