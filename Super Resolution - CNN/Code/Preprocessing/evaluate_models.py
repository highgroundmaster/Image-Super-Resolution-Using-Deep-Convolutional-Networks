"""
    Functions for evaluating model performance.
"""

# Import the necessary libraries
import os
import numpy as np
import torch

from PIL import Image
from skimage.metrics import peak_signal_noise_ratio
from skimage.measure import compare_ssim

# Import the necessary source codes
from Preprocessing.distort_images import distort_image
from Preprocessing.utils import ycbcr2rgb

# Initialise the default device
DEVICE = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def evaluate_model(path, model, pixel_mean, pixel_std, SR_FACTOR=3, sigma=1):

    """
        Computes average Peak Signal to Noise Ratio (PSNR) and mean structural similarity
        index (SSIM) over a set of target images and their super resolved versions.

        Args:
            path (string): relative path to directory containing images for evaluation
            model (PyTorch model): the model to be evaluated
            pixel_mean (float): mean luminance value to be used for standardization
            pixel_std (float): std. dev. of luminance value to be used for standardization
            SR_FACTOR (int): super resolution factor
            sigma (int): the std. dev. to use for the gaussian blur
    """

    # Store all the training images based on their extension
    # Add support for images as BMP, JPG, PNG files
    img_names = [im for im in os.listdir(path) if im[-4:] in {'.bmp' '.jpg','.png'}]

    # To store the error values
    blurred_img_psnrs = []
    out_img_psnrs = []
    blurred_img_ssims = []
    out_img_ssims = []

    # Iterate through the images
    for test_im in img_names:

        # Generate the distorted image
        blurred_test_im = distort_image(path=path+test_im, factor=SR_FACTOR, sigma=sigma)
        ImageFile = Image.open(path+test_im)

        # Convert the image from the RGB to the YCbCr color coding with float data-type to give an enhanced color contrast
        im = np.array(ImageFile.convert('YCbCr'))

        # Normalize the images into the standard size of 256 pixels
        model_input = blurred_test_im[:, :, 0] / 255.0

        # As the Mean and Standard Deviation have been pre-computed

        # Standardize the images by subtracting Mean and Dividing Standard Deviation
        model_input -= pixel_mean
        model_input /= pixel_std

        # Build a Tensor out of the images and set it to the default device
        im_out_Y = model(torch.tensor(model_input,
                                      dtype=torch.float).unsqueeze(0).unsqueeze(0).to(DEVICE))

        im_out_Y = im_out_Y.detach().squeeze().squeeze().cpu().numpy().astype(np.float64)
        im_out_viz = np.zeros((im_out_Y.shape[0], im_out_Y.shape[1], 3))

        # Unstandardize the images by Multiplying Standard Deviation and Adding Mean
        im_out_Y = (im_out_Y * pixel_std) + pixel_mean

        # Un-normalize the images
        im_out_Y *= 255.0

        im_out_viz[:, :, 0] = im_out_Y
        im_out_viz[:, :, 1] = im[:, :, 1]
        im_out_viz[:, :, 2] = im[:, :, 2]

        im_out_viz[:, :, 0] = np.around(im_out_viz[:, :, 0])

        # Compute the Peak Signal to Noise Ratio

        # Refer to the documentation of the function in the source file
        blur_psnr = peak_signal_noise_ratio(ycbcr2rgb(im), ycbcr2rgb(blurred_test_im))
        sr_psnr = peak_signal_noise_ratio(ycbcr2rgb(im), ycbcr2rgb(im_out_viz))

        # Store the results
        blurred_img_psnrs.append(blur_psnr)
        out_img_psnrs.append(sr_psnr)

        # Compute the Structural Similarity Index

        # Refer to the documentation of the function in the source file
        blur_ssim = compare_ssim(ycbcr2rgb(im), ycbcr2rgb(blurred_test_im), multichannel=True)
        sr_ssim = compare_ssim(ycbcr2rgb(im), ycbcr2rgb(im_out_viz), multichannel=True)

        # Store the results
        blurred_img_ssims.append(blur_ssim)
        out_img_ssims.append(sr_ssim)

    # Compute the mean of the results obtained
    mean_blur_psnr = np.mean(np.array(blurred_img_psnrs))
    mean_sr_psnr = np.mean(np.array(out_img_psnrs))
    mean_blur_ssim = np.mean(np.array(blurred_img_ssims))
    mean_sr_ssim = np.mean(np.array(out_img_ssims))

    # Return the outputs received
    return mean_blur_psnr, mean_sr_psnr, mean_blur_ssim, mean_sr_ssim
