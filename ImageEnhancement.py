"""
Step 3: Image Enhancement (section 3.2.3)
Include functions for iris image enhancement using method proposed by Ma et al. (2003).
Take normalized iris image as input (shape 64*512), estimate and remove background illumination, and then do histogram equalization.
"""

import numpy as np
import cv2 as cv

# helper function 1 for image_enhancement
def appro_intensity(img):
    """
    Estimate intensity variations across the whole image.

    Args:
        img (np.ndarray): normalized iris image (shape 64*512)

    Returns:
        full_bgl (np.ndarray): estimated background illumination (4 * 32)
    
    """
    # empty list to save estimated background illumination
    full_bgl = []
    
    # calculate average pixel intensity for each 16 * 16 small block as estimation for background illumination
    # loop through every 16 rows
    for y in range(0, 64, 16):
        # empty list to save the background illumination for the current 16 rows
        ls_bgl = []
        # loop through every 16 columns
        for x in range(0, 512, 16):
            # get out the small block of the image
            block = img[y:y+16, x:x+16]

            # calculate the mean value of each small block and append to the list for the current rows
            mean = np.mean(block)
            ls_bgl.append(mean)

        # append the list for the current rows to the full background illumination estimation list
        full_bgl.append(ls_bgl)

    return np.array(full_bgl)

# helper function 2 for image_enhancement
def histogram_equalization(img):
    """
    Apply histogram equalization on every 32*32 small block from the image.

    Args:
        img (np.ndarray): normalized iris image after removing background illumination (shape 64*512)

    Returns:
        full_eq (np.ndarray): image after histogram equalization
    """
    # create an empty 2D array to save the equalized values
    full_eq = np.empty_like(img)

    # do histogram equalization on every 32*32 small block to improve local contrast. (Ma et al.)
    # loop through each small block
    for y in range(0, 64, 32):
        for x in range(0, 512, 32):
            # get out the small block from the input image
            block = img[y:y+32, x:x+32]

            # apply histogram equalization and save the values to the full equalized 2D array
            eq = cv.equalizeHist(block)
            full_eq[y:y+32, x:x+32]  = eq

    return full_eq

# main function to be imported to IrisRecognition.py
def image_enhancement(img):
    """
    Take normalized iris image as input (shape 64*512), estimate and remove background illumination, and then do histogram equalization.

    Args:
        img (np.ndarray): normalized iris image (64*512)

    Returns:
        enhanced_img (np.ndarray): enhanced iris image (64*512)
    """
    # transform image type to np.float32 for later calculation
    img = img.astype(np.float32)

    # get the estimated background illumination (4 * 32)
    full_bgl = appro_intensity(img)

    # extend to 64 * 512 so the size is the same as the normalized image
    # cv.INTER_CUBIC: bicubic interpolation to estimate intermediate values to create a smoother background illumination map (Ma et al.)
    estimated = cv.resize(full_bgl, (512, 64), interpolation=cv.INTER_CUBIC)

    # subtracted estimated background illumination from the normalized image 
    # corrected_img = image after removing the background illumination
    corrected_img = img - estimated

    # use cv.normalize deal with out-of-range (0~255) values so we can do histogram equalizatoin to the image
    corrected_img = cv.normalize(corrected_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # histogram equalization
    enhanced_img = histogram_equalization(corrected_img)

    return enhanced_img
