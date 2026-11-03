"""
Step 4: Feature Extraction (section 3.3):
Include functions for iris feature extraction using the spatial filter proposed by Ma et al. (2003).
Take normalized and enhanced iris image as input (shape 64*512) and extract 1536 features for each image.
"""

import numpy as np
import cv2 as cv

# helper function 1 for feature_extraction
def create_filter(delta_x, delta_y, f):
    """
    Build a spatial filter kernel using Ma et al.'s method (reference: eq.3)

    Args:
        delta_x (float): parameters for gaussian envelope. Larger delta_x gives a wider spread along x-axis, so further pixels from the center could been considered.
        delta_y (float): parameters for gaussian envelope. Similar to delta_x, larger delta_y gives a wider spread along y-axis.
        f (float): control the spatial frequency in m1 (the second part of the function). 
                   A larger f captures high-frequency patterns better, and a lower f captures lower ones better.  
                   0.05 <= f <= 0.4 has been explored in previous Gabor-based texture analysis research. 
    """

    def m1(x, y): 
        """
        The second half of the spatial filter. Represent a cosine function that depends on the distance from the center sqrt(x^2 + y^2)

        Arg:
            x (np.ndarray): horizontal coordinates (ksize, ksize)
            y (np.ndarray): vertical coordinates (ksize, ksize)
        """
        return np.cos(2 * np.pi * f * (x**2 + y**2)**(0.5))
    
    def G(x, y):
        """
        The full formula to create the spatial filter in Ma et al.

        Args:
            x (np.array): horizontal coordinates (ksize*ksize)
            y (np.array): vertical coordinates (ksize*ksize)

        Returns:
            kernel (np.array): spatial filter kernel
        """
        # first_half = Gaussian envelope: largest at the center, fading out with distance, with a different spread along x and y
        first_half = np.exp(-0.5 * (x**2 / delta_x**2 + y**2 / delta_y**2)) / (2 * np.pi * delta_x * delta_y)
        sec_half = m1(x, y)     # cosine wave part

        return first_half * sec_half

    # determine kernel size for the filter
    # take the larger delta value and make the kernel being large enough to hold almost all of the Gaussian value
    # ksize equation comes from the 3-sigma rule, making sure it covers most of the Gaussian envelope
    # Add additional 1 to make sure the kernel size is odd (so there would be a center for (0,0)
    max_delta = max(delta_x, delta_y)
    ksize = int(2 * np.ceil(3 * max_delta) + 1)

    # create 2 coordinate matrices x, y for all kernel position
    # x contains horizontal coordinates and y contains vertical ones, where (0,0) located at the center
    half_size = ksize // 2      # the number of pixels from the center to the edge.
    y, x = np.mgrid[-half_size:half_size+1, -half_size:half_size+1]    

    # kernel = the final spatial filtering kernel
    kernel = G(x, y)    # plug into the filter formula at every cell

    return kernel

# helper function 2 for feature_extraction
def get_features(img):
    """
    Get a list of features from a given filtered region of interest iris image (48*512).

    Args:
        img (np.ndarray): filtered region of interest iris image.

    Returns:
        features (list): extracted features from the image (768 elements).
    """
    #Describe a filtered ROI (48 x 512) with two numbers per 8 x 8 block
    # an empty list to save all captured features
    features = []

    # Ma et al. cut the full roi into 384 8*8 small block, and extract 2 kinds of features from each small block.
    # loop through each block and calculate the 2 features and append it to the full list
    for y in range(0, 48, 8):
        for x in range(0, 512, 8):
            # get out the small block of the image
            block = img[y:y+8, x:x+8]

            # m is the first feature.
            # mag is the magnitue of each filter response. Sum up mag and take average over 64 pixels as the feature "m"
            mag = np.abs(block) 
            m = np.sum(mag) / 64 

            # sigma is the second feature represent the average distance from the mean (m)
            sigma = np.sum(np.abs(mag - m)) / 64 

            # save the features to the full list
            features.append(m)
            features.append(sigma)

    return features

# main function to be imported to IrisRecognition.py
def feature_extraction(img, f=0.05): 
    """
    Extract features from a normalized and enhanced grayscale iris image and return an array of features.

    Args:
        img (np.ndarray): normalized and enhanced grayscale iris image (shape=64*512)
        f (float): cosine frequency passed to create_filter. A larger f captures high-frequency patterns better, and a lower f captures lower ones better.  
                           0.05 <= f <= 0.4 has been explored in previous Gabor-based texture analysis research.

    Return:
        full_features (np.array): extracted 1536 features from the normalized, enhanced iris image.
    """
    # roi = Region of interest (48*512). Paper states to only consider the first 48 rows, as iris area close to pupil 
    #       would contain more useful texture information, and would be less influenced by eyelids and eyelashes.
    roi = img[:48, :]

    # filter parameters reference to (delta_x, delta_y) in the paper.
    # Ma et al. uses 2 sets of parameters in their filtering method. After applying the two filters, we will get 2 images
    # to be used to extract features.
    filter_parms = [(3, 1.5), (4.5, 1.5)]

    # a list to save all the extracted features. Expected to have 1536 features.
    full_features = []

    # loop through each filter, apply it to the roi, use get_features to get the features, and save them to the full list.
    for delta_x, delta_y in filter_parms:
        # create a spatial filter kernel with the given parameters delta_x and delta_y
        filter_kernel = create_filter(delta_x, delta_y, f)

        # apply the spatial filter to the roi
        # filtered_roi is the region of interest image (array) after filtering
        # cv.CV_32F marks the data type for each pixel being 32 bit float. 
        filtered_roi = cv.filter2D(roi, cv.CV_32F, filter_kernel)

        # get 768 features from the filtered roi
        features = get_features(filtered_roi)

        # save the features to the full list
        full_features.extend(features)

    return np.array(full_features)
