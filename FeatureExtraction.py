"""
Feature Extraction
"""
import numpy as np
import cv2 as cv

def create_filter(delta_x, delta_y, f):
    """
    Build one of the paper's spatial filters (eq.3)
    test: 0.05 < f < 0.4
    """
    def m1(x, y): #a cosine wave that depends on the distance from the center sqrt(x^2 + y^2)
        return np.cos(2 * np.pi * f * (x**2 + y**2)**(0.5))
    
    def G(x, y):
        # Gaussian envelope: largest at the center, fading out with distance, with a different spread along x and y
        first_half = np.exp(-0.5 * (x**2 / delta_x**2 + y**2 / delta_y**2)) / (2 * np.pi * delta_x * delta_y)
        sec_half = m1(x, y)
        return first_half * sec_half

    #Make the kernel large enough to hold almost all of the Gaussian
    max_delta = max(delta_x, delta_y)
    ksize = int(2 * np.ceil(3 * max_delta) + 1)

    half_size = ksize // 2
    y, x = np.mgrid[-half_size:half_size+1, -half_size:half_size+1] #coordinates of every kernel cell relative to the center

    kernel = G(x, y) #evaluate the filter formula at every cell
    return kernel

def get_features(img):
    #Describe a filtered ROI (48 x 512) with two numbers per 8 x 8 block
    features = []

    #8 * 8 small block
    for y in range(0, 48, 8):
        for x in range(0, 512, 8):
            block = img[y:y+8, x:x+8]

            mag = np.abs(block) #strength of the response, sign ignored
            m = np.sum(mag) / 64 #mean over the 64 pixels of the block
            sigma = np.sum(np.abs(mag - m)) / 64 #average distance from that mean

            features.append(m)
            features.append(sigma)

    return features

def feature_extraction(img, f=0.05): 
    #f: cosine frequency passed to create_filter

    roi = img[:48, :] #size: 48 * 512
    filter_parms = [(3, 1.5), (4.5, 1.5)]

    full_features = []

    for delta_x, delta_y in filter_parms:
        filter_kernel = create_filter(delta_x, delta_y, f)

        # use filter to do convolution (48*512)
        filtered_roi = cv.filter2D(roi, cv.CV_32F, filter_kernel)

        features = get_features(filtered_roi)

        full_features.extend(features)

    return np.array(full_features)
