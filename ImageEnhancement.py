import numpy as np
import cv2 as cv

def appro_intensity(img):
    """approximate intensity variations across the whole image"""
    full_bgl = []
    
    # 16 * 16 small block
    for y in range(0, 64, 16):
        ls_bgl = []
        for x in range(0, 512, 16):
            block = img[y:y+16, x:x+16]

            mean = np.mean(block)
            ls_bgl.append(mean)

        full_bgl.append(ls_bgl)

    return np.array(full_bgl)

def histogram_equlization(img):
    
    full_eq = np.empty_like(img)

    # 32 * 32 region
    for y in range(0, 64, 32):
        for x in range(0, 512, 32):
            block = img[y:y+32, x:x+32]

            eq = cv.equalizeHist(block)
            full_eq[y:y+32, x:x+32]  = eq

    return full_eq

def image_enhancement(img):
    img = img.astype(np.float32)

    # 4 * 32
    full_bgl = appro_intensity(img)

    # extend to 64 * 512
    estimated = cv.resize(full_bgl, (512, 64), interpolation=cv.INTER_CUBIC)

    # subtracted estimated backgroud illumination from the normalized image
    corrected_img = img - estimated

    # deal with negative values
    corrected_img = cv.normalize(corrected_img, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)

    # histogram equlization
    enhanced_img = histogram_equlization(corrected_img)

    return enhanced_img
