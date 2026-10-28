
import cv2
import numpy as np

from IrisLocalization import (N_BINARIZE_ITER, estimate_center_by_projection, refine_center_by_binarization)


#stated in Section 3.1
PATCH = 64
BANDS = [(0, 6), (6, 22), (22, 32)]


def quality_descriptor(patch):
    #Compute D = [F1 + F2 + F3, F2 / (F1 + F3)] from Equation (1).
    if patch.shape != (PATCH, PATCH):
        raise ValueError("Each quality patch must be 64 x 64 pixels.")

    # Equation (1)
    spectrum = np.abs(np.fft.fft2(patch.astype(float)))
    band_totals = [0.0, 0.0, 0.0]

    for row in range(PATCH):
        v = row
        if row >= PATCH // 2:
            v = row - PATCH

        for column in range(PATCH):
            u = column
            if column >= PATCH // 2:
                u = column - PATCH

            #Note that the FFT stores negative frequencies in its second half.
            radius = np.sqrt(u * u + v * v)
            for band in range(len(BANDS)):
                lower, upper = BANDS[band]
                if lower < radius <= upper:
                    band_totals[band] += spectrum[row, column]

    F1 = band_totals[0]
    F2 = band_totals[1]
    F3 = band_totals[2]
    if F1 + F3 == 0:
        raise ValueError("The quality ratio is undefined for this patch.")

    total_power = F1 + F2 + F3
    middle_ratio = F2 / (F1 + F3)
    return np.array([total_power, middle_ratio])


def iris_patches(img, xp, yp, rp):
    #Extract two 64 x 64 patches beside the coarsely located pupil
    height, width = img.shape
    if height < PATCH or width < PATCH:
        raise ValueError("The image is too small for the quality patches.")

    #each patch starts just outside the estimated pupil boundary
    right_x = int(round(xp + rp))
    left_x = int(round(xp - rp)) - PATCH
    y0 = int(round(yp)) - PATCH // 2
    y0 = max(0, min(y0, height - PATCH))

    patches = []
    for x0 in [right_x, left_x]:
        x0 = max(0, min(x0, width - PATCH))
        patch = img[y0:y0 + PATCH, x0:x0 + PATCH]
        patches.append(patch)
    return patches


def ImageQualityAssessment(img):
    #Return the mean local descriptor and the rough pupil circle
    if img is None:
        raise ValueError("The image could not be read.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError("Use a grayscale or BGR image with 8-bit pixel values.")

    xc, yc = estimate_center_by_projection(img)
    for i in range(N_BINARIZE_ITER):
        xc, yc, radius = refine_center_by_binarization(img, xc, yc)

    patches = iris_patches(img, xc, yc, radius)
    right_descriptor = quality_descriptor(patches[0])
    left_descriptor = quality_descriptor(patches[1])
    D = (right_descriptor + left_descriptor) / 2.0
    return D, (xc, yc, radius)



#decide clear / not clear
#In the paper, they use the image dataset with labeles, but CASIA does not hage lables
#Suppose that most images are clear and follows a normal distribution, and build a threshold to cut the tail
DEFAULT_THRESHOLDS = {"max_total_power": 2.44e6, "min_ratio": 0.695}

def fit_thresholds(D_train, k_power=4.0, k_ratio=3.0):
    D_train = np.array(D_train, dtype=float)
    power = D_train[:, 0] #first column: total power
    ratio = D_train[:, 1] #second column: middle-frequency ratio

    power_median = np.median(power) #typical
    power_spread = 1.4826 * np.median(np.abs(power - power_median)) #not typical value
    ratio_median = np.median(ratio)
    ratio_spread = 1.4826 * np.median(np.abs(ratio - ratio_median))

    max_total_power = power_median + k_power * power_spread #too high -> occluded
    min_ratio = ratio_median - k_ratio * ratio_spread #too low -> blurred
    return {"max_total_power": float(max_total_power), "min_ratio": float(min_ratio)}


def is_qualified(D, thresholds=DEFAULT_THRESHOLDS):
    total_power = D[0]
    middle_ratio = D[1]
    if total_power > thresholds["max_total_power"]:
        return False, "occluded (total power too high)"
    if middle_ratio < thresholds["min_ratio"]:
        return False, "blurred (too little middle frequency)"
    return True, "clear"