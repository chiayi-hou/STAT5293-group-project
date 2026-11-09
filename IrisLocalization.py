import sys
import cv2
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_circle

#Parameters
REGION_SIZE = 120 #side length of the square region binarized in step 2
N_BINARIZE_ITER = 2 #number of times step 2 is repeated

HIST_OFFSET = 10 #added to the pupil's grey level to get the binarization threshold, so slightly brighter pupil-edge pixels are included
PUPIL_R_SEARCH = 8 #in step 3, the pupil center and radius may differ by at most this many px from the step 2 estimates
IRIS_R_MIN = 90
IRIS_R_MAX = 135 #smallest and largest outer iris radius tried by the Hough transform
IRIS_CENTER_SEARCH = 10 #the outer circle's center may be at most this many px from the pupil center (circles are not concentric)
CANNY_SIGMA = 2.0 #amount of Gaussian smoothing inside Canny before edges are found
IRIS_CANNY_LOW = 3
IRIS_CANNY_HIGH = 6 #Canny thresholds for the outer boundary; low because the iris/sclera edge is soft
OPEN_SIZE = 5 #square size used to erase eyelashes in step 2


def estimate_center_by_projection(img):
    #Step 1: projection along the x and y axes
    #column_sums: total brightness of each column; row_sums: total brightness of each row
    column_sums = np.sum(img, axis=0)
    row_sums = np.sum(img, axis=1)

    xp = int(np.argmin(column_sums)) #position of the smallest column total
    yp = int(np.argmin(row_sums)) #position of the smallest row total
    return xp, yp


def crop_box(img, xc, yc, size):
    #Return the top-left corner of a square kept inside the image
    #(xc, yc): point the square should be centered on; size: side length of the square
    #(x0, y0): returned top-left corner (y grows downward, so y0 is the TOP edge)
    height, width = img.shape
    if height < size or width < size:
        raise ValueError("The image is smaller than the requested square.")

    x0 = int(round(xc)) - size // 2 #half a box to the left of the center
    y0 = int(round(yc)) - size // 2 #half a box above the center
    x0 = max(0, min(x0, width - size)) #keep the box inside left or right edges
    y0 = max(0, min(y0, height - size)) #keep the box inside top or bottom edges
    return x0, y0


def histogram_threshold(region, offset=HIST_OFFSET):
    #Choose a threshold from the dark part of the local histogram
    #pupil_peak: the most common dark grey level in the region (the pupil); threshold: pupil_peak + offset
    histogram = np.bincount(region.flatten(), minlength=256) #pixel count per grey level 0..255
    #Only grey levels 0..99 are searched, so a big bright area (skin, white of the eye) can never be mistaken for the pupil. Pupils in dataset are 33...64
    pupil_peak = int(np.argmax(histogram[:100])) #!!!Here choose 100 as the threshold for dark, could change according to real result!!!
    threshold = pupil_peak + offset
    return threshold


def refine_center_by_binarization(img, xc, yc, size=REGION_SIZE):
    #Step 2: use the binary region's centroid and area
    #(xc, yc): the current guess of the pupil center
    #(x0, y0): top-left corner of the 120 x 120 region; region: the cropped part of the image
    x0, y0 = crop_box(img, xc, yc, size)
    region = img[y0:y0 + size, x0:x0 + size]
    threshold = histogram_threshold(region)
    binary = (region <= threshold).astype(np.uint8) #1 = dark enough to be pupil

    #square: 5 x 5 structuring element; the opening removes dark structures thinner than 5 px (eyelashes)
    square = np.ones((OPEN_SIZE, OPEN_SIZE), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, square) #remove eyelashes

    #Each True pixel belongs to the estimated dark pupil region.
    #rows, columns: positions (inside the region) of all remaining pupil pixels
    rows, columns = np.nonzero(binary)
    if len(columns) == 0:
        raise ValueError("No pupil pixels were found. Check the threshold.")

    #new_x, new_y: centroid (average position) of the pupil pixels, converted back to full-image coordinates by adding x0, y0
    new_x = x0 + float(np.mean(columns))
    new_y = y0 + float(np.mean(rows))

    #For a circular pupil, assume area = pi * radius**2.
    #area: number of pupil pixels; rough_radius: radius of a circle with that area
    area = len(columns)
    rough_radius = np.sqrt(area / np.pi)
    return new_x, new_y, float(rough_radius)


def hough_best_circle(edges, radii, center_box):
    #Find the circle with the largest Hough vote in the center box
    #edges: black-and-white edge map from Canny (True = edge pixel)
    #radii: list of radii to try
    #center_box: (xmin, xmax, ymin, ymax), the area where the circle's center is allowed to be
    height, width = edges.shape
    xmin, xmax, ymin, ymax = center_box
    #Keep the allowed box inside the edge map
    xmin = max(0, xmin)
    xmax = min(width - 1, xmax)
    ymin = max(0, ymin)
    ymax = min(height - 1, ymax)

    if xmin > xmax or ymin > ymax or len(radii) == 0:
        raise ValueError("The circle search region is empty.")

    votes = hough_circle(edges, radii) #votes[i] is the vote image for radius radii[i]
    #best_vote: highest vote found so far; best_circle: (x, y, radius) of that circle
    best_vote = 0.0
    best_circle = None

    #Check each radius separately
    #local_votes: votes for radius radii[i], only inside the allowed center box
    #position: index of the highest vote in local_votes (as one number), converted to row and column
    for i in range(len(radii)):
        local_votes = votes[i, ymin:ymax + 1, xmin:xmax + 1]
        position = int(np.argmax(local_votes))
        row = position // local_votes.shape[1]
        column = position % local_votes.shape[1]
        vote = local_votes[row, column]

        if vote > best_vote: #keep this circle if it beats the best one found so far
            best_vote = vote
            x = xmin + column #convert box position back to edge-map position
            y = ymin + row
            radius = radii[i]
            best_circle = (float(x), float(y), float(radius))

    if best_circle is None:
        raise ValueError("No circle was found. Check the image and search settings.")
    return best_circle


def detect_pupil_circle(img, xc, yc, rough_radius):
    #Step 3: refine the pupil using Canny edges and circular Hough votes
    #(xc, yc), rough_radius: the estimates from step 2
    #region: 140 x 140 crop around the pupil; edges: its Canny edge map
    size = REGION_SIZE + 20 #a box a little bigger than in step 2
    x0, y0 = crop_box(img, xc, yc, size)
    region = img[y0:y0 + size, x0:x0 + size]
    edges = canny(region, sigma=CANNY_SIGMA)

    #radii: candidate pupil radii, rough_radius +/- PUPIL_R_SEARCH (at least 10 px)
    minimum_radius = max(10, int(rough_radius) - PUPIL_R_SEARCH)
    maximum_radius = int(rough_radius) + PUPIL_R_SEARCH
    radii = np.arange(minimum_radius, maximum_radius + 1)

    #(local_x, local_y): step 2 center inside the crop; center_box: allowed centers, +/- PUPIL_R_SEARCH around it
    local_x = int(round(xc - x0)) #the Hough center uses coordinates inside the cropped region
    local_y = int(round(yc - y0))
    center_box = (
        local_x - PUPIL_R_SEARCH, local_x + PUPIL_R_SEARCH,
        local_y - PUPIL_R_SEARCH, local_y + PUPIL_R_SEARCH,
    )
    #x, y, radius: best circle in crop coordinates; adding x0, y0 converts back to the full image
    x, y, radius = hough_best_circle(edges, radii, center_box)
    return x + x0, y + y0, radius


def detect_iris_circle(img, pupil): 
    #Step 3: search for the outer boundary near the pupil center
    #pupil: (xp, yp, pupil_radius) from detect_pupil_circle
    #half_size: half the side of the crop around the pupil, big enough for the largest allowed iris plus the center shift
    #(x0, y0) to (x1, y1): corners of that crop, kept inside the image; region: the crop; edges: its Canny edge map
    xp, yp, pupil_radius = pupil
    height, width = img.shape
    half_size = IRIS_R_MAX + IRIS_CENTER_SEARCH + 5
    x0 = max(0, int(round(xp)) - half_size)
    y0 = max(0, int(round(yp)) - half_size)
    x1 = min(width, int(round(xp)) + half_size + 1)
    y1 = min(height, int(round(yp)) + half_size + 1)
    region = img[y0:y1, x0:x1]
    edges = canny(
        region, sigma=CANNY_SIGMA,
        low_threshold=IRIS_CANNY_LOW, high_threshold=IRIS_CANNY_HIGH,
    )

    #radii: candidate outer radii, from max(IRIS_R_MIN, pupil radius + 25) to IRIS_R_MAX
    #(center_x, center_y): pupil center inside the crop; center_box: allowed centers, +/- IRIS_CENTER_SEARCH around it
    minimum_radius = max(IRIS_R_MIN, int(pupil_radius) + 25) #!!!radius limits can be modified according to real situation!!!
    radii = np.arange(minimum_radius, IRIS_R_MAX + 1)
    center_x = int(round(xp - x0))
    center_y = int(round(yp - y0))
    center_box = (
        center_x - IRIS_CENTER_SEARCH, center_x + IRIS_CENTER_SEARCH,
        center_y - IRIS_CENTER_SEARCH, center_y + IRIS_CENTER_SEARCH,
    )

    x, y, radius = hough_best_circle(edges, radii, center_box)
    return x + x0, y + y0, radius


def IrisLocalization(img):
    #img : gray eye image (8-bit)
    #(xc, yc): current pupil-center estimate, improved by each step; rough_radius: pupil radius estimate from step 2
    #pupil, iris: final circles from step 3
    if img is None:
        raise ValueError("The image could not be read.")
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.ndim != 2 or img.dtype != np.uint8:
        raise ValueError("Use a grayscale or BGR image with 8-bit pixel values.")

    #step 1
    xc, yc = estimate_center_by_projection(img)

    #step 2
    for i in range(N_BINARIZE_ITER):
        xc, yc, rough_radius = refine_center_by_binarization(img, xc, yc)

    #step 3
    pupil = detect_pupil_circle(img, xc, yc, rough_radius)
    iris = detect_iris_circle(img, pupil)

    return pupil, iris #Return pupil (x, y, radius) and iris (x, y, radius)

#stated in Section 3.1 Image Assessment 
#PATCH: side length (px) of each of the two iris patches used for quality assessment
#BANDS: (lower, upper] radial frequency ranges for low (F1), middle (F2) and high (F3) frequencies, in cycles per patch
PATCH = 64
BANDS = [(0, 6), (6, 22), (22, 32)]


#Image Quality Assessment (Section 3.1 from paper)
def quality_descriptor(patch):
    #Compute D = [F1 + F2 + F3, F2 / (F1 + F3)] from Equation (1).
    if patch.shape != (PATCH, PATCH):
        raise ValueError("Each quality patch must be 64 x 64 pixels.")

    # Equation (1)
    #spectrum: |F(u, v)|, the strength of every frequency of the patch (2-D Fourier transform)
    #band_totals: running sums of the spectrum in the three bands -> F1, F2, F3
    spectrum = np.abs(np.fft.fft2(patch.astype(float)))
    band_totals = [0.0, 0.0, 0.0]

    #u, v: horizontal and vertical frequency of the current spectrum position (negative in the second half)
    #radius: sqrt(u^2 + v^2), how fast the pattern changes regardless of direction; decides the band
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

    F1 = band_totals[0] #low
    F2 = band_totals[1] #middle
    F3 = band_totals[2] #high
    if F1 + F3 == 0:
        raise ValueError("The quality ratio is undefined for this patch.")

    #total_power: very high -> iris covered by eyelashes (occluded)
    #middle_ratio: middle-frequency share; low -> blurred image
    total_power = F1 + F2 + F3
    middle_ratio = F2 / (F1 + F3)
    return np.array([total_power, middle_ratio])


def iris_patches(img, xp, yp, rp):
    #Extract two 64 x 64 patches beside the coarsely located pupil
    #(xp, yp, rp): rough pupil circle from steps 1-2
    #right_x, left_x: left edge of the right patch and of the left patch; y0: top edge of both patches
    #patches: [right patch, left patch]
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
    #Main function for quality assessment
    #1. Rough pupil location, using steps 1-2 of the localization above
    #2. Two 64 x 64 patches beside the pupil
    #3. Quality descriptor of each patch, then the average of the two
    #Return the mean local descriptor and the rough pupil circle
    #(xc, yc), radius: rough pupil circle; D: [total power, middle ratio] averaged over the two patches
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
#DEFAULT_THRESHOLDS: limits computed with fit_thresholds on the 324 training images (session 1);
#  max_total_power: above this -> occluded; min_ratio: below this -> blurred
DEFAULT_THRESHOLDS = {"max_total_power": 2.44e6, "min_ratio": 0.695}

def fit_thresholds(D_train, k_power=4.0, k_ratio=3.0):
    #Learn the normal range of D from training images
    #D_train: one row [total power, middle ratio] per training image
    #k_power: how many spreads above normal counts as occluded
    #k_ratio: how many spreads below normal counts as blurred

    D_train = np.array(D_train, dtype=float)
    power = D_train[:, 0] #first column: total power
    ratio = D_train[:, 1] #second column: middle-frequency ratio

    #*_median: typical value; *_spread: 1.4826 x median absolute deviation, a robust estimate of the standard deviation
    #(median and MAD are used because the few bad images would pull the mean and standard deviation toward themselves)
    power_median = np.median(power) #typical
    power_spread = 1.4826 * np.median(np.abs(power - power_median)) #not typical value
    ratio_median = np.median(ratio)
    ratio_spread = 1.4826 * np.median(np.abs(ratio - ratio_median))

    max_total_power = power_median + k_power * power_spread #too high -> occluded
    min_ratio = ratio_median - k_ratio * ratio_spread #too low -> blurred
    return {"max_total_power": float(max_total_power), "min_ratio": float(min_ratio)}


def is_qualified(D, thresholds=DEFAULT_THRESHOLDS):
    #D: [total power, middle ratio] from ImageQualityAssessment
    #thresholds: limits from fit_thresholds (default: DEFAULT_THRESHOLDS)
    #returns (True/False, reason)
    total_power = D[0]
    middle_ratio = D[1]
    if total_power > thresholds["max_total_power"]:
        return False, "occluded (total power too high)"
    if middle_ratio < thresholds["min_ratio"]:
        return False, "blurred (too little middle frequency)"
    return True, "clear"

#Quick test: python IrisLocalization.py image.bmp prints the two circles found in that image
if __name__ == "__main__":
    image_path = "001_1_1.bmp"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    pupil_circle, iris_circle = IrisLocalization(image)
    print("Pupil (x, y, radius):", pupil_circle)
    print("Iris (x, y, radius):", iris_circle)
