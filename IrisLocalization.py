import sys
import cv2
import numpy as np
from skimage.feature import canny
from skimage.transform import hough_circle

#Parameters
REGION_SIZE = 120
N_BINARIZE_ITER = 2

HIST_OFFSET = 10
PUPIL_R_SEARCH = 8
IRIS_R_MIN = 90
IRIS_R_MAX = 135
IRIS_CENTER_SEARCH = 10
CANNY_SIGMA = 2.0
IRIS_CANNY_LOW = 3
IRIS_CANNY_HIGH = 6
OPEN_SIZE = 5 #square size used to erase eyelashes in step 2


def estimate_center_by_projection(img):
    #Step 1: projection along the x and y axes
    column_sums = np.sum(img, axis=0)
    row_sums = np.sum(img, axis=1)

    xp = int(np.argmin(column_sums))
    yp = int(np.argmin(row_sums))
    return xp, yp


def crop_box(img, xc, yc, size):
    #Return the top-left corner of a square kept inside the image
    height, width = img.shape
    if height < size or width < size:
        raise ValueError("The image is smaller than the requested square.")

    x0 = int(round(xc)) - size // 2
    y0 = int(round(yc)) - size // 2
    x0 = max(0, min(x0, width - size))
    y0 = max(0, min(y0, height - size))
    return x0, y0


def histogram_threshold(region, offset=HIST_OFFSET):
    #Choose a threshold from the dark part of the local histogram
    histogram = np.bincount(region.flatten(), minlength=256)
    pupil_peak = int(np.argmax(histogram[:100])) #!!!Here choose 100 as the threshold for dark, could change according to real result!!!
    threshold = pupil_peak + offset
    return threshold


def refine_center_by_binarization(img, xc, yc, size=REGION_SIZE):
    #Step 2: use the binary region's centroid and area
    x0, y0 = crop_box(img, xc, yc, size)
    region = img[y0:y0 + size, x0:x0 + size]
    threshold = histogram_threshold(region)
    binary = (region <= threshold).astype(np.uint8)

    square = np.ones((OPEN_SIZE, OPEN_SIZE), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, square) #remove eyelashes

    #Each True pixel belongs to the estimated dark pupil region.
    rows, columns = np.nonzero(binary)
    if len(columns) == 0:
        raise ValueError("No pupil pixels were found. Check the threshold.")

    new_x = x0 + float(np.mean(columns))
    new_y = y0 + float(np.mean(rows))

    #For a circular pupil, assume area = pi * radius**2.
    area = len(columns)
    rough_radius = np.sqrt(area / np.pi)
    return new_x, new_y, float(rough_radius)


def hough_best_circle(edges, radii, center_box):
    #Find the circle with the largest Hough vote in the center box
    height, width = edges.shape
    xmin, xmax, ymin, ymax = center_box
    xmin = max(0, xmin)
    xmax = min(width - 1, xmax)
    ymin = max(0, ymin)
    ymax = min(height - 1, ymax)

    if xmin > xmax or ymin > ymax or len(radii) == 0:
        raise ValueError("The circle search region is empty.")

    votes = hough_circle(edges, radii)
    best_vote = 0.0
    best_circle = None

    #Check each radius separately
    for i in range(len(radii)):
        local_votes = votes[i, ymin:ymax + 1, xmin:xmax + 1]
        position = int(np.argmax(local_votes))
        row = position // local_votes.shape[1]
        column = position % local_votes.shape[1]
        vote = local_votes[row, column]

        if vote > best_vote:
            best_vote = vote
            x = xmin + column
            y = ymin + row
            radius = radii[i]
            best_circle = (float(x), float(y), float(radius))

    if best_circle is None:
        raise ValueError("No circle was found. Check the image and search settings.")
    return best_circle


def detect_pupil_circle(img, xc, yc, rough_radius):
    #Step 3: refine the pupil using Canny edges and circular Hough votes
    size = REGION_SIZE + 20
    x0, y0 = crop_box(img, xc, yc, size)
    region = img[y0:y0 + size, x0:x0 + size]
    edges = canny(region, sigma=CANNY_SIGMA)

    minimum_radius = max(10, int(rough_radius) - PUPIL_R_SEARCH)
    maximum_radius = int(rough_radius) + PUPIL_R_SEARCH
    radii = np.arange(minimum_radius, maximum_radius + 1)

    local_x = int(round(xc - x0)) #the Hough center uses coordinates inside the cropped region
    local_y = int(round(yc - y0))
    center_box = (
        local_x - PUPIL_R_SEARCH, local_x + PUPIL_R_SEARCH,
        local_y - PUPIL_R_SEARCH, local_y + PUPIL_R_SEARCH,
    )
    x, y, radius = hough_best_circle(edges, radii, center_box)
    return x + x0, y + y0, radius


def detect_iris_circle(img, pupil):
    #Step 3: search for the outer boundary near the pupil center
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


if __name__ == "__main__":
    image_path = "001_1_1.bmp"
    if len(sys.argv) > 1:
        image_path = sys.argv[1]

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    pupil_circle, iris_circle = IrisLocalization(image)
    print("Pupil (x, y, radius):", pupil_circle)
    print("Iris (x, y, radius):", iris_circle)
