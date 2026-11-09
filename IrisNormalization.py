# Step 2 (Part 3.2.2 in Paper): Iris Normalization

import numpy as np
import cv2


M_ROWS = 64 #number of rows of the normalized image = samples from the pupil edge (row 0) out to the iris edge
N_COLS = 512 #number of columns = samples around the full circle (one column per angle)

def IrisNormalization(img, pupil, iris, M=M_ROWS, N=N_COLS, init_angle_deg=0.0):
    #img: the original gray eye image
    #pupil: (xp, yp, rp), the pupil circle found by IrisLocalization
    #iris: (xi, yi, ri), the outer iris circle found by IrisLocalization
    #M, N: output size (64 rows, 512 columns by default)
    #init_angle_deg: where the unwrapping starts, in degrees. Shifting the start angle is the same as rotating the eye

    #xp, yp, rp: pupil center (column, row) and radius; xi, yi, ri: the same for the outer iris circle
    xp, yp, rp = pupil
    xi, yi, ri = iris
    
    #X, Y: M x N grids holding the column number (0..N-1) and the row number (0..M-1) of every output pixel
    X, Y = np.meshgrid(np.arange(N), np.arange(M)) #X=angle index, Y=radial index

    #theta: the angle (in radians) that each column represents; the N columns cover one full circle
    theta = 2.0 * np.pi * X / N + np.deg2rad(init_angle_deg) #theta = 2*pi*X/N, shifted by the initial angle

    #Boundary points in direction theta
    #(x_p, y_p): point on the pupil circle; (x_i, y_i): point on the outer iris circle
    #"minus sin" because image rows grow downward, so increasing theta goes counterclockwise on screen
    x_p = xp + rp * np.cos(theta)
    y_p = yp - rp * np.sin(theta)
    x_i = xi + ri * np.cos(theta)
    y_i = yi - ri * np.sin(theta)

    #Eq.2
    #(x, y): position in the original image to sample for each output pixel;
    #Y / M is the fraction of the way from the pupil point (0) toward the iris point (close to 1)
    x = x_p + (x_i - x_p) * Y / M
    y = y_p + (y_i - y_p) * Y / M

    #bilinear interpolation
    #norm: the normalized M x N iris image; positions outside the original image are filled with 0 (black)
    norm = cv2.remap(img, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return norm
