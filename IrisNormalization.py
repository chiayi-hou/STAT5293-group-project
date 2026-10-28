# Step 2 (Part 3.2.2 in Paper): Iris Normalization

import numpy as np
import cv2

M_ROWS = 64
N_COLS = 512

def IrisNormalization(img, pupil, iris, M=M_ROWS, N=N_COLS, init_angle_deg=0.0):

    xp, yp, rp = pupil
    xi, yi, ri = iris
    
    X, Y = np.meshgrid(np.arange(N), np.arange(M)) #X=angle index, Y=radial index

    theta = 2.0 * np.pi * X / N + np.deg2rad(init_angle_deg) #theta = 2*pi*X/N, shifted by the initial angle

    #Boundary points in direction theta
    x_p = xp + rp * np.cos(theta)
    y_p = yp - rp * np.sin(theta)
    x_i = xi + ri * np.cos(theta)
    y_i = yi - ri * np.sin(theta)

    #Eq.2
    x = x_p + (x_i - x_p) * Y / M
    y = y_p + (y_i - y_p) * Y / M

    #bilinear interpolation
    norm = cv2.remap(img, x.astype(np.float32), y.astype(np.float32), interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    return norm
