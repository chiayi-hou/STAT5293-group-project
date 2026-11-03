import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from IrisLocalization import (IrisLocalization,ImageQualityAssessment,is_qualified, fit_thresholds)
from IrisNormalization import IrisNormalization
from ImageEnhancement import image_enhancement
from FeatureExtraction import feature_extraction
from IrisMatching import (iris_matching, fit_fisher_lda)
from PerformanceEvaluation import (evaluate_verification, test_dimensionality, create_table_3)

"""
IrisRecognition.py - Main program for iris recognition.

This script loads CASIA-IrisV1 images and separates them into
training and testing sets by session. It assesses image quality,
extracts 1536-dimensional iris features, performs iris matching,
and evaluates recognition and verification performance.

It also tests seven rotation angles and compares recognition
performance with different LDA dimensions.
"""

# helper function to read all iris images
def load_dataset(data_path):
    """
    Load BMP images and split them by acquisition session.

    data_path: folder containing the CASIA iris dataset.
    train_files, test_files: lists of image file paths.
    train_labels, test_labels: arrays of iris class IDs.

    os.walk searches all subfolders. The inner loop checks
    each filename and skips non-BMP files.
    The filename is split to obtain the class ID and session ID.
    Session 1 images are used for training, and Session 2
    images are used for testing.
    """
    train_files = []
    train_labels = []
    test_files = []
    test_labels = []

    for root, dirs, files in os.walk(data_path):
        for filename in sorted(files):
            if not filename.lower().endswith(".bmp"):
                continue

            parts = filename[:-4].split("_")

            if len(parts) != 3:
                continue

            class_id = int(parts[0])
            session_id = int(parts[1])
            file_path = os.path.join(root, filename)

            if session_id == 1:
                train_files.append(file_path)
                train_labels.append(class_id)
            elif session_id == 2:
                test_files.append(file_path)
                test_labels.append(class_id)
    return (train_files,
        np.array(train_labels),
        test_files,
        np.array(test_labels))

# helper function to assess image quality
def assess_image_quality(img_files):
    """
    Calculate image quality descriptors for each iris image.

    img_files: list of image paths.
    img: grayscale iris image.
    D: quality descriptor returned by ImageQualityAssessment.
    pupil_circle: detected pupil information, not used here.
    descriptors: array of quality descriptors for all images.

    The loop reads each image, checks that it is valid,
    calculates its quality descriptor, and stores the result.
    """
    descriptors = []
    for file_path in img_files:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if img is None:
            raise ValueError("Cannot read image: " + file_path)

        D, pupil_circle = ImageQualityAssessment(img)
        descriptors.append(D)
    return np.array(descriptors)

def count_qualified_images(descriptors, thresholds):
    """
    Count how many images satisfy the quality thresholds.

    descriptors: quality descriptors for all images.
    thresholds: quality limits estimated from training images.
    D: descriptor of the current image.
    qualified: whether the image passes the quality check.
    reason: explanation returned by the quality check.
    count: number of images that pass.

    The loop checks every descriptor and increases count
    when an image passes. No images are removed.
    """
    count = 0
    for D in descriptors:
        qualified, reason = is_qualified(D, thresholds)
        if qualified:
            count += 1
    return count

# helper function to do image preprocessing and extract features
def get_features(img_paths, is_training, f=0.05):
    """
    Read in iris images with given image paths, do image preprocessing (iris localization, normalization, enhancement),
    and extract features for the images. If the given images are training images, it will create 7 different versions of 
    the same image with different rotation angles.

    Args:
        img_paths (np.array): an array of image paths to be used to extract features
        is_training (bool): whether this is training set or not
        f (float): cosine frequency passed to create_filter. A larger f captures high-frequency patterns better, and a lower f captures lower ones better.  
                   0.05 <= f <= 0.4 has been explored in previous Gabor-based texture analysis research.

    Returns:
        all_features (dict): dictionary containing extracted features. 
                             Keys are rotation angles. Values are np.ndarray with shape (# of images, 1536)
                             Training set would have 7 angles, and testing set would only have "angle 0"
    """
    # do 7 different rotation if this is a training set. If it is testing, then only use angle = 0 (no rotation)
    if is_training:
        # 7 rotation angles stated in Ma et al.
        angles = [-9, -6, -3, 0, 3, 6, 9]
    else:
        angles = [0]

    # create a dictionary to save all features
    # keys are the rotation angles, values would be a list of feature arrays
    all_features = {angle: [] for angle in angles}

    # loop through each image to extract features
    for file_path in tqdm(img_paths, desc="Extracting training features" if is_training else "Extracting testing features"):
        # read in the iris image
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        # iris localization to identify pupil and iris
        pupil, iris = IrisLocalization(img)

        # for each image, loop through every rotation angle that we need to try on
        for angle in angles:
            # iris normalization (rotate during this process)
            normalized_img = IrisNormalization(img, pupil, iris, init_angle_deg=angle)
            # image enhancement on the normalized image
            enhanced_img = image_enhancement(normalized_img)

            # extract features from the enhanced image with given f
            feature = feature_extraction(enhanced_img, f)

            # save the feature array to the dictionary for the corresponding angle 
            all_features[angle].append(feature)

    # loop through each items in the dictionary to convert the list of arrays into 2D np.arrays
    for angle in angles:
        all_features[angle] = np.array(all_features[angle])

    return all_features

"""
Main execution for the Iris Recognition program which loads the dataset, checks image quality, extracts features, performs 
matching using both original and dimensinoally reduced features, and generates results.

Dataset: contains 108 iris classes with 3 training images and 4 testing images per class.
"""
if __name__ == "__main__":
    
    data_path = "CASIA Iris Image Database (version 1.0)" # dataset folder

    # load classes and file paths for each image
    train_files, train_labels, test_files, test_labels = (load_dataset(data_path))
    print("Training images:", len(train_files))
    print("Testing images:", len(test_files))
    print("Number of classes:", len(set(train_labels)))

    # Image quality assessment 
    print("\nAssessing image quality...")
    train_descriptors = assess_image_quality(train_files)
    test_descriptors = assess_image_quality(test_files)

    # Quality thresholds are fitted using training descriptors only.
    # The quality assessment reports qualified image counts but does not remove images from feature extraction or matching.
    thresholds = fit_thresholds(train_descriptors)
    print("Quality thresholds:", thresholds)

    train_qualified = count_qualified_images(train_descriptors, thresholds)
    test_qualified = count_qualified_images(test_descriptors, thresholds)
    print("Qualified training images:", train_qualified)
    print("Qualified testing images:", test_qualified)

    # Image preprocessing and feature extraction 
    print("\nExtracting features...")
    # train_features and test_features are a dictionary of np.ndarrays where keys are the angles and the values are the features for images {angle: np.ndarray,}
    # f is not specified in the paper. We tried out f=0.05 in the very beginning, and it works and results in accuracy greater than 80%
    train_features = get_features(train_files, is_training=True, f=0.05)
    test_features = get_features(test_files, is_training=False, f=0.05)

    # Fit Fisher LDA for the largest possible dimension (in this case 107 because we have 108 classes)
    print("\n Fitting Fisher LDA")
    # lda_matrix is the fitted Fisher LDA model that will be used to transform original features into a lower dimensional space
    # dict_train_reduced = a dictionary that store reduced features for training data. Keys are the 7 angles for image rotation. 
    #                      values are the corresponding np.ndarray for reduced features (in 107 dimension) for 108*3=324 images in total (shape=324*107)
    lda_matrix, dict_train_reduced = fit_fisher_lda(train_features, train_labels)
    # use the trained lda_matrix to reduce feature dimension for test images 
    test_reduced = lda_matrix.transform(test_features[0])   # np.ndarray (432*107)

    # Start iris matching, create visualizaitons, and do verification
    # Create Fig. 10: test out different lda dimension
    print("\n Creating Fig. 10 (try out different LDA dimension)")

    # test different lda reduction dimension, plot visualization, and save the selected result for later comparison
    # we selected 100 (among dimensions = [10, 20, 40, 60, 80, 90, 100, 107]) as the final dimension as the CRR for all three methods seem to converge around 100.
    # final_predictions and final_distances are dictionaries that contains predictions and distances between testing and training centers for L1, L2, and Cosine methods
    # using the selected 100-dimensional features.
    final_predictions, final_distances = test_dimensionality(dict_train_reduced, train_labels, test_reduced, test_labels, selected_dimension=100)
  
    # Create Table 3: compare original result and lda result
    print("\n Creating Table 3: compare original result and LDA result")
    # do iris matching on the original features (without reduction)
    # save only the predictions as original_predictions because we won't use the distances
    original_predictions, _ = iris_matching(train_features, train_labels, test_features[0])
    # create table to compare results between original features and the 100-dimensional features
    create_table_3(final_predictions, original_predictions, test_labels)

    # Verification (Create Table 4 and Fig. 11)
    classes = np.unique(train_labels)
    print("\nVerification Results:")
    evaluate_verification(final_distances, test_labels, classes)
    







