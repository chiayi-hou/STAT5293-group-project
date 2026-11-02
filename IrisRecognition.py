import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from IrisLocalization import (IrisLocalization,ImageQualityAssessment,is_qualified)
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

# read all iris images
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

# assess image quality
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

def get_features(img_paths, is_training, f=0.05):
    if is_training:
        angles = [-9, -6, -3, 0, 3, 6, 9]
    else:
        angles = [0]

    all_features = {angle: [] for angle in angles}
    for file_path in tqdm(img_paths, desc="Extracting training features" if is_training else "Extracting testing features"):
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        pupil, iris = IrisLocalization(img)

        for angle in angles:
            normalized_img = IrisNormalization(img, pupil, iris, init_angle_deg=angle)
            enhanced_img = image_enhancement(normalized_img)
            feature = feature_extraction(enhanced_img, f)

            all_features[angle].append(feature)

    # Convert lists into NumPy arrays
    for angle in angles:
        all_features[angle] = np.array(all_features[angle])

    return all_features

# Main execution:
# This block runs only when IrisRecognition.py is executed directly.
# It loads the dataset, checks image quality, extracts features,
# performs baseline and rotation matching, and generates results.
#
# The dataset contains 108 iris classes with 3 training images
# and 4 testing images per class.
#
# Quality thresholds are fitted using training descriptors only.
# The quality assessment reports qualified image counts but
# does not remove images from feature extraction or matching.
#
# The rotation experiment compares seven normalization angles.
# The baseline uses the original 0-degree features.
#
# Finally, the script compares original and LDA features
# and plots recognition accuracy against LDA dimensions.
if __name__ == "__main__":
    data_path = "CASIA Iris Image Database (version 1.0)" # dataset folder

    # load dataset
    train_files, train_labels, test_files, test_labels = (load_dataset(data_path))
    print("Training images:", len(train_files))
    print("Testing images:", len(test_files))
    print("Number of classes:", len(set(train_labels)))

    # 2. image quality assessment 
    #print("\nAssessing image quality...")
    #train_descriptors = assess_image_quality(train_files)
    #test_descriptors = assess_image_quality(test_files)

    #thresholds = fit_thresholds(train_descriptors)
    #print("Quality thresholds:", thresholds)

    #train_qualified = count_qualified_images(train_descriptors, thresholds)
    #test_qualified = count_qualified_images(test_descriptors, thresholds)
    #print("Qualified training images:", train_qualified)
    #print("Qualified testing images:", test_qualified)

    # Image preprocessing and feature extraction 
    # {angle: np.array,}
    print("\nExtracting features...")
    train_features = get_features(train_files, is_training=True)
    test_features = get_features(test_files, is_training=False)

    # Fisher LDA
    print("\n Fitting Fisher LDA")
    lda_matrix, dict_train_reduced = fit_fisher_lda(train_features, train_labels)
    test_reduced = lda_matrix.transform(test_features[0])

    # Create Fig. 10: test out different lda dimension
    print("\n Creating Fig. 10 (try out different LDA dimension)")
    dimensions = [10, 20, 40, 60, 80, 90, 100, 107]
    methods = ["L1", "L2", "Cosine"]
    results = {"L1": [], "L2": [], "Cosine": []}

    # save matching results for final evaluation
    final_predictions, final_distances = test_dimensionality(dict_train_reduced, train_labels, test_reduced, test_labels, selected_dimension=100)
  
    # Create Table 3: compare original result and lda result
    print("\n Creating Table 3: compare original result and LDA result")
    original_predictions, _ = iris_matching(train_features, train_labels, test_features[0])
    create_table_3(final_predictions, original_predictions, test_labels)

    # Verification
    classes = np.unique(train_labels)
    print("\nVerification Results:")
    evaluate_verification(final_distances, test_labels, classes)
    







