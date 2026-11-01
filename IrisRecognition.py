import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from IrisLocalization import (IrisLocalization,ImageQualityAssessment,fit_thresholds,is_qualified)
from IrisNormalization import IrisNormalization
from ImageEnhancement import image_enhancement
from FeatureExtraction import feature_extraction
from IrisMatching import (iris_matching,iris_matching_rotation,fisher_lda,
                          calculate_centers,calculate_distance_matrix,compare_features)
from PerformanceEvaluation import (evaluate_identification,evaluate_verification)

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

# get image features:已修改/加入angle参数-待确认！
def get_img_features(img_files, f=0.05,angle=0):
    """
    Extract a 1536-dimensional feature vector from each image.

    img_files: list of iris image paths.
    f: frequency parameter for feature extraction (default 0.05).
    angle: initial normalization angle in degrees (default 0).
    img: input grayscale image.
    pupil, iris: boundaries detected by IrisLocalization.
    normalized_img: iris image mapped to polar coordinates.
    enhanced_img: normalized image after enhancement.
    feature: extracted iris feature vector.
    img_features: feature vectors collected from all images.

    The loop processes images one by one through localization,
    normalization, enhancement, and feature extraction.
    The result is an array with one feature vector per image.
    """
    img_features = []
    for file_path in img_files:
        img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE) # Read image as grayscale

        if img is None:
            raise ValueError("Cannot read image: " + file_path)
        
        pupil, iris = IrisLocalization(img) # Iris Localization
        normalized_img = IrisNormalization(img, pupil, iris, init_angle_deg=angle) # Iris Normalization
        enhanced_img = image_enhancement(normalized_img) # Image Enhancement
        feature = feature_extraction(enhanced_img, f) # Feature Extraction
        img_features.append(feature)
    return np.array(img_features)

# rotation task
def get_rotated_features(img_files,zero_features):
    """
    Extract iris features at seven normalization angles.

    img_files: paths of the test images.
    zero_features: previously extracted features at 0 degrees.
    angles: seven rotation angles from -9 to 9 degrees.
    rotated_features: dictionary mapping each angle to
                      its corresponding feature matrix.

    The loop processes each angle. At 0 degrees, existing
    features are reused to avoid repeated computation.
    At other angles, features are extracted again using
    the specified normalization angle.
    """
    angles = [-9, -6, -3, 0, 3, 6, 9]
    rotated_features = {}
    for angle in angles:
        if angle == 0:
            rotated_features[angle] = zero_features
        else:
            rotated_features[angle] = get_img_features(img_files, angle=angle)
    return rotated_features

# Iris recognition and evaluation
def run_recognition(train_features, train_labels,test_features, test_labels):
    """
    Run iris matching and evaluate identification and verification.

    train_features, test_features: iris feature matrices.
    train_labels, test_labels: corresponding class ID arrays.
    predictions: predicted iris classes for each distance method.
    distances: test-to-class-center distance matrices.
    classes: unique iris class IDs in the training set.
    crr_results: identification accuracy for L1, L2, and Cosine.

    iris_matching performs LDA and nearest-center matching.
    Identification evaluates the predicted class labels.
    Verification evaluates matching distances using FMR/FNMR.
    """
    # Iris matching
    predictions, distances = iris_matching(
        train_features, train_labels, test_features)

    # identification evaluation
    print("\nIdentification Results:")
    crr_results = evaluate_identification(
        test_labels, predictions)

    # verification evaluation
    classes = np.unique(train_labels)
    print("\nVerification Results:")
    evaluate_verification(distances, test_labels, classes)
    return crr_results

# comparison
def compare_original_and_lda(train_features, train_labels,test_features, test_labels):
    """
    Compare recognition accuracy before and after Fisher LDA.

    train_features, test_features: original feature matrices.
    train_labels, test_labels: true iris class IDs.
    original: CRR results using original 1536-D features.
    lda: CRR results using LDA-reduced features.
    method: one of L1, L2, or Cosine.

    The loop prints the original and LDA recognition rates
    for each distance measure.
    """
    original, lda = compare_features(
        train_features, train_labels,test_features, test_labels)
    print("\nOriginal vs LDA Features:")
    print("Method     Original CRR    LDA CRR")
    for method in ["L1", "L2", "Cosine"]:
        print(f"{method:<10} {original[method]:.2f}%"
              f"          {lda[method]:.2f}%")

def compare_lda_dimensions(train_features, train_labels,test_features, test_labels):
    """
    Study how LDA dimensions affect iris recognition accuracy.

    dimensions: tested numbers of LDA components.
    n: current number of LDA dimensions.
    train_reduced, test_reduced: LDA-transformed features.
    classes: unique iris class IDs.
    centers: mean feature vector for each iris class.
    distances: distances from test samples to class centers.
    predictions: nearest-center class predictions.
    crr: correct recognition rate in percentage.
    results: CRR values for each method and dimension.

    The outer loop tests each LDA dimension. The inner loop
    evaluates L1, L2, and Cosine distances. Results are
    plotted and saved as LDA_Dimensions.png.

    With 108 classes, LDA supports at most 107 components.
    """
    dimensions = [10, 20, 40, 60, 80, 100, 107]
    results = {"L1": [], "L2": [], "Cosine": []}

    for n in dimensions:
        train_reduced, test_reduced = fisher_lda(
            train_features, train_labels,
            test_features, n_components=n)

        classes, centers = calculate_centers(train_reduced, train_labels)
        for method in ["L1", "L2", "Cosine"]:
            distances = calculate_distance_matrix(test_reduced, centers, method)
            predictions = classes[np.argmin(distances, axis=1)]
            crr = np.mean(predictions == test_labels) * 100
            results[method].append(crr)
        print("Completed dimension:", n)
    plt.figure(figsize=(8, 5))

    for method in ["L1", "L2", "Cosine"]:
        plt.plot(dimensions, results[method],marker="o", label=method)

    plt.xlabel("Number of LDA Dimensions")
    plt.ylabel("CRR (%)")
    plt.title("Effect of LDA Dimensions on Iris Recognition")
    plt.legend()
    plt.grid(True)
    plt.savefig("LDA_Dimensions.png",dpi=300, bbox_inches="tight")
    plt.close()


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

    # 1. load dataset
    train_files, train_labels, test_files, test_labels = (load_dataset(data_path))
    print("Training images:", len(train_files))
    print("Testing images:", len(test_files))
    print("Number of classes:", len(set(train_labels)))

    # 2. image quality assessment 
    print("\nAssessing image quality...")
    train_descriptors = assess_image_quality(train_files)
    test_descriptors = assess_image_quality(test_files)

    thresholds = fit_thresholds(train_descriptors)
    print("Quality thresholds:", thresholds)

    train_qualified = count_qualified_images(train_descriptors, thresholds)
    test_qualified = count_qualified_images(test_descriptors, thresholds)
    print("Qualified training images:", train_qualified)
    print("Qualified testing images:", test_qualified)

    # 3. image preprocessing and feature extraction 
    print("\nExtracting features...")
    train_features = get_img_features(train_files)
    test_features = get_img_features(test_files)
    print("Training feature shape:", train_features.shape)
    print("Testing feature shape:", test_features.shape)

    # seven-angle rotation compensation test
    print("\nExtracting rotated testing features...")
    rotated_test_features = get_rotated_features(test_files,test_features)
    print("Rotation angles:", list(rotated_test_features.keys()))
    print("0 degree shape:", rotated_test_features[0].shape)

    # check feature dimensions and values
    assert train_features.shape == (len(train_labels), 1536)
    assert test_features.shape == (len(test_labels), 1536)
    assert np.all(np.isfinite(train_features))
    assert np.all(np.isfinite(test_features))
    
    # 4. Iris matching and evaluation 
    run_recognition(train_features, train_labels,test_features, test_labels)

    # rotation compensation
    rotation_predictions, rotation_distances = iris_matching_rotation(train_features,
                                                                      np.array(train_labels),
                                                                      rotated_test_features)
    print("\nFull Dataset Rotation Matching Results:")
    for method in ["L1", "L2", "Cosine"]:
        crr = np.mean(
            rotation_predictions[method] == np.array(test_labels)) * 100
        print(method, "Rotation CRR:", round(crr, 2), "%")

    # 5. comparison
    compare_original_and_lda(train_features, train_labels,test_features, test_labels)

    # compare different LDA dimensions
    compare_lda_dimensions(train_features, train_labels,test_features, test_labels)








