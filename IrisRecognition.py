import os
import numpy as np
from IrisLocalization import IrisLocalization
from IrisNormalization import IrisNormalization
from ImageEnhancement import image_enhancement
from FeatureExtraction import feature_extraction
from IrisMatching import iris_matching
from PerformanceEvaluation import (evaluate_identification,evaluate_verification)

# read all iris images
def load_dataset(data_path):
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

# 待处理！！
def run_recognition(train_features, train_labels,test_features, test_labels):
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


if __name__ == "__main__":
    data_path = "CASIA Iris Image Database (version 1.0)" # dataset folder
    train_files, train_labels, test_files, test_labels = (
        load_dataset(data_path))

    print("Training images:", len(train_files))
    print("Testing images:", len(test_files))
    print("Number of classes:", len(set(train_labels)))

    # image preprocessing and feature extraction (training set)
    train_features = []
    for img in train_files:
        # Image preprocessing 
        pupil, iris = IrisLocalization(img)
        normalized_img = IrisNormalization(img, pupil, iris)
        
        # image enhancement
        enhanced_img = image_enhancement(normalized_img)

        # feature extraction 
        feature = feature_extraction(enhanced_img, f=0.05)
        train_features.append(feature)

    # image preprocessing and feature extraction (test set)
    test_features = []
    for img in test_files:
        # Image preprocessing
        pupil, iris = IrisLocalization(img)
        normalized_img = IrisNormalization(img, pupil, iris)
        
        # image enhancement
        enhanced_img = image_enhancement(normalized_img)

        # feature extraction 
        feature = feature_extraction(enhanced_img, f=0.05)
        test_features.append(feature)
    
    # Iris matching and evaluation 
    # run_recognition(
    #     train_features, train_labels,
    #     test_features, test_labels)







