import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

"""
IrisMatching.py - Iris feature matching.

This script uses Fisher Linear Discriminant Analysis (LDA)
for dimensionality reduction and a nearest-center classifier
for iris identification.

Three distance measures are used: L1, squared L2, and
Cosine distance. Rotation matching selects the minimum
distance across seven normalization angles.
"""

# calculate the centers for each class
def calculate_centers(features, labels):
    """
    Calculate the mean feature vector for each iris class.

    features: training feature matrix.
    labels: class ID for each training sample.
    classes: sorted unique class IDs.
    class_features: feature vectors belonging to class c.
    center: mean feature vector of the current class.
    centers: matrix containing all class centers.

    The loop selects training samples from each class
    and calculates their mean along axis 0.
    """
    classes = np.unique(labels)
    centers = []
    
    for c in classes:
        class_features = features[labels == c]
        center = np.mean(class_features, axis=0)
        centers.append(center)
    return classes, np.array(centers)

# calculate the distance of the sample through different methods
def calculate_distance(x, center, method):
    """
    Calculate the distance between a sample and a class center.

    x: feature vector of one test sample.
    center: mean feature vector of one class.
    method: distance measure, L1, L2, or Cosine.
    numerator: dot product used for Cosine similarity.
    denominator: product of the two vector norms.
    distance: calculated distance; smaller means more similar.

    L1 sums absolute differences.
    L2 sums squared differences (squared Euclidean distance).
    Cosine distance is 1 minus Cosine similarity.
    A zero denominator is handled to avoid division by zero.
    """
    if method == "L1":
        distance = np.sum(np.abs(x - center)) # sum of absolute difference
    elif method == "L2":
        distance = np.sum((x - center) ** 2) # sum of squared difference
    elif method == "Cosine":
        numerator = np.dot(x, center)
        denominator = np.linalg.norm(x) * np.linalg.norm(center) 
        if denominator == 0:
            distance = 1.0
        else:
            distance = 1 - numerator / denominator # 1 - cosine similarity 
    else:
        raise ValueError("Unknown distance method")
    return distance

# find the closest center with smaller distance
def calculate_distance_matrix(test_features, centers, method):
    """
    Calculate distances between all test samples and class centers.

    test_features: matrix of test feature vectors.
    centers: matrix of class center vectors.
    method: selected distance measure.
    x: current test feature vector.
    center: current class center.
    row: distances from one test sample to all centers.
    distances: complete test-to-center distance matrix.

    The outer loop visits each test sample.
    The inner loop compares it with every class center.
    Each row contains distances to all classes.
    """
    distances = []
    for x in test_features:
        row = []
        for center in centers:
            distance = calculate_distance(x, center, method)
            row.append(distance)
        distances.append(row)
    return np.array(distances)

# one of the main functions to be imported to IrisRecognition.py
def fit_fisher_lda(train_features_dict, train_labels, n_components=None):
    """
    
    """
    angles = [-9, -6, -3, 0, 3, 6, 9]

    # Combine features from all angles
    X_train = np.concatenate([train_features_dict[angle] for angle in angles], axis=0)
    # Repeat labels for all angles
    y_train = np.tile(train_labels, len(angles))

    # Fit one LDA model
    lda = LinearDiscriminantAnalysis(
        n_components=n_components,
        solver="svd"
    )

    X_train_reduced = lda.fit_transform(X_train, y_train)

    # split back to 7 anlges
    reduced_parts = np.split(X_train_reduced, len(angles), axis=0)
    dict_train_reduced = {angle: features for angle, features in zip(angles, reduced_parts)}

    return lda, dict_train_reduced

# combing matching functions
def iris_matching(dict_train, train_labels, test_features):
    """
    Perform iris identification using Fisher LDA and class centers.

    train_features, test_features: original feature matrices.
    train_labels: known training class IDs.
    train_reduced, test_reduced: LDA-transformed features.
    classes, centers: iris classes and their mean vectors.
    predictions: dictionary of predicted labels by method.
    distances: dictionary of distance matrices by method.
    distance_matrix: distances from test samples to centers.
    row: distances for one test sample.
    nearest_index: index of the smallest distance.
    predicted_labels: predicted classes for one method.

    First, LDA reduces feature dimensions using training data.
    Next, class centers are calculated.
    The outer loop evaluates the three distance methods.
    The inner loop assigns each test sample to its closest
    class center using the minimum distance.
    """

    # calculate class centers
    #classes, centers = calculate_centers(dict_train_reduced, train_labels)
    rotation_centers = {}

    for angle, features in dict_train.items():
        classes, centers = calculate_centers(features, train_labels)
        rotation_centers[angle] = centers

    predictions = {}
    distances = {}
    for method in ["L1", "L2", "Cosine"]:
        best_distances = None

        # Compare against centers of all seven angles
        for angle, centers in rotation_centers.items():
            distance_matrix = calculate_distance_matrix(test_features, centers, method)

            # save the value with the smallest distance 
            # reference: "the minimum of the seven scores is taken as the final matching distance"
            if best_distances is None:
                best_distances = distance_matrix
            else:
                best_distances = np.minimum(best_distances, distance_matrix)

        # Find the nearest class
        smallest_idx = np.argmin(best_distances, axis=1)
        predicted_labels = classes[smallest_idx]

        predictions[method] = predicted_labels
        distances[method] = best_distances

    return predictions, distances

