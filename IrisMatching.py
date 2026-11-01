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

# Fisher LDA：higher dimension to lower dimension
def fisher_lda(train_features, train_labels, test_features,n_components=None):
    """
    Reduce feature dimensions using Fisher LDA.

    train_features: original training feature matrix.
    train_labels: training class IDs.
    test_features: original test feature matrix.
    n_components: number of output dimensions; None uses
                  the maximum available LDA dimensions.
    lda: fitted LinearDiscriminantAnalysis model.
    train_reduced: transformed training features.
    test_reduced: transformed testing features.

    LDA learns a discriminative projection using training
    data and labels. The same projection is applied to
    test data without refitting, avoiding test-data leakage.
    The SVD solver is used.
    """
    lda = LinearDiscriminantAnalysis(n_components=n_components,solver="svd")
    train_reduced = lda.fit_transform(train_features, train_labels)
    test_reduced = lda.transform(test_features)
    return train_reduced, test_reduced

# combing matching functions
def iris_matching(train_features, train_labels, test_features):
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
    # 1: Fisher LDA
    train_reduced, test_reduced = fisher_lda(
        train_features, train_labels, test_features)

    # 2: calculate class centers
    classes, centers = calculate_centers(
        train_reduced, train_labels)

    # 3: matching using three distance methods
    predictions = {}
    distances = {}
    for method in ["L1", "L2", "Cosine"]:
        distance_matrix = calculate_distance_matrix(
            test_reduced, centers, method)
        predicted_labels = []

        for row in distance_matrix:
            nearest_index = np.argmin(row)
            predicted_labels.append(classes[nearest_index])

        predictions[method] = np.array(predicted_labels)
        distances[method] = distance_matrix
    return predictions, distances

# rotation task
def iris_matching_rotation(train_features, train_labels,rotated_test_features):
    """
    Match iris images using seven-angle rotation compensation.

    train_features: original training feature matrix.
    train_labels: training class IDs.
    rotated_test_features: dictionary of test features by angle.
    lda: LDA model fitted on training features only.
    train_reduced: LDA-transformed training features.
    classes, centers: class IDs and their feature centers.
    method: L1, L2, or Cosine.
    angle: one of -9, -6, -3, 0, 3, 6, or 9 degrees.
    test_reduced: transformed test features at one angle.
    distances: distance matrix for the current angle.
    best_distances: minimum distances across all angles.
    predictions: predicted classes from minimum distances.
    results: predicted labels for each distance method.
    distance_matrices: final minimum-distance matrices.

    LDA is fitted once using training data.
    The outer loop evaluates each distance method.
    The inner loop transforms features at seven angles
    using the same LDA model and calculates distances.
    np.minimum keeps the smallest distance for each
    test-image/class pair across the seven angles.
    The nearest class center determines the final prediction.
    """
    # Fit LDA using training features only
    lda = LinearDiscriminantAnalysis(solver="svd")
    train_reduced = lda.fit_transform(train_features, train_labels)

    # Calculate class centers
    classes, centers = calculate_centers(train_reduced, train_labels)
    results = {}
    distance_matrices = {}

    for method in ["L1", "L2", "Cosine"]:
        best_distances = None
        for angle in [-9, -6, -3, 0, 3, 6, 9]:
            # Transform rotated features using the same LDA
            test_reduced = lda.transform(rotated_test_features[angle])
            distances = calculate_distance_matrix(test_reduced,centers,method)
           
            if best_distances is None:
                best_distances = distances
            else:
                best_distances = np.minimum(best_distances, distances)
        predictions = classes[np.argmin(best_distances, axis=1)]
        results[method] = predictions
        distance_matrices[method] = best_distances
    return results, distance_matrices

# comparison
def compare_features(train_features, train_labels, test_features, 
                     test_labels,n_components=None):
    """
    Compare nearest-center recognition with and without LDA.

    train_features, test_features: original feature matrices.
    train_labels, test_labels: true iris class IDs.
    n_components: optional number of LDA dimensions.
    classes, centers: class IDs and class mean vectors.
    original_results: CRR using original feature vectors.
    train_reduced, test_reduced: LDA-transformed features.
    lda_results: CRR using reduced feature vectors.
    distances: test-to-center distance matrix.
    predictions: nearest-center class predictions.

    The first loop evaluates L1, L2, and Cosine using
    original features. The second loop repeats the
    evaluation after Fisher LDA dimensionality reduction.
    """
    # Original features
    classes, centers = calculate_centers(train_features, train_labels)
    original_results = {}

    for method in ["L1", "L2", "Cosine"]:
        distances = calculate_distance_matrix(test_features, centers, method)
        predictions = classes[np.argmin(distances, axis=1)]
        original_results[method] = (np.mean(predictions == test_labels) * 100)

    # LDA features
    train_reduced, test_reduced = fisher_lda(train_features, train_labels, 
                                             test_features,n_components=n_components)
    classes, centers = calculate_centers(train_reduced, train_labels)
    lda_results = {}

    for method in ["L1", "L2", "Cosine"]:
        distances = calculate_distance_matrix(test_reduced, centers, method)
        predictions = classes[np.argmin(distances, axis=1)]
        lda_results[method] = (np.mean(predictions == test_labels) * 100)
    return original_results, lda_results



    

   



