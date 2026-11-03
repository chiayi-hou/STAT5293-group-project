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
    Fit Fisher LDA model with given training features and labels. Transform the original training features into lower dimensional features (n_components).

    Args:
        train_features_dict (dict): Dictionary containing the extracted training features.
                                    Keys are the 7 rotation angles (-9, -6, -3, 0, 3, 6, 9), and values are np.ndarray with shape (324, 1536) for each angle.
        train_labels (np.array): true labels for the training images
        n_components (int): Number of LDA dimensions to retain. Default is None; in this case, the reduced dimension would be 107.

    Returns:
        lda (LinearDiscriminantAnalysis): Fitted Fisher LDA model
        dict_train_reduced (dict): Dictionary containing the LDA-transformed training features.
                                    Keys are the 7 rotation angles, and values are np.ndarray with shape (324, n) for each angle.
    """
    angles = [-9, -6, -3, 0, 3, 6, 9]

    # concatenate features from all angles into one big array (to fit the LDA)
    X_train = np.concatenate([train_features_dict[angle] for angle in angles], axis=0)     
    # Repeat labels for all angles
    y_train = np.tile(train_labels, len(angles))

    # create a LDA model
    lda = LinearDiscriminantAnalysis(n_components=n_components, solver="svd")

    # fit the lda model with training data
    # at the same time, transform the X_train features to lower dimension (X_train_reduced)
    X_train_reduced = lda.fit_transform(X_train, y_train)

    # split the transformed big array X_train_reduced back to 7 angles
    # each containing 324 training images with n LDA components.
    reduced_parts = np.split(X_train_reduced, len(angles), axis=0)
    # reorganize back to the original format
    dict_train_reduced = {angle: features for angle, features in zip(angles, reduced_parts)}

    return lda, dict_train_reduced

# one of the main functions to be imported to IrisRecognition.py
def iris_matching(dict_train, train_labels, test_features):
    """
    Perform iris matching using three matching methods (L1, L2, and Cosine).
    For each class, calculate 7 centers from the training features at different rotation angles. Compare each testing image with the centers and select
    the smallest distance among the 7 angles as the final distance for each class.

    Args:
        dict_train (dict):  dictionary containing training features.
                            Keys are the 7 rotation angles, and values are np.ndarray with shape (324, n).
                            n can be 1536 for original features or the number of dimensions retained after Fisher LDA.
        train_labels (np.ndarray): true labels for the 324 training images.
        test_features (np.ndarray): testing features with shape (432, n). Must have the same feature dimensions as the training features.
   
    Returns:
        predictions (dict): dictionary that contains predicted class labels for L1, L2, and Cosine.
                            Each value is an np.ndarray with shape (432,).
        distances (dict): dictionary that contains the representing matching distances between testing images and the 108 iris classes.
                          Keys are L1, L2, and Cosine, and each value is an np.ndarray with shape (432, 108).
    """

    # Ma et al. suggests to create 7 templates for each class, so we will find centers among 3 training features for each angle
    # rotation_centers would be used to save the angles and their corresponding centers 
    rotation_centers = {}

    # loop through each angle and its corresponding feature array to calculate centers
    for angle, features in dict_train.items():
        # classes is an array containing the 108 iris class IDs
        # centers is an array of the center for each class for the current angle
        classes, centers = calculate_centers(features, train_labels)
        # save the centers to the dictionary
        rotation_centers[angle] = centers

    # empty dictionaries to save predictions and distances between testing and training centers
    # keys are the name of the matching methods ("L1", "L2", "Cosine")
    predictions = {}
    distances = {}
    # loop through each matching method and do iris matching
    for method in ["L1", "L2", "Cosine"]:
        # Ma et al. selected the smallest distance among the 7 angle centers as the representation
        # keep track of the best distance as representation. The best distance would be the smallest distance among the 7 angles
        best_distances = None

        # loop through each angle and compare distance
        for angle, centers in rotation_centers.items():
            # calculate distance matrix
            distance_matrix = calculate_distance_matrix(test_features, centers, method)

            # save the value with the smallest distance 
            # reference: "the minimum of the seven scores is taken as the final matching distance"
            if best_distances is None:
                # initialize the best_distances for the first round
                best_distances = distance_matrix
            else:
                # update elements that have a smaller value accross rotation angles to best_distances
                best_distances = np.minimum(best_distances, distance_matrix)

        # select the class with the smallest matching distance for each testing image.
        # use smallest_idx to select the class as predicted_labels
        smallest_idx = np.argmin(best_distances, axis=1)
        predicted_labels = classes[smallest_idx]

        # save the predictions and the matching distance matrices to the dictionaries
        predictions[method] = predicted_labels
        distances[method] = best_distances

    return predictions, distances

