import numpy as np
import matplotlib.pyplot as plt

"""
PerformanceEvaluation.py - Iris recognition evaluation.

This script evaluates two tasks:
1. Identification: Correct Recognition Rate (CRR).
2. Verification: False Match Rate (FMR) and
   False Non-Match Rate (FNMR).

It generates a verification table and plots FMR against
FNMR for L1, L2, and Cosine distance measures.
"""

# CRR：identification 
def calculate_crr(true_labels, predicted_labels):
    """
    Calculate Correct Recognition Rate (CRR).

    true_labels: actual iris class IDs.
    predicted_labels: predicted iris class IDs.
    correct: number of correctly classified test images.
    total: total number of test images.
    crr: percentage of correctly recognized images.

    CRR = correct / total * 100.
    """
    correct = np.sum(true_labels == predicted_labels)
    total = len(true_labels)
    crr = correct / total * 100
    return crr

def evaluate_identification(true_labels, predictions):
    """
    Evaluate identification accuracy for three distance methods.

    true_labels: actual test class IDs.
    predictions: dictionary of predicted labels by method.
    method: L1, L2, or Cosine.
    predicted_labels: predictions for the current method.
    crr: correct recognition rate for that method.
    results: dictionary storing all three CRR values.

    The loop calculates and prints CRR for each method.
    """
    results = {}
    for method in ["L1", "L2", "Cosine"]:
        predicted_labels = predictions[method]
        crr = calculate_crr(
            true_labels, predicted_labels)
        results[method] = crr
        print(method, "CRR:", round(crr, 2), "%")
    return results

# verification process
def calculate_fmr_fnmr(distances, same_class, threshold):
    """
    Calculate verification error rates at a given threshold.

    distances: distances for genuine and impostor comparisons.
    same_class: Boolean array; True means the classes match.
    threshold: maximum distance allowed for acceptance.
    accepted: Boolean array indicating accepted comparisons.
    TP: genuine comparisons correctly accepted.
    FP: impostor comparisons incorrectly accepted.
    TN: impostor comparisons correctly rejected.
    FN: genuine comparisons incorrectly rejected.
    fmr: FP / (FP + TN).
    fnmr: FN / (FN + TP).

    A comparison is accepted when distance <= threshold.
    The function counts the four outcomes and calculates
    the two error rates, avoiding division by zero.
    """
    accepted = (distances <= threshold) # accept if distance is small enough

    # count four possible outcomes
    TP = np.sum(accepted & same_class)
    FP = np.sum(accepted & ~same_class)
    TN = np.sum(~accepted & ~same_class)
    FN = np.sum(~accepted & same_class)

    # calculate error rates
    if FP + TN > 0: # FMR
        fmr = FP / (FP + TN)
    else:
        fmr = 0

    if FN + TP > 0: # FNMR
        fnmr = FN / (FN + TP)
    else:
        fnmr = 0

    return fmr, fnmr

# ROC 
def calculate_roc(distances, same_class):
    """
    Calculate FMR and FNMR over different thresholds.

    distances: all verification distances.
    same_class: Boolean labels for genuine comparisons.
    thresholds: sorted unique distances plus two endpoints.
    threshold: current acceptance threshold.
    fmr_list: FMR values at all thresholds.
    fnmr_list: FNMR values at all thresholds.

    The loop changes the acceptance threshold and calculates
    FMR and FNMR at each value. These values are used to
    plot the verification error trade-off curve.
    """
    thresholds = np.concatenate((
        [np.min(distances) - 1],
        np.unique(distances),
        [np.max(distances) + 1]))

    fmr_list = []
    fnmr_list = []
    for threshold in thresholds:
        fmr, fnmr = calculate_fmr_fnmr(
            distances, same_class, threshold)
        fmr_list.append(fmr)
        fnmr_list.append(fnmr)
    return fmr_list, fnmr_list

def prepare_verification_data(distance_matrix, test_labels, classes):
    """
    Prepare genuine and impostor comparisons for verification.

    distance_matrix: test-to-class-center distance matrix.
    test_labels: true class IDs of test samples.
    classes: class IDs corresponding to matrix columns.
    i: index of the current test sample.
    j: index of the current class center.
    distances: flattened list of all comparison distances.
    same_class: Boolean list identifying genuine comparisons.

    The outer loop visits each test image.
    The inner loop compares it with every class center.
    A comparison is genuine when the test class matches
    the center class; otherwise, it is an impostor comparison.
    """
    distances = []
    same_class = []
    for i in range(len(test_labels)):
        for j in range(len(classes)):
            distances.append(distance_matrix[i, j])
            same_class.append(test_labels[i] == classes[j])
    return np.array(distances), np.array(same_class)

def print_verification_table(fmr, fnmr, method):
    """
    Print FNMR at selected target FMR levels.

    fmr: array of calculated False Match Rates.
    fnmr: array of calculated False Non-Match Rates.
    method: selected distance measure.
    target_fmr: target FMR values of 0.1%, 1%, 5%, and 10%.
    target: current target FMR.
    index: index of the observed FMR closest to the target.

    The loop finds the closest available FMR to each target
    and prints the corresponding FMR and FNMR percentages.
    The reported FMR may differ slightly from its target.
    """
    # conver lists to arrays
    fmr = np.array(fmr)
    fnmr = np.array(fnmr)
    target_fmr = [0.001, 0.01, 0.05, 0.10]
    print("\nVerification Results:", method)
    print("FMR (%)     FNMR (%)")
    for target in target_fmr:
        index = np.argmin(np.abs(fmr - target))
        print(f"{fmr[index] * 100:.3f}"
              f"       {fnmr[index] * 100:.3f}")

def evaluate_verification(distance_matrices, test_labels, classes):
    """
    Evaluate verification performance and save the ROC plot.

    distance_matrices: dictionary of distance matrices.
    test_labels: true class IDs of test images.
    classes: class IDs corresponding to class centers.
    method: L1, L2, or Cosine.
    distance_matrix: distances for the current method.
    distances: flattened verification distances.
    same_class: Boolean labels for genuine comparisons.
    fmr_list: calculated FMR values across thresholds.
    fnmr_list: calculated FNMR values across thresholds.

    The loop evaluates each distance method separately.
    It prepares genuine/impostor comparisons, calculates
    error rates across thresholds, prints a verification
    table, and plots the resulting curve.
    The figure is saved as ROC_Curves.png.
    """
    plt.figure(figsize=(7, 5))
    for method in ["L1", "L2", "Cosine"]:
        distance_matrix = distance_matrices[method]
        distances, same_class = prepare_verification_data(
            distance_matrix, test_labels, classes)
        fmr_list, fnmr_list = calculate_roc(distances, same_class)
        print_verification_table(fmr_list, fnmr_list, method)
        plt.plot(fmr_list, fnmr_list, label=method)
    plt.xlabel("False Match Rate (FMR)")
    plt.ylabel("False Non-Match Rate (FNMR)")
    plt.title("ROC Curves - Iris Verification")
    plt.legend()
    plt.grid(True)
    plt.savefig("ROC_Curves.png", dpi=300, bbox_inches="tight")
    plt.close()


