import numpy as np


# Function to calculate the angular distance between two points in spherical coordinates
def angular_distance(theta1, phi1, theta2, phi2):
    # Convert degrees to radians
    #phi1, theta1, phi2, theta2 = map(np.radians, [phi1, theta1, phi2, theta2])

    # Apply the formula for angular distance
    return np.arccos(np.sin(phi1) * np.sin(phi2) + np.cos(phi1) * np.cos(phi2) * np.cos(theta1 - theta2))


def sensitivity(thetaphi):
    # Define target coordinates in (phi, theta)
    target_1 = np.radians((3, 9))  # (theta, phi) for label 0
    target_2 = np.radians((-3, -12))  # (theta, phi) for label 2

    thresholds = []#np.zeros(np.shape(thetaphi))
    for theta, phi in zip(thetaphi[0], thetaphi[1]):
        distance_to_target_1 = angular_distance(theta, phi, target_1[0], target_1[1])
        distance_to_target_2 = angular_distance(theta, phi, target_2[0], target_2[1])# If the point is within 10 degrees (converted to radians)
        if distance_to_target_1 <= np.radians(10):
            thresholds.append(0)
        elif distance_to_target_2 <= np.radians(10):
            thresholds.append(3)
        else:
            thresholds.append(7)

    return np.array(thresholds)