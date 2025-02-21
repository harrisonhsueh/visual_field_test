import numpy as np
from numpy import arange, pi, sin, cos, arccos
from math import ceil
def hfa_grid(radius=24, spacing=6):
    """Generate a grid of points for the HFA 24-2 protocol"""
    theta_values = np.arange(-(ceil(radius / spacing) + 0.5) * spacing, (ceil(radius / spacing) + 0.5) * spacing,
                             spacing)
    phi_values = np.arange(-(ceil(radius / spacing) + 0.5) * spacing, (ceil(radius / spacing) + 0.5) * spacing, spacing)
    points = []

    for theta in theta_values:
        for phi in phi_values:
            if theta == 0 and phi == 0:
                continue
            distance = np.sqrt(theta ** 2 + phi ** 2)
            if distance <= radius:
                points.append([phi, theta])

    return np.array(points)

def humpfrey_phitheta_to_xy(phitheta, WIDTH, HEIGHT, VIEWER_DISTANCE, PIXELS_PER_CM):
    phitheta = phitheta / 180 * pi
    phi = phitheta[:,0]
    theta = phitheta[:,1]+pi/2
    x, y, z = sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta)

    # project to 2D screen that is VIEWER_DISTANCE away.
    # AKA x is VIEWER_DISTANCE, find y and z. -y is 2d screen's x. z is 2d screen's y.
    # r = VIEWER_DISTANCE / (cos(theta) * sin(phi))
    # screen_x, screen_y = -sin(theta) * sin(phi)*r, cos(phi)*r
    # z is viewer distance
    # r = viewer_distance/cos(theta)
    #screen x is y, screen y is negative x
    screen_x, screen_y = VIEWER_DISTANCE * (sin(phi) / cos(phi)), VIEWER_DISTANCE * (
                cos(theta) / sin(theta) / cos(phi))

    humpfrey_positions = np.asarray((PIXELS_PER_CM * screen_x, PIXELS_PER_CM * screen_y))
    humpfrey_positions[0] += WIDTH // 2
    humpfrey_positions[1] += HEIGHT // 2
    # print(humpfrey_positions)
    DOT_SIZE_radius_radian = 0.43 / 180 * pi # https://www.ncbi.nlm.nih.gov/books/NBK585112/ # Target 	Size ( in square mm)	Degrees:III	4	0.43 degrees
    phi += DOT_SIZE_radius_radian
    theta += DOT_SIZE_radius_radian
    screen_x_plus_dot, screen_y_plus_dot = VIEWER_DISTANCE * ( sin(phi) / cos(phi)), VIEWER_DISTANCE * (
                cos(theta) / sin(theta) / cos(phi))


    humpfrey_positions_plus_dot = np.asarray((PIXELS_PER_CM * screen_x_plus_dot, PIXELS_PER_CM * screen_y_plus_dot))
    humpfrey_positions_plus_dot[0] += WIDTH // 2
    humpfrey_positions_plus_dot[1] += HEIGHT // 2
    dot_size = np.abs(humpfrey_positions_plus_dot - humpfrey_positions)
    # print(humpfrey_positions_plus_dot)
    return humpfrey_positions.T, dot_size.T

def hfa_24_2_grid():
    points = hfa_grid(radius=24, spacing=6)
    # points to add
    add_points = [[-27, 3], [-27, 3]]  # Example points to add
    standard_24_2_points = postprocess_add_remove(points, manual_add=add_points)
    return standard_24_2_points

def postprocess_add_remove(points, manual_add=None, manual_remove=None):
    """Postprocess the grid: add and remove specific points"""
    # Convert manual_add and manual_remove to numpy arrays for easier comparison
    if manual_remove is None:
        manual_remove = []
    if manual_add is None:
        manual_add = []
    manual_add = np.array(manual_add)
    manual_remove = np.array(manual_remove)

    # Remove points specified in manual_remove
    points = [point for point in points if not any(np.array_equal(point, rm) for rm in manual_remove)]

    # Add points specified in manual_add, ensuring no duplicates
    for point in manual_add:
        if not any(np.array_equal(point, p) for p in points):
            points.append(point)

    return np.array(points)

def remove_points_farther_than_distance(points, ref_points=None, distance_include=999):
    """Remove points that are farther than a given distance from any of the reference points"""
    if ref_points is None:
        ref_points = [0, 0]
    filtered_points = []
    for point in points:
        # Check the distance to each reference point
        if any(np.linalg.norm(np.array(point) - np.array(ref_point)) <= distance_include for ref_point in ref_points):
            filtered_points.append(point)

    return np.array(filtered_points)
