import numpy as np
from numpy import arange, pi, sin, cos, arccos
def humpfrey_point_generator():
    manual_theta_phi = [[[21,21,21,21],[-9,-3,3,9]]] #skip top and bottom most

    #manual_theta_phi = []
    manual_theta_phi.append([[15, 15, 15, 15, 15, 15], [-15, -9, -3, 3, 9, 15]])
    #manual_theta_phi.append([[15, 15, 15], [3, 9, 15]])#removed negatives
    #manual_theta_phi.append([[15, 15, 15, 15, 15, 15], [-15, -9, -3, 3, 9, 15]])
    manual_theta_phi.append([[9, 9, 9, 9, 9, 9, 9, 9], [-21, -15, -9, -3, 3, 9, 15, 21]])
    #manual_theta_phi.append([[9, 9, 9, 9, 9, 9, 9], [-15, -9, -3, 3, 9, 15, 21]]) #remove -21
    manual_theta_phi.append([[3,3,3,3,3,3,3,3,3],[-27,-21,-15,-9,-3,3,9,15,21]])
    #manual_theta_phi.append([[3, 3, 3, 3, 3, 3, 3], [-15, -9, -3, 3, 9, 15, 21]]) #remove 2 leftmost
    manual_theta_phi = [[[3,3],[-3,3]]]
    merged_manual_theta_phi = [[],[]]
    for i in manual_theta_phi:
        merged_manual_theta_phi[0] += i[0]
        merged_manual_theta_phi[1] += i[1]
    half_theta_phi = np.asarray(merged_manual_theta_phi)
    theta_phi = np.hstack((half_theta_phi,np.asarray([half_theta_phi[0]*-1,half_theta_phi[1]])))
    theta_phi = np.asarray([[3, 3, -3, -3],[-3,3,3, -3]])
    #theta_phi = np.hstack((theta_phi,np.asarray([[0],[-33]])))
    #print(theta_phi)
    return theta_phi

def humpfrey_thetaphi_to_xy(thetaphi, WIDTH, HEIGHT, VIEWER_DISTANCE, PIXELS_PER_CM):
    theta_phi = thetaphi / 180 * pi
    theta = theta_phi[0]+pi/2
    phi = theta_phi[1]
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
    return np.vstack((humpfrey_positions, dot_size))