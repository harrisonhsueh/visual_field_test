# game.py
import time
import random
import numpy as np
import pygame
from utils import draw_cross, print_results, print_setup, bayesian_all
# Import the constants from constants.py
from constants import SCREEN_SIZE, VIEWER_DISTANCE, PIXELS_PER_CM, WIDTH, HEIGHT, gamma, dBstep_size, background_color, \
    background_level, stimuli_dBlevels, stimuli_cdm2, stimuli_colors, dBlevels_count, prior, BACKGROUND, WHITE, ORANGE, CROSS_SIZE, \
    CROSS_WIDTH, GAME_DURATION, response_window, time_pause_limit, stimulus_duration, scotoma_points, scotoma_margin
from humpfrey import hfa_grid, humpfrey_phitheta_to_xy, hfa_24_2_grid, remove_points_farther_than_distance
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull
import test_subject_response


# Initialize game state variables here
def initialize_game_state():
    # Initializing positions, responses, and other game variables
    humpfrey_phitheta = hfa_grid(radius=24, spacing=2)
    humpfrey_phitheta = remove_points_farther_than_distance(humpfrey_phitheta, ref_points=scotoma_points, distance_include=scotoma_margin)
    #humpfrey_positions = humpfrey_phitheta(humpfrey_phitheta, WIDTH, HEIGHT, VIEWER_DISTANCE, PIXELS_PER_CM).T
    humpfrey_positions, dot_radii = humpfrey_phitheta_to_xy(humpfrey_phitheta, WIDTH, HEIGHT, VIEWER_DISTANCE, PIXELS_PER_CM)
    responses_positions = np.empty((humpfrey_positions.shape[0], len(dBlevels_count), 10))  # 3D array for storing responses, only supports up to 10 tests per position
    priors = np.repeat( prior[None,:], humpfrey_positions.shape[0], axis=0 )
    responses_positions[:] = np.nan  # Initialize with NaN values to mark no response
    responses_lists = [[] for _ in range(humpfrey_positions.shape[0])]
    responses_times = []  # List to store response times
    thresholds = np.empty(humpfrey_positions.shape[0])  # Threshold for each position
    phitheta_kdtree = KDTree(humpfrey_phitheta)
    print(f'dBlevels: {stimuli_dBlevels}')
    time.sleep(3)
    return humpfrey_positions, dot_radii, responses_positions, responses_lists, responses_times, thresholds, phitheta_kdtree


def find_next_color_index(index, responses_positions, dBlevelsCount):
    low, high = 0, dBlevelsCount - 1
    responses_at_pos = responses_positions[index, :, 0]

    while low <= high:
        mid = (low + high) // 2
        if np.isnan(responses_at_pos[mid]):
            return mid
        elif responses_at_pos[mid]:
            high = mid - 1
        else:
            low = mid + 1
    return None
# Function to select the next light intensity based on the current posterior
def choose_next_intensity_index(posterior):
    # Find all indices where the posterior is maximized
    max_posterior_value = np.max(posterior)
    max_indices = np.where(posterior == max_posterior_value)[0]

    # Choose the center intensity from those indices
    center_index = (max_indices[0] + max_indices[-1]) // 2
    return center_index


def all_thresholds_found(responses_lists, humpfrey_positions):
    a = [len(x) for x in responses_lists]
    if min(a) > 5:
        return True
    return False


# Build KD-Tree for the positions
def build_kd_tree(humpfrey_positions):
    positions = humpfrey_positions # Only x and y coordinates
    return KDTree(positions)


def display_heatmap_greyscale(screen, humpfrey_positions, responses_positions, responses_lists, dot_colors, dBlevelsCount, dBlevels):
    thresholds_test = test_subject_response.sensitivity(hfa_24_2_grid())
    # Build KD-Tree once at the beginning
    kd_tree = build_kd_tree(humpfrey_positions)

    heatmap = pygame.Surface((WIDTH, HEIGHT))
    heatmap.fill(WHITE)
    # Instead of querying every pixel, let's sample every 10th pixel
    step_size = 5  # Reduce the resolution by a factor of 10 for faster processing
    # Define the size of the squares (adjustable)
    square_size = step_size  # This will match the step size, so each square covers the area we're sampling
    # Use the Convex Hull to create a smooth boundary around the points
    points_2d = humpfrey_positions
    try:
        hull = ConvexHull(points_2d)  # We only need the (x, y) positions
        hull_points = humpfrey_positions[hull.vertices]  # Get the points that form the convex hull
        # Draw the convex hull as a polygon on the heatmap (smooth boundary)
        pygame.draw.polygon(heatmap, (255, 255, 255), hull_points, width=2)  # Red outline
    except Exception as e:
        print("Error calculating Convex Hull:", e)
    # Create a Path object from the convex hull vertices (for point-in-polygon checks)
    from matplotlib.path import Path
    hull_path = Path(hull_points)

    for y in range(0, HEIGHT, step_size):
        for x in range(0, WIDTH, step_size):
            # Check if the point (x, y) is inside the convex hull using matplotlib's Path.contains_point
            if hull_path.contains_point((x, y)):
                # Find the nearest data point in humpfrey_positions
                # Find the nearest data point using the KD-Tree
                dist, nearest_index = kd_tree.query([[x, y]], k=1)  # k=1 to find the closest point

                # Calculate the color of the point based on the nearest data point's dB level
                # You can also use the dot_levels or any other color logic ba
                index = nearest_index[0][0]
                #response = responses_positions[index, :, 0]

                #last_seen_index = np.max(np.nonzero(response == True), initial=-99)
                #thresholds_dB = thresholds_test[index]
                thresholds_dB = dBlevels[np.argmax(bayesian_all(
                    np.ones(dBlevelsCount), dBlevelsCount, dBlevels, responses_lists[index], k_guess=10))]
                #print(thresholds_dB)
                thresh_levels = 255 * (10 ** (-thresholds_dB / 10))
                gamma_results = 0.7
                color = 255 - 255 * (thresh_levels / 255) ** (1 / gamma_results)
                # color = dot_colors[
                #            dBlevelsCount - 1 - last_seen_index] - fudge_darker if last_seen_index >= -1 else 10
                # Drawing a small square at the (x, y) location, colored based on the nearest point
                pygame.draw.rect(heatmap, (color, color, color), pygame.Rect(x, y, square_size, square_size))
    pygame.draw.polygon(heatmap, (0, 0, 0), hull_points, width=2)  # Red outline
    screen.blit(heatmap, (0, 0))
    draw_cross(screen, WIDTH, HEIGHT, ORANGE, CROSS_SIZE, CROSS_WIDTH)
    pygame.display.flip()


def display_heatmap(screen, humpfrey_positions, responses_positions, responses_lists, dot_colors, dBlevelsCount,
                    dBlevels):
    thresholds_test = test_subject_response.sensitivity(hfa_24_2_grid())
    # Build KD-Tree once at the beginning
    kd_tree = build_kd_tree(humpfrey_positions)

    heatmap = pygame.Surface((WIDTH, HEIGHT))
    heatmap.fill(WHITE)

    # Use the Convex Hull to create a smooth boundary around the points
    points_2d = humpfrey_positions
    try:
        hull = ConvexHull(points_2d)  # We only need the (x, y) positions
        hull_points = humpfrey_positions[hull.vertices]  # Get the points that form the convex hull
        # Draw the convex hull as a polygon on the heatmap (smooth boundary)
        pygame.draw.polygon(heatmap, (255, 255, 255), hull_points, width=2)  # White outline
    except Exception as e:
        print("Error calculating Convex Hull:", e)
    # Create a Path object from the convex hull vertices (for point-in-polygon checks)
    from matplotlib.path import Path
    hull_path = Path(hull_points)

    font = pygame.font.SysFont('Arial', 12)  # Choose a font and size for threshold text

    for index in range(humpfrey_positions.shape[0]):  # Iterate through all test points
        x, y = humpfrey_positions[index]  # Get the position of the test point

        # Check if the point is inside the convex hull using matplotlib's Path.contains_point
        if 1:#hull_path.contains_point((x, y)):
            # Calculate the color of the point based on the nearest data point's dB level
            thresholds_dB = dBlevels[np.argmax(bayesian_all(
                np.ones(dBlevelsCount), dBlevelsCount, dBlevels, responses_lists[index], k_guess=10))]

            thresh_levels = 255 * (10 ** (-thresholds_dB / 10))
            gamma_results = 0.7
            color = 255 - 255 * (thresh_levels / 255) ** (1 / gamma_results)

            # Draw a small square at the (x, y) location, colored based on the nearest point
            #pygame.draw.rect(heatmap, (color, color, color),
            #                 pygame.Rect(x - 3, y - 3, 6, 6))  # Slightly adjust for square size

            # Render the threshold value as text
            threshold_text = f'{thresholds_dB:.0f}'  # Show the threshold value as text
            text_surface = font.render(threshold_text, True, (0, 0, 0))  # Black color for text
            text_rect = text_surface.get_rect(center=(x, y))  # Center the text on the test point
            heatmap.blit(text_surface, text_rect)

    pygame.draw.polygon(heatmap, (0, 0, 0), hull_points, width=2)  # Black outline around convex hull
    screen.blit(heatmap, (0, 0))
    draw_cross(screen, WIDTH, HEIGHT, ORANGE, CROSS_SIZE, CROSS_WIDTH)
    pygame.display.flip()

def main(screen):
    running = True
    game_over = False
    dot_positions = []
    responses = []
    start_time = time.time()
    last_dot_time = 0
    dot_visible = False
    print_setup(start_time, SCREEN_SIZE, VIEWER_DISTANCE, PIXELS_PER_CM, WIDTH, HEIGHT, gamma, dBstep_size,
                background_color, background_level, stimuli_dBlevels, stimuli_cdm2, stimuli_colors)
    # Initialize the game state
    (
        humpfrey_positions,
        dot_radii,
        responses_positions,
        responses_lists,
        responses_times,
        thresholds,
        phitheta_kdtree
    ) = initialize_game_state()

    while running:
        screen.fill(BACKGROUND)
        draw_cross(screen, WIDTH, HEIGHT, ORANGE, CROSS_SIZE, CROSS_WIDTH)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not game_over:
                        if dot_visible or (time.time() - last_dot_time <= response_window):
                            responses[-1] = True
                            # set last non np.nan to true
                            non_nan_indices = ~np.isnan(responses_positions)
                            if np.any(non_nan_indices):
                                last_non_nan_index = np.where(non_nan_indices)[0][-1]
                                responses_positions[last_non_nan_index] = True
                            responses_lists[index][-1] = [dot_color_index, 1]
                            responses_times.append([index, dot_color_index, time.time(), 1])
                        responses_times.append([np.inf, np.inf, time.time(), 1])
                    else:
                        dot_positions = []
                        responses = []
                        start_time = time.time()
                        game_over = False
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        if not game_over:
            if time.time() - start_time >= GAME_DURATION or (all_thresholds_found(responses_lists,
                                                                                 humpfrey_positions) and time.time() - last_dot_time > response_window):
                game_over = True
                display_heatmap(screen, humpfrey_positions, responses_positions, responses_lists, stimuli_colors, dBlevels_count, dBlevels_count)
                print_results(responses_positions, humpfrey_positions, responses_lists, stimuli_colors, start_time)
                running = False
            else:
                if len(dot_positions) == 0 or (time.time() - last_dot_time > time_pause and not dot_visible):
                    time_pause = random.randint(time_pause_limit[0], time_pause_limit[1])
                    index = np.random.choice(humpfrey_positions.shape[0], 1, replace=False)[0]
                    #dot_color_index = find_next_color_index(index, responses_positions, dBlevelsCount)
                    posterior = bayesian_all(np.ones(dBlevels_count), dBlevels_count, stimuli_dBlevels, responses_lists[index], k_guess = 10)
                    dot_color_index = choose_next_intensity_index(posterior)
                    if dot_color_index is not None:
                        dot_pos = (humpfrey_positions[index, 0], humpfrey_positions[index, 1])
                        dot_radius = (dot_radii[index,0] + dot_radii[index,1]) / 2
                        dot_color = (stimuli_colors[dot_color_index],) * 3

                        dot_positions.append(index)
                        responses.append(False)
                        #set first np.nan to false
                        nan_indices = np.isnan(responses_positions)
                        if np.any(nan_indices):
                            first_nan_index = np.where(nan_indices)[0][0]
                            responses_positions[first_nan_index] = False
                        last_dot_time = time.time()
                        responses_times.append([index, dot_color_index, last_dot_time, 0])
                        responses_lists[index].append([dot_color_index,0])
                        dot_visible = True

                if dot_visible:
                    pygame.draw.circle(screen, dot_color, dot_pos, dot_radius)
                    pygame.display.flip()
                    time.sleep(stimulus_duration)
                    screen.fill(BACKGROUND)
                    draw_cross(screen, WIDTH, HEIGHT, ORANGE, CROSS_SIZE, CROSS_WIDTH)
                    pygame.display.flip()
                    dot_visible = False

        pygame.display.flip()

    while game_over:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                game_over = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    dot_positions = []
                    responses = []
                    start_time = time.time()
                    game_over = False
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                    game_over = False
        display_heatmap(screen, humpfrey_positions, responses_positions, responses_lists, stimuli_colors, dBlevelsCount,
                        stimuli_dBlevels)

    pygame.quit()


if __name__ == "__main__":
    main()
