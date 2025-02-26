# game.py
import time
import random
import numpy as np
import pygame
from utils import draw_cross, print_results, print_setup, bayesian_all, confidence_interval_vectorized, \
    choose_next_intensity_1d, choose_next_intensity_from_lookup
# Import the constants from constants.py
from constants import SCREEN_SIZE, VIEWER_DISTANCE, PIXELS_PER_CM, WIDTH, HEIGHT, gamma, dBstep_size, background_color, \
    background_level, stimuli_dBlevels, stimuli_cdm2, stimuli_colors, dBlevels_count, b_values, prior, k_guess, \
    max_prob_guess, min_prob_guess, lookup_file, BACKGROUND, WHITE, ORANGE, \
    CROSS_SIZE, \
    CROSS_WIDTH, GAME_DURATION, response_window, time_pause_limit, stimulus_duration, scotoma_points, scotoma_margin, \
    total_point_radius, point_degree_spacing
from humpfrey import hfa_grid, humpfrey_phitheta_to_xy, hfa_24_2_grid, remove_points_farther_than_distance
from sklearn.neighbors import KDTree
from scipy.spatial import ConvexHull
import test_subject_response
import pickle


# Initialize game state variables here
def initialize_game_state():
    # Initializing positions, responses, and other game variables
    humpfrey_phitheta = hfa_grid(radius=total_point_radius, spacing=point_degree_spacing)
    print(f'shape of humpfrey_phitheta {np.shape(humpfrey_phitheta)}')
    humpfrey_phitheta = remove_points_farther_than_distance(humpfrey_phitheta, ref_points=scotoma_points,
                                                            distance_include=scotoma_margin)
    print(f'shape of humpfrey_phitheta {np.shape(humpfrey_phitheta)}')
    humpfrey_positions, dot_radii = humpfrey_phitheta_to_xy(humpfrey_phitheta, WIDTH, HEIGHT, VIEWER_DISTANCE,
                                                            PIXELS_PER_CM)
    print(f'shape of humpfrey_positions {np.shape(humpfrey_positions)}')
    responses_positions = np.empty((humpfrey_positions.shape[0], len(stimuli_dBlevels),
                                    20))  # 3D array for storing responses, only supports up to 10 tests per position
    results = np.zeros((humpfrey_positions.shape[0] + 1, 30,
                        4))  # [-1] is for false positives. Shape (m+1 ,n,p). m is number of positions. n is max number of tests per position, and size p is for time of stimuli, response time, stimuli dB, and 1 or 0 for the result
    responses_positions[:] = np.nan  # Initialize with NaN values to mark no response ##replace with results
    responses_lists = [[] for _ in range(humpfrey_positions.shape[0])]  # replace with results
    responses_times = []  # List to store response times #replace with results
    thresholds = np.empty(humpfrey_positions.shape[0])  # Threshold for each position
    phitheta_kdtree = KDTree(humpfrey_phitheta)
    # Load the lookup table from the file
    with open(lookup_file, 'rb') as f:
        lookup_table = pickle.load(f)
    print(f'dBlevels: {stimuli_dBlevels}')
    time.sleep(3)
    return humpfrey_phitheta, humpfrey_positions, dot_radii, results, responses_positions, responses_lists, responses_times, thresholds, phitheta_kdtree, lookup_table


def choose_next_intensity(position_index, results, posteriors, b_values, lookup_table):
    '''
    :param position_index:
    :param results:
    :param posteriors:
    :param b_values:
    :param lookup_table:
    :return: next_intensity to test
    '''
    # check the lookup table to see if this situation is inside it
    # It is a dictionary with keys '', '0', '1', '00', '01', '11' etc. they stand for the results so far for a position index. The value it has is the test intensity we should choose.
    # If not inside the dictionary, then find next intinsity using choose_next_intensity_1d

    # Identify valid indices where results were recorded (not initialized as zero)
    valid_indices = results[position_index, :, 0] != 0  # Shape: (n,)

    # Extract result sequence using valid indices
    result_sequence = results[position_index, valid_indices, 3]
    result_sequence = ''.join(map(str, result_sequence))  # Convert to string
    if lookup_table and lookup_table.get(result_sequence):
        next_intensity_refined = choose_next_intensity_from_lookup(lookup_table, result_sequence)
    else:
        _, next_intensity_refined = choose_next_intensity_1d(posteriors[position_index], b_values, max_prob_guess=0.95,
                                                             min_prob_guess=0.05)
    return next_intensity_refined


def all_thresholds_found(posteriors, intensities, thresholds, confidence=0.95, confidence_width_tolerance=6):
    widths, lowers, uppers = confidence_interval_vectorized(posteriors, intensities, confidence)
    if np.all(widths < confidence_width_tolerance) and np.all((thresholds > lowers) & (thresholds < uppers)):
        return True
    return False


def update_thresholds(prior, results):
    ''' 
    :param thresholds: array to store calculated thresholds (shape: (m,)).
    :param prior:  array (shape: (m, q)).
    :param results:  array (shape: (m, n, p)).
    :return: posterior (shape: (m, q)).
    '''
    new_posteriors = bayesian_all(prior, b_values, results[:-1], k_guess, max_prob_guess, min_prob_guess)
    new_thresholds = b_values[np.argmax(new_posteriors, axis=1)]
    return new_posteriors, new_thresholds


# Build KD-Tree for the positions
def build_kd_tree(humpfrey_positions):
    positions = humpfrey_positions  # Only x and y coordinates
    return KDTree(positions)


def update_results(pressed, key_press_time, last_dot_time, results, index, dot_color_index, stimuli_dBs):
    print(
        f'updating results with pressed = {pressed}, key_press_time = {key_press_time}, last_dot_time = {last_dot_time}, index = {index}, dot_color_index = {dot_color_index}, dot_dB = {stimuli_dBs[dot_color_index]}')
    print(f'before update: {results}')
    print(results)
    if not pressed:  # stimulus was just flashed
        print('just logging stimuli shown')
        non_zero_indices = results[index, :, 0] != 0  # Find non-zero entries
        if np.any(non_zero_indices):
            last_non_zero_index = np.where(non_zero_indices)[0][-1]
            next_index = last_non_zero_index + 1
        else:
            next_index = 0  # First entry
        if next_index >= results.shape[1]:  # Check for available space
            print(f'no more space at position index {index}')
        else:
            results[index, next_index] = np.array(
                [last_dot_time, 0, stimuli_dBs[dot_color_index], False])
    elif pressed and key_press_time - last_dot_time <= response_window:  # key pressed within time
        # responses[-1] = True
        # set last non np.nan to true
        print('logging stimuli responded positively')
        non_zero_indices = results[index, :, 0] != 0
        if np.any(non_zero_indices):
            print('logging stimuli responded positively, has previous responses')
            last_non_zero_index = np.where(non_zero_indices)[0][-1]
            next_index = last_non_zero_index
        else:
            print('logging stimuli responded positively, has no previous responses')
            next_index = 0
        if results[
            index, next_index, 3] == 0:  # check if stimuli was shown/logged, and this is first key press. otherwise it is a double click and a false positive
            print('logging stimuli responded positively, checked that previous response is 0')
            results[index, next_index] = np.array(
                [last_dot_time, key_press_time - last_dot_time, stimuli_dBs[dot_color_index], True])
        else:
            # false positive, double clicked
            print('logging false positive')
            non_zero_indices = results[results.shape[0] - 1, :, 0] != 0
            if np.any(non_zero_indices):
                last_non_zero_index = np.where(non_zero_indices)[0][-1]
                next_index = last_non_zero_index + 1
            else:
                next_index = 0
            if next_index >= results.shape[1]:
                print(f'no more space for logging false positives')
            else:
                results[results.shape[0] - 1, next_index] = np.array(
                    [last_dot_time, key_press_time - last_dot_time, stimuli_dBs[dot_color_index], True])
    else:
        # false positive, pressed outside of response window or pressed already
        print('logging false positive')
        non_zero_indices = results[results.shape[0] - 1, :, 0] != 0
        if np.any(non_zero_indices):
            last_non_zero_index = np.where(non_zero_indices)[0][-1]
            next_index = last_non_zero_index + 1
        else:
            next_index = 0
        if next_index >= results.shape[1]:
            print(f'no more space for logging false positives')
        else:
            results[results.shape[0] - 1, next_index] = np.array(
                [last_dot_time, key_press_time - last_dot_time, stimuli_dBs[dot_color_index], True])
    return


def display_heatmap_greyscale(screen, humpfrey_positions, responses_positions, responses_lists, dot_colors,
                              dBlevelsCount, dBlevels):
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
                # response = responses_positions[index, :, 0]

                # last_seen_index = np.max(np.nonzero(response == True), initial=-99)
                # thresholds_dB = thresholds_test[index]
                thresholds_dB = dBlevels[np.argmax(bayesian_all(
                    np.ones(dBlevelsCount), dBlevelsCount, dBlevels, responses_lists[index], k_guess=10))]
                # print(thresholds_dB)
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


# def display_heatmap(screen, humpfrey_positions, responses_positions, responses_lists, dot_colors, dBlevelsCount,
#                    dBlevels):
def display_heatmap(screen, humpfrey_positions, results, thresholds, dot_colors, dBlevelsCount,
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
        if 1:  # hull_path.contains_point((x, y)):
            # Calculate the color of the point based on the nearest data point's dB level
            thresholds_dB = thresholds[
                index]  # dBlevels[np.argmax(bayesian_all(np.ones(dBlevelsCount), dBlevelsCount, dBlevels, responses_lists[index], k_guess=10))]

            thresh_levels = 255 * (10 ** (-thresholds_dB / 10))
            gamma_results = 0.7
            color = 255 - 255 * (thresh_levels / 255) ** (1 / gamma_results)
            color = 255 - dot_colors[index]
            # Draw a small square at the (x, y) location, colored based on the nearest point
            # pygame.draw.rect(heatmap, (color, color, color),
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
    screen.fill(BACKGROUND)
    draw_cross(screen, WIDTH, HEIGHT, ORANGE, CROSS_SIZE, CROSS_WIDTH)
    pygame.display.flip()
    running = True
    game_over = False
    responses = []
    start_time = time.time()
    last_dot_time = 0
    dot_visible = False
    print_setup(start_time, SCREEN_SIZE, VIEWER_DISTANCE, PIXELS_PER_CM, WIDTH, HEIGHT, gamma, dBstep_size,
                background_color, background_level, stimuli_dBlevels, stimuli_cdm2, stimuli_colors)
    # Initialize the game state
    (
        humpfrey_phitheta,
        humpfrey_positions,
        dot_radii,
        results,
        responses_positions,
        responses_lists,
        responses_times,
        thresholds,
        phitheta_kdtree,
        lookup_table
    ) = initialize_game_state()
    posteriors = np.tile(prior, (humpfrey_positions.shape[0], 1))
    time_pause = 0  # initialize variable
    while running:
        screen.fill(BACKGROUND)
        draw_cross(screen, WIDTH, HEIGHT, ORANGE, CROSS_SIZE, CROSS_WIDTH)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    if not game_over:
                        update_results(1, time.time(), last_dot_time, results, index, dot_color_index, stimuli_dBlevels)
                        posteriors, thresholds = update_thresholds(prior, results, posteriors, thresholds)
                    else:
                        print("doing nothing if game over and user presses space key")
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False

        if not game_over:
            if time.time() - start_time >= GAME_DURATION or (
                    all_thresholds_found(posteriors, b_values, thresholds, 0.95,
                                         6) and time.time() - last_dot_time > response_window):
                game_over = True
                display_heatmap(screen, humpfrey_positions, results, thresholds, stimuli_colors,
                                dBlevels_count, stimuli_dBlevels)
                print_results(humpfrey_phitheta, humpfrey_positions, results, thresholds, posteriors, stimuli_colors,
                              start_time)
                running = False
            else:
                if time.time() - last_dot_time > time_pause and not dot_visible:
                    time_pause = random.uniform(time_pause_limit[0], time_pause_limit[1])
                    index = np.random.choice(humpfrey_positions.shape[0], 1, replace=False)[0]
                    dot_dB = choose_next_intensity(index, results, posteriors, b_values, lookup_table)
                    dot_color_index = np.argmin(np.abs(stimuli_dBlevels - dot_dB))
                    if dot_color_index is not None:
                        dot_pos = (humpfrey_positions[index, 0], humpfrey_positions[index, 1])
                        dot_radius = (dot_radii[index, 0] + dot_radii[index, 1]) / 2
                        dot_color = (stimuli_colors[dot_color_index],) * 3

                        # responses.append(False)
                        # set first np.nan to false
                        # nan_indices = np.isnan(responses_positions)
                        # if np.any(nan_indices):
                        #     first_nan_index = np.where(nan_indices)[0][0]
                        #     responses_positions[first_nan_index] = False
                        last_dot_time = time.time()
                        update_results(0, 0, last_dot_time, results, index, dot_color_index, stimuli_dBlevels)
                        posteriors, thresholds = update_thresholds(prior, results, posteriors, thresholds)
                        # responses_times.append([index, dot_color_index, last_dot_time, 0])
                        # responses_lists[index].append([dot_color_index, 0])
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
                    responses = []
                    start_time = time.time()
                    # game_over = False keep showing even if just space is clicked
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                    game_over = False
        display_heatmap(screen, humpfrey_positions, results, thresholds, stimuli_colors, len(stimuli_dBlevels),
                        stimuli_dBlevels)

    pygame.quit()


if __name__ == "__main__":
    main()
