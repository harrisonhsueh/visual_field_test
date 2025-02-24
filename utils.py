# utils.py
import pygame
import numpy as np
import pandas as pd
from datetime import datetime
import csv
import time
from numpy import trapz

def draw_cross(screen, width, height, cross_color, cross_size, line_width):
    pygame.draw.line(screen, cross_color, (width // 2 - cross_size, height // 2),
                     (width // 2 + cross_size, height // 2), line_width)
    pygame.draw.line(screen, cross_color, (width // 2, height // 2 - cross_size),
                     (width // 2, height // 2 + cross_size), line_width)

def print_results(responses_positions, humpfrey_positions, responses_lists, posterior, dot_colors, start_time):
    results = []
    total_time_elapsed = time.time() - start_time
    print(responses_lists)
    for index, pos in enumerate(humpfrey_positions):
        responses_at_pos = responses_positions[index]
        true_indices = np.where(responses_at_pos == True)[0]
        #threshold_index = np.min(true_indices) if true_indices.size > 0 else np.nan
        #threshold_color = dot_colors[threshold_index] if not np.isnan(threshold_index) else np.nan

        results.append({
            "Position Index": index,
            "X Position": pos[0],
            "Y Position": pos[1],
            "Threshold Color Index": threshold_index,
            "Threshold Color Value": threshold_color
        })

    df = pd.DataFrame(results)
    print(f"Total Time Elapsed: {total_time_elapsed:.2f} seconds")
    #print(df)
    start_time = datetime.fromtimestamp(start_time)
    start_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"visual_field_test_results_{start_time}.csv", index=False)

def print_setup(start_time, SCREEN_SIZE, VIEWER_DISTANCE, PIXELS_PER_CM, WIDTH, HEIGHT, gamma, dBstep_size, background_color, background_level, dBlevels, dot_levels, dot_colors):
    start_time = datetime.fromtimestamp(start_time)
    start_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    float_list = [['SCREEN_SIZE', 'VIEWER_DISTANCE', 'PIXELS_PER_CM'],
                  [SCREEN_SIZE, VIEWER_DISTANCE, PIXELS_PER_CM],
                  ['WIDTH', 'HEIGHT'], [WIDTH, HEIGHT],
                  ['gamma', 'dBstep_size', 'background_color', 'background_level'],
                  [gamma, dBstep_size, background_color, background_level],
                  ['dBlevels'], dBlevels.tolist(),
                  ['dot_levels'], dot_levels.tolist(),
                  ['dot_colors'], dot_colors.tolist()]
    output_path = f"visual_field_test_setup_{start_time}.csv"
    with open(output_path, "w") as file:
        writer = csv.writer(file, delimiter=',')
        for i in float_list:
            writer.writerow(i)

# Logistic function to model probability of success for a given dBlevel stimulus
def logistic_function(I, k, b):
    return 1 - 1 / (1 + np.exp(-(k * (I - b))))


# Bayesian update of the posterior using Bayes' Theorem
def bayesian_all(prior, intensity_levels, intensities, results, k_guess=10):
    """
    Update the posterior distribution after each trial using Bayes' theorem.

    Args:
    - prior: The current prior distribution over I_threshold.
    - I: The light intensity being tested.
    - result: The observed result (0 or 1) for that intensity.

    Returns:
    - posterior: Updated posterior distribution.
    """
    results = np.array(results)
    if results.shape[0] == 0:
        return prior / trapz(prior, dx=(max(intensities) - min(intensities)) / intensity_levels)
    # Likelihood of getting 1, given various b guesses
    likelihoods = logistic_function(np.tile(results[:, 0, None], (1, intensity_levels)), k_guess,
                                    np.tile(intensities, (results.shape[0], 1)))

    # Likelihood is 1-likelihood for where results were actually 0, and mask if not tested
    likelihoods[results[:, 1] == 0] = 1 - likelihoods[results[:, 1] == 0]
    likelihoods[np.isnan(results[:, 1])] = np.nan
    likelihoods = np.ma.array(likelihoods, mask=np.isnan(likelihoods))  # Use a mask to mark the NaNs

    # Compute the unnormalized posterior: likelihood * prior
    posterior = np.prod(likelihoods, axis=0) * prior
    # Normalize the posterior to ensure it sums to 1 (since it's a probability distribution)
    posterior /= trapz(posterior, dx=(max(intensities) - min(intensities)) / intensity_levels)
    print(results)
    print(posterior)
    return posterior
