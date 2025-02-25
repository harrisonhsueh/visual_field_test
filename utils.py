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


# def print_results(responses_positions, humpfrey_positions, responses_lists, posterior, dot_colors, start_time):
def print_results(humpfrey_phitheta, humpfrey_positions, results, thresholds, posteriors, stimuli_colors,
                  start_time):
    print_results_variable = []
    total_time_elapsed = time.time() - start_time
    print(results)
    for index, pos in enumerate(humpfrey_positions):
        #responses_at_pos = responses_positions[index]
        #true_indices = np.where(responses_at_pos == True)[0]
        #threshold_index = np.min(true_indices) if true_indices.size > 0 else np.nan
        #threshold_color = dot_colors[threshold_index] if not np.isnan(threshold_index) else np.nan

        print_results_variable.append({
            "Position Index": index,
            "X Position": pos[0],
            "Y Position": pos[1],
            "Threshold Color Value": thresholds[index]
        })

    df = pd.DataFrame(print_results_variable)
    print(f"Total Time Elapsed: {total_time_elapsed:.2f} seconds")
    #print(df)
    start_time = datetime.fromtimestamp(start_time)
    start_time = start_time.strftime("%Y-%m-%d_%H-%M-%S")
    df.to_csv(f"visual_field_test_results_{start_time}.csv", index=False)
    np.save(f'visual_field_test_results_{start_time}.npy', results)
    np.save(f'visual_field_test_thresholds_{start_time}.npy', thresholds)
    np.save(f'visual_field_test_positions_{start_time}.npy', humpfrey_positions)
    np.save(f'visual_field_test_phitheta_{start_time}.npy', humpfrey_phitheta)


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

def bayesian_all(prior, intensities, results, k_guess, max_prob_guess, min_prob_guess):
    """
    Update the posterior distribution row-wise after each trial using Bayes' theorem.

    Parameters:
    - prior: Prior probability distribution (shape: (q,)).
    - intensities: Array of intensity values (shape: (q,)).
    - results: Array of test results (shape: (m, n, p)).
    - k_guess: Parameter for the logistic function.
    - max_prob_guess: Maximum probability for the logistic function.
    - min_prob_guess: Minimum probability for the logistic function.

    Returns:
    - posteriors: Updated posterior distribution for each row (shape: (m, q)).
    """
    m, n, _ = results.shape
    q = intensities.shape[0]
    posteriors = np.zeros((m, q))

    for i in range(m):
        # Extract intensity and test result for the i-th row
        intensities_results = results[i, :, 2]  # Shape: (n,)
        test_results = results[i, :, 3]  # Shape: (n,)

        # Flatten and filter out NaNs
        valid_mask = ~np.isnan(test_results)
        if not np.any(valid_mask):
            # If all results in this row are NaN, return the prior
            posteriors[i] = prior.copy()
            continue

        intensities_results = intensities_results[valid_mask]
        test_results = test_results[valid_mask]

        # Compute likelihoods using the logistic function
        intensities_tiled = np.tile(intensities_results[:, None], (1, q))  # Shape: (valid_trials, q)
        likelihoods = logistic_function(intensities_tiled, k_guess, intensities)  # Shape: (valid_trials, q)
        likelihoods = likelihoods * (max_prob_guess - min_prob_guess) + min_prob_guess

        # Adjust likelihoods for test results (0 or 1)
        likelihoods[test_results == 0] = 1 - likelihoods[test_results == 0]

        # Compute the unnormalized posterior: likelihood * prior
        posterior = np.prod(likelihoods, axis=0) * prior  # Shape: (q,)

        # Normalize the posterior to ensure it sums to 1
        posterior /= trapz(posterior, intensities)  # Normalize using trapezoidal integration

        posteriors[i] = posterior  # Store the result

    return posteriors

def confidence_interval_vectorized(posteriors, intensities, confidence):
    """
    Vectorized version of confidence_interval for 2D posteriors.

    Args:
    - posteriors: The posterior distributions (shape: (m, n)).
    - intensities: The array of possible intensity values (shape: (n,)).
    - confidence: The confidence level (e.g., 0.95 for 95% CI).

    Returns:
    - widths: Widths of the confidence intervals (shape: (m,)).
    - lowers: Lower bounds of the confidence intervals (shape: (m,)).
    - uppers: Upper bounds of the confidence intervals (shape: (m,)).
    """
    # Compute cumulative sums along rows
    cumulative = np.cumsum(posteriors, axis=1)

    # Compute lower and upper bounds for each row
    lower_idx = np.array([np.searchsorted(row, (1 - confidence) / 2) for row in cumulative])
    upper_idx = np.array([np.searchsorted(row, (1 + confidence) / 2) for row in cumulative])

    # Use indices to get lower and upper bounds from intensities
    lowers = intensities[lower_idx]
    uppers = intensities[upper_idx]
    widths = uppers - lowers

    return widths, lowers, uppers

# Bayesian update of the posterior using Bayes' Theorem
def bayesian_update_1d(prior, intensities, intensity, result, k_guess, max_prob_guess, min_prob_guess):
    """
    Update the posterior distribution after each trial using Bayes' theorem.

    Args:
    - prior: The current prior distribution over I_threshold.
    - I: The light intensity being tested.
    - result: The observed result (0 or 1) for that intensity.

    Returns:
    - posterior: Updated posterior distribution.
    """
    likelihood = logistic_function(intensity, k_guess, intensities) * (max_prob_guess - min_prob_guess) + min_prob_guess
    # Compute likelihood for success (1) and failure (0)
    if result == 1:
        likelihood = likelihood
    else:
        likelihood = 1 - likelihood

    # Compute the unnormalized posterior: likelihood * prior
    posterior = likelihood * prior

    # Normalize the posterior to ensure it sums to 1 (since it's a probability distribution)
    posterior /= trapz(posterior, intensities)
    return posterior


def choose_next_intensity_1d(prior, intensities, k_guess=10, max_prob_guess=1.0, min_prob_guess=0.0):
    """
    Select the next intensity level to test by maximizing the expected information gain.

    Args:
    - prior: The current prior distribution over the threshold (b).
    - intensities: The array of possible light intensities.
    - k_guess: The slope parameter for the logistic function.
    - max_prob_guess: Maximum probability of success.
    - min_prob_guess: Minimum probability of failure.

    Returns:
    - The intensity level that maximizes the expected information gain.
    """
    # Initialize an array to store the expected information gain for each intensity
    expected_information_gain = np.zeros(len(intensities))
    expected_entropy = np.zeros(len(intensities))

    # Compute bin widths
    bin_widths = np.diff(intensities)  # Differences between adjacent centers
    bin_widths = np.append(bin_widths, bin_widths[-1])  # Assume last bin has the same width as the previous one

    # Compute the entropy of the current prior
    # entropy_prior = entropy(prior)
    entropy_prior = -np.sum(prior * np.log(prior) * bin_widths)

    # Loop over all possible intensities to test
    for i, intensity in enumerate(intensities):
        # Compute the likelihood of success and failure for all possible b values
        likelihood_success_all_b = logistic_function(intensity, k_guess, intensities) * (
                max_prob_guess - min_prob_guess) + min_prob_guess
        likelihood_failure_all_b = 1 - likelihood_success_all_b

        # Compute the expected likelihood by weighting with the prior
        expected_likelihood_success = np.sum(likelihood_success_all_b * prior)
        expected_likelihood_failure = np.sum(likelihood_failure_all_b * prior)

        # Compute the posterior for both possible outcomes (success and failure)
        posterior_success = bayesian_update_1d(prior, intensities, intensity, 1, k_guess, max_prob_guess,
                                               min_prob_guess)
        posterior_failure = bayesian_update_1d(prior, intensities, intensity, 0, k_guess, max_prob_guess,
                                               min_prob_guess)

        # Compute the entropy of the posterior distributions
        # entropy_success = entropy(posterior_success)
        # entropy_failure = entropy(posterior_failure)
        entropy_success = -np.sum(posterior_success * np.log(posterior_success) * bin_widths)
        entropy_failure = -np.sum(posterior_failure * np.log(posterior_failure) * bin_widths)

        expected_entropy_i = expected_likelihood_success * entropy_success + expected_likelihood_failure * entropy_failure
        # Expected information gain is the reduction in entropy
        expected_information_gain[i] = entropy_prior - expected_entropy_i
        expected_entropy[i] = expected_entropy_i

    # Choose the intensity with the maximum expected information gain
    max_info_gain_index = np.argmax(expected_information_gain)
    min_expected_entropy = np.min(expected_entropy)
    return min_expected_entropy, intensities[max_info_gain_index]


# Function to choose the next intensity based on the lookup table
def choose_next_intensity_from_lookup(lookup_table, result_sequence):
    """
    Choose the next intensity based on the precomputed lookup table and the sequence of results so far.
    """

    # print(lookup_table)
    return lookup_table[result_sequence]['intensity']
