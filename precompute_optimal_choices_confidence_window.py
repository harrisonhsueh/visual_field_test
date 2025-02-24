from bimodal_distribution import generate_bimodal_prior
import numpy as np
from constants import stimuli_dBlevels
from numpy import trapz
from scipy.stats import entropy
import time
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator


# Logistic function to model probability of success
def logistic_function(I, k, b):
    return 1 - 1 / (1 + np.exp(-(k * (I - b))))

def confidence_interval(posterior, intensities, confidence=0.95):
    """
    Compute the confidence interval for the posterior distribution.

    Args:
    - posterior: The posterior distribution (shape: (n,)).
    - intensities: The array of possible intensity values (shape: (n,)).
    - confidence: The confidence level (e.g., 0.95 for 95% CI).

    Returns:
    - width: Width of the confidence interval.
    - lower: Lower bound of the confidence interval.
    - upper: Upper bound of the confidence interval.
    """
    cumulative = np.cumsum(posterior)
    lower = intensities[np.searchsorted(cumulative, (1 - confidence) / 2)]
    upper = intensities[np.searchsorted(cumulative, (1 + confidence) / 2)]
    width = upper - lower
    return width, lower, upper

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


from functools import lru_cache


@lru_cache(maxsize=None)
def expected_confidence_interval_after_n_trials(prior_tuple, intensities_tuple, k_guess, max_prob_guess, min_prob_guess, n_trials=4):
    """
    Recursively compute the expected confidence interval width after n_trials,
    given the current prior. At each step, choose the best candidate intensity.
    """
    # Convert tuples back to numpy arrays
    prior = np.array(prior_tuple)
    intensities = np.array(intensities_tuple)

    # Base case: if n_trials == 0, return the confidence interval of the current prior
    if n_trials == 0:
        width, _, _ = confidence_interval(prior, intensities)
        return width, -1, prior

    # Initialize the minimum expected confidence interval width to a large value
    min_expected_width = float('inf')
    best_intensity = None

    # Loop over all candidate intensities to find the one that minimizes the expected confidence interval width
    for i, candidate_intensity in enumerate(intensities):
        # Compute the likelihood of success and failure for the candidate intensity
        likelihood_success = logistic_function(candidate_intensity, k_guess, intensities) * (max_prob_guess - min_prob_guess) + min_prob_guess
        likelihood_failure = 1 - likelihood_success

        # Compute the posterior for success and failure
        posterior_success = bayesian_update_1d(prior, intensities, candidate_intensity, 1, k_guess, max_prob_guess, min_prob_guess)
        posterior_failure = bayesian_update_1d(prior, intensities, candidate_intensity, 0, k_guess, max_prob_guess, min_prob_guess)

        # Recursively compute the expected confidence interval width for the remaining trials
        expected_width_success, _, _ = expected_confidence_interval_after_n_trials(
            tuple(posterior_success), tuple(intensities), k_guess, max_prob_guess, min_prob_guess, n_trials - 1)
        expected_width_failure, _, _ = expected_confidence_interval_after_n_trials(
            tuple(posterior_failure), tuple(intensities), k_guess, max_prob_guess, min_prob_guess, n_trials - 1)

        # Compute the expected confidence interval width for this candidate intensity
        expected_width = np.sum(likelihood_success * expected_width_success + likelihood_failure * expected_width_failure)
        # Update the minimum expected confidence interval width
        if expected_width < min_expected_width:
            min_expected_width = expected_width
            best_intensity = candidate_intensity

    return min_expected_width, best_intensity, prior


def choose_next_intensity_confidence(prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials=4):
    """
    Choose the next intensity level to minimize the expected confidence interval width after n_trials.
    """
    # Convert numpy arrays to tuples for memoization
    prior_tuple = tuple(prior)
    intensities_tuple = tuple(intensities)

    # Compute the expected confidence interval width and best intensity
    expected_width, best_intensity, _ = expected_confidence_interval_after_n_trials(
        prior_tuple, intensities_tuple, k_guess, max_prob_guess, min_prob_guess, n_trials)

    return best_intensity


def precompute_optimal_choices_confidence(prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials=4, sampling_interval=5):
    """
    Precompute the optimal intensity choices for all possible sequences of trial results,
    using the confidence interval-based strategy.
    """
    # Initialize the lookup table
    lookup_table = {}

    # Recursive function to build the lookup table
    def build_lookup(current_prior, current_trial, current_sequence):
        if current_trial == n_trials:
            return None  # Base case: no more trials

        # Choose the best intensity for the current trial
        best_intensity = choose_next_intensity_confidence(current_prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials - current_trial)

        # Simulate both possible outcomes (success and failure)
        likelihood_success = logistic_function(best_intensity, k_guess, intensities) * (max_prob_guess - min_prob_guess) + min_prob_guess
        likelihood_failure = 1 - likelihood_success

        # Compute the posterior for success and failure
        posterior_success = bayesian_update_1d(current_prior, intensities, best_intensity, 1, k_guess, max_prob_guess, min_prob_guess)
        posterior_failure = bayesian_update_1d(current_prior, intensities, best_intensity, 0, k_guess, max_prob_guess, min_prob_guess)

        # Recursively build the lookup table for the next trial
        lookup_table[current_sequence] = {
            'intensity': best_intensity
        }
        build_lookup(posterior_success, current_trial + 1, current_sequence + '1')
        build_lookup(posterior_failure, current_trial + 1, current_sequence + '0')

    # Start building the lookup table
    build_lookup(prior, 0, '')

    return lookup_table


import pickle

intensities = stimuli_dBlevels

# Define the parameters for the bimodal distribution
mean_b1 = 35  # Mean of the first Gaussian
std_b1 = 10  # Standard deviation of the first Gaussian
mean_b2 = 0  # Mean of the second Gaussian
std_b2 = 10  # Standard deviation of the second Gaussian

# Set the weight for each Gaussian
weight_b1 = 1.1  # First Gaussian has twice the weight of the second
weight_b2 = 1  # Second Gaussian has normal weight

# Generate the bimodal prior distribution using the function
#b_values = np.linspace(0, 40, 81)  # Example range for b
b_values = np.linspace(10, 42, 66)  # Example range for b
prior = generate_bimodal_prior(b_values, mean_b1, std_b1, mean_b2, std_b2, weight_b1, weight_b2)

k_guess = 2
max_prob_guess = 0.95
min_prob_guess = 0.05
# Precompute the optimal choices using the confidence interval-based strategy
precompute_start_time = time.time()
lookup_table = precompute_optimal_choices_confidence(prior, b_values, k_guess, max_prob_guess, min_prob_guess, n_trials=3, sampling_interval=5)
precompute_end_time = time.time()
print(f'precompute_total_time = {precompute_end_time - precompute_start_time}')
# Save the lookup table to a file
with open('optimal_choices_confidence_window.pkl', 'wb') as f:
    pickle.dump(lookup_table, f)

