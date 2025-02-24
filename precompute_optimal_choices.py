from bimodal_distribution import generate_bimodal_prior
import numpy as np
from constants import stimuli_dBlevels
from numpy import trapz
from scipy.stats import entropy
import time

# Logistic function to model probability of success
def logistic_function(I, k, b):
    return 1 - 1 / (1 + np.exp(-(k * (I - b))))


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
        likelihood_success_all_b = logistic_function(intensity, k_guess, intensities) * (max_prob_guess - min_prob_guess) + min_prob_guess
        likelihood_failure_all_b = 1 - likelihood_success_all_b

        # Compute the expected likelihood by weighting with the prior
        expected_likelihood_success = np.sum(likelihood_success_all_b * prior)
        expected_likelihood_failure = np.sum(likelihood_failure_all_b * prior)

        # Compute the posterior for both possible outcomes (success and failure)
        posterior_success = bayesian_update_1d(prior, intensities, intensity, 1, k_guess, max_prob_guess, min_prob_guess)
        posterior_failure = bayesian_update_1d(prior, intensities, intensity, 0, k_guess, max_prob_guess, min_prob_guess)

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

from functools import lru_cache
@lru_cache(maxsize=None)  # Cache all results
def expected_entropy_after_n_trials(prior_tuple, intensities_tuple, k_guess, max_prob_guess, min_prob_guess, n_trials=5):
    """
    Recursively compute the expected entropy of the posterior after n_trials,
    given the current prior. At each step, choose the best candidate intensity.
    """
    # Convert tuples back to numpy arrays
    prior = np.array(prior_tuple)
    intensities = np.array(intensities_tuple)
    if n_trials>1:
        print(f'expected_entropy_after_n_trials {n_trials}')
    # Base case: if n_trials == 0, return the entropy of the current prior
    if n_trials == 0:
        return entropy(prior)
    # Special case: if n_trials == 1, use choose_next_intensity_1d for efficiency
    if n_trials == 1:
        # Compute the expected entropy for the next trial only
        return choose_next_intensity_1d(prior, intensities, k_guess, max_prob_guess, min_prob_guess)
    # Initialize the minimum expected entropy to a large value
    min_expected_entropy = float('inf')
    best_intensity =  None

    # Loop over all candidate intensities to find the one that minimizes the expected entropy
    for candidate_intensity in intensities:
        # Compute the likelihood of success and failure for the candidate intensity
        likelihood_success = logistic_function(candidate_intensity, k_guess, intensities) * (max_prob_guess - min_prob_guess) + min_prob_guess
        likelihood_failure = 1 - likelihood_success

        # Compute the posterior for success and failure
        posterior_success = bayesian_update_1d(prior, intensities, candidate_intensity, 1, k_guess, max_prob_guess, min_prob_guess)
        posterior_failure = bayesian_update_1d(prior, intensities, candidate_intensity, 0, k_guess, max_prob_guess, min_prob_guess)

        # Recursively compute the expected entropy for the remaining trials
        #expected_entropy_success, _ = expected_entropy_after_n_trials(posterior_success, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials - 1)
        expected_entropy_success, _ = expected_entropy_after_n_trials(tuple(posterior_success), tuple(intensities),
                                                                   k_guess, max_prob_guess, min_prob_guess,
                                                                   n_trials - 1)
        #print(expected_entropy_success)
        #print(f'returned once {n_trials-1}')
        #expected_entropy_failure, _ = expected_entropy_after_n_trials(posterior_failure, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials - 1)
        expected_entropy_failure, _ = expected_entropy_after_n_trials(tuple(posterior_failure), tuple(intensities),
                                                                   k_guess, max_prob_guess, min_prob_guess,
                                                                   n_trials - 1)

        # Compute the expected entropy for this candidate intensity (scalar value)
        expected_entropy = np.sum(
            likelihood_success * expected_entropy_success + likelihood_failure * expected_entropy_failure)

        # Update the minimum expected entropy
        if expected_entropy < min_expected_entropy:
            min_expected_entropy = expected_entropy
            best_intensity = candidate_intensity

    return min_expected_entropy, best_intensity


def choose_next_intensity_minimize_entropy(prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials=5):
    """
    Choose the next intensity level to minimize the expected entropy after n_trials.
    """
    # Convert numpy arrays to tuples for memoization
    prior_tuple = tuple(prior)
    intensities_tuple = tuple(intensities)
    # Initialize the best intensity and minimum expected entropy
    print(f'choose_next_intensity_minimize_entropy {n_trials}')

    # Compute the expected entropy for this candidate intensity
    expected_entropy, best_intensity = expected_entropy_after_n_trials(prior_tuple, intensities_tuple, k_guess, max_prob_guess, min_prob_guess, n_trials)
    print('completed outer')

    return best_intensity
def precompute_optimal_choices(prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials=5):
    """
    Precompute the optimal intensity choices for all possible sequences of trial results.
    Returns a nested dictionary where each level corresponds to a trial, and each branch corresponds to the result (1 or 0).
    """
    # Initialize the lookup table
    lookup_table = {}

    # Recursive function to build the lookup table
    def build_lookup(current_prior, current_trial, current_sequence):
        if current_trial == n_trials:
            return None  # Base case: no more trials

        # Choose the best intensity for the current trial
        best_intensity = choose_next_intensity_minimize_entropy(current_prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials - current_trial)

        # Simulate both possible outcomes (success and failure)
        likelihood_success = logistic_function(best_intensity, k_guess, intensities) * (max_prob_guess - min_prob_guess) + min_prob_guess
        likelihood_failure = 1 - likelihood_success

        # Compute the posterior for success and failure
        posterior_success = bayesian_update_1d(current_prior, intensities, best_intensity, 1, k_guess, max_prob_guess, min_prob_guess)
        posterior_failure = bayesian_update_1d(current_prior, intensities, best_intensity, 0, k_guess, max_prob_guess, min_prob_guess)

        # Recursively build the lookup table for the next trial
        lookup_table[current_sequence] = {
            'intensity': best_intensity,
            'success': build_lookup(posterior_success, current_trial + 1, current_sequence + '1'),
            'failure': build_lookup(posterior_failure, current_trial + 1, current_sequence + '0')
        }

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
b_values = np.linspace(0, 40, 81)  # Example range for b
prior = generate_bimodal_prior(b_values, mean_b1, std_b1, mean_b2, std_b2, weight_b1, weight_b2)

k_guess = 2
max_prob_guess = 0.95
min_prob_guess = 0.05
# Precompute the optimal choices
# Start the timer for the precomputation
precompute_start_time = time.time()
lookup_table = precompute_optimal_choices(prior, b_values, k_guess, max_prob_guess, min_prob_guess, n_trials=4)
precompute_end_time = time.time()
print(f'precompute_total_time = {precompute_end_time-precompute_start_time}')
# Save the lookup table to a file
with open('optimal_choices.pkl', 'wb') as f:
    pickle.dump(lookup_table, f)