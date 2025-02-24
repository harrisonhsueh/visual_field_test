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
def bayesian_update_1d_vectorized(prior, intensities, test_intensities, results, k_guess, max_prob_guess, min_prob_guess):
    """
    Vectorized version of bayesian_update_1d.

    Args:
    - prior: The current prior distribution over I_threshold (shape: (n,)).
    - intensities: Array of possible intensity values (shape: (n,)).
    - test_intensities: Array of intensities being tested (shape: (m,)).
    - results: Array of observed results (0 or 1) for each test intensity (shape: (m,)).
    - k_guess: Slope parameter for the logistic function.
    - max_prob_guess: Maximum probability of success.
    - min_prob_guess: Minimum probability of failure.

    Returns:
    - posterior: Updated posterior distribution (shape: (n,)).
    """
    # Compute likelihood for all test intensities and all b values
    # Shape: (m, n)
    likelihood = logistic_function(test_intensities[:, None], k_guess, intensities) * (max_prob_guess - min_prob_guess) + min_prob_guess

    # Adjust likelihood based on results (success or failure)
    # Shape: (m, n)
    likelihood = np.where(results[:, None] == 1, likelihood, 1 - likelihood)

    # Compute the unnormalized posterior for each test intensity: likelihood * prior
    # Shape: (m, n)
    posteriors = likelihood * prior

    # Normalize each posterior to ensure it sums to 1
    # Shape: (m, n)
    posteriors /= trapz(posteriors, intensities, axis=1)[:, None]

    return posteriors

def choose_next_intensity_1d_non_vectorized_fully(prior, intensities, k_guess=10, max_prob_guess=1.0, min_prob_guess=0.0):
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

    #time_start = time.perf_counter()

    expected_information_gain = np.zeros(len(intensities))
    expected_entropy = np.zeros(len(intensities))

    # Compute bin widths
    bin_widths = np.diff(intensities)  # Differences between adjacent centers
    bin_widths = np.append(bin_widths, bin_widths[-1])  # Assume last bin has the same width as the previous one

    # Compute the entropy of the current prior
    # entropy_prior = entropy(prior)
    # entropy_prior = -np.sum(prior * np.log(prior) * bin_widths)

    # Compute likelihood_success_all_b for all intensities at once
    likelihood_success_all_b = logistic_function(intensities[:, None], k_guess, intensities) * (
                max_prob_guess - min_prob_guess) + min_prob_guess

    # Compute likelihood_failure_all_b for all intensities
    likelihood_failure_all_b = 1 - likelihood_success_all_b

    # Compute expected_likelihood_success and expected_likelihood_failure for all intensities
    expected_likelihood_success = np.sum(likelihood_success_all_b * prior, axis=1)
    expected_likelihood_failure = np.sum(likelihood_failure_all_b * prior, axis=1)

    # Loop over all possible intensities to test
    for i, intensity in enumerate(intensities):
        # Compute the likelihood of success and failure for all possible b values
        ## likelihood_success_all_b = logistic_function(intensity, k_guess, intensities) * (
        ##            max_prob_guess - min_prob_guess) + min_prob_guess
        ## likelihood_failure_all_b = 1 - likelihood_success_all_b

        # Compute the expected likelihood by weighting with the prior
        ## expected_likelihood_success = np.sum(likelihood_success_all_b * prior)
        ## expected_likelihood_failure = np.sum(likelihood_failure_all_b * prior)

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

        expected_entropy_i = expected_likelihood_success[i] * entropy_success + expected_likelihood_failure[i] * entropy_failure
        # Expected information gain is the reduction in entropy
        #expected_information_gain[i] = entropy_prior - expected_entropy_i
        expected_entropy[i] = expected_entropy_i

    # Choose the intensity with the maximum expected information gain
    #max_info_gain_index = np.argmax(expected_information_gain)
    max_info_gain_index = np.argmin(expected_entropy)
    min_expected_entropy = np.min(expected_entropy)
    #time_end = time.perf_counter()
    #time_elapsed = time_end - time_start
    #print(f'time for choose_next_intensity_1d = {time_elapsed}')
    return min_expected_entropy, intensities[max_info_gain_index], expected_entropy


def choose_next_intensity_1d(prior, intensities, k_guess=10, max_prob_guess=1.0, min_prob_guess=0.0):
    """
    Fully vectorized version of choose_next_intensity_1d.
    """
    #time_start = time.perf_counter()

    # Compute bin widths once
    bin_widths = np.diff(intensities)
    bin_widths = np.append(bin_widths, bin_widths[-1])

    # Compute the entropy of the current prior
    # entropy_prior = -np.sum(prior * np.log(prior) * bin_widths)

    # Vectorized computation of likelihood_success_all_b and likelihood_failure_all_b
    likelihood_success_all_b = logistic_function(intensities[:, None], k_guess, intensities) * (max_prob_guess - min_prob_guess) + min_prob_guess
    likelihood_failure_all_b = 1 - likelihood_success_all_b

    # Compute expected_likelihood_success and expected_likelihood_failure for all intensities
    expected_likelihood_success = np.sum(likelihood_success_all_b * prior, axis=1)
    expected_likelihood_failure = np.sum(likelihood_failure_all_b * prior, axis=1)

    # Vectorized computation of posteriors for all intensities
    posteriors_success = bayesian_update_1d_vectorized(prior, intensities, intensities, np.ones(len(intensities)), k_guess, max_prob_guess, min_prob_guess)
    posteriors_failure = bayesian_update_1d_vectorized(prior, intensities, intensities, np.zeros(len(intensities)), k_guess, max_prob_guess, min_prob_guess)

    # Vectorized computation of entropy for all posteriors
    entropy_success = -np.sum(posteriors_success * np.log(posteriors_success) * bin_widths, axis=1)
    entropy_failure = -np.sum(posteriors_failure * np.log(posteriors_failure) * bin_widths, axis=1)

    # Vectorized computation of expected entropy
    expected_entropy = expected_likelihood_success * entropy_success + expected_likelihood_failure * entropy_failure

    # Choose the intensity with the minimum expected entropy
    max_info_gain_index = np.argmin(expected_entropy)
    min_expected_entropy = np.min(expected_entropy)

    #time_end = time.perf_counter()
    #time_elapsed = time_end - time_start
    #print(f'Time for choose_next_intensity_1d_fully_vectorized = {time_elapsed:.6f} seconds')

    return min_expected_entropy, intensities[max_info_gain_index], expected_entropy
from functools import lru_cache


@lru_cache(maxsize=None)  # Cache all results
def expected_entropy_after_n_trials(prior_tuple, intensities_tuple, k_guess, max_prob_guess, min_prob_guess,
                                    n_trials=5):
    """
    Recursively compute the expected entropy of the posterior after n_trials,
    given the current prior. At each step, choose the best candidate intensity.
    """
    # Convert tuples back to numpy arrays
    prior = np.array(prior_tuple)
    intensities = np.array(intensities_tuple)
    expected_entropy_2d = np.zeros((np.shape(intensities)[0], np.shape(intensities)[0]))
    expected_entropy_1d = np.zeros(np.shape(intensities)[0])
    ##if n_trials>1:
    ##    print(f'expected_entropy_after_n_trials {n_trials}')
    # Base case: if n_trials == 0, return the entropy of the current prior
    if n_trials == 0:
        return entropy(prior), -1, prior  # this is an error, no "best" intensity choice if no trials
    # Special case: if n_trials == 1, use choose_next_intensity_1d for efficiency
    if n_trials == 1:
        # Compute the expected entropy for the next trial only
        return choose_next_intensity_1d(prior, intensities, k_guess, max_prob_guess, min_prob_guess)
    # Initialize the minimum expected entropy to a large value
    min_expected_entropy = float('inf')
    best_intensity = None

    # Loop over all candidate intensities to find the one that minimizes the expected entropy
    for i, candidate_intensity in enumerate(intensities):
        # Compute the likelihood of success and failure for the candidate intensity
        likelihood_success = logistic_function(candidate_intensity, k_guess, intensities) * (
                    max_prob_guess - min_prob_guess) + min_prob_guess
        likelihood_failure = 1 - likelihood_success

        # Compute the expected likelihood by weighting with the prior
        expected_likelihood_success = np.sum(likelihood_success * prior)
        expected_likelihood_failure = np.sum(likelihood_failure * prior)

        #time_start = time.perf_counter()
        # Compute the posterior for success and failure
        posterior_success = bayesian_update_1d(prior, intensities, candidate_intensity, 1, k_guess, max_prob_guess,
                                               min_prob_guess)
        posterior_failure = bayesian_update_1d(prior, intensities, candidate_intensity, 0, k_guess, max_prob_guess,
                                               min_prob_guess)
        #time_end = time.perf_counter()
        #time_elapsed = time_end - time_start
        #print(f'time for 2 bayesian updates = {time_elapsed}')

        #time_start = time.perf_counter()
        # Recursively compute the expected entropy for the remaining trials
        # expected_entropy_success, _ = expected_entropy_after_n_trials(posterior_success, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials - 1)
        expected_entropy_success, _, expected_entropy_success_1d = expected_entropy_after_n_trials(
            tuple(posterior_success), tuple(intensities),
            k_guess, max_prob_guess, min_prob_guess,
            n_trials - 1)
        # print(expected_entropy_success)
        # print(f'returned once {n_trials-1}')
        # expected_entropy_failure, _ = expected_entropy_after_n_trials(posterior_failure, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials - 1)
        expected_entropy_failure, _, expected_entropy_failure_1d = expected_entropy_after_n_trials(
            tuple(posterior_failure), tuple(intensities),
            k_guess, max_prob_guess, min_prob_guess,
            n_trials - 1)
        #time_end = time.perf_counter()
        #time_elapsed = time_end - time_start
        #print(f'time for 2 n_trials n={n_trials - 1} = {time_elapsed}')
        expected_entropy_2d[
            i] = expected_likelihood_success * expected_entropy_success + expected_likelihood_failure * expected_entropy_failure
        # Compute the expected entropy for this candidate intensity (scalar value)
        expected_entropy = np.sum(expected_entropy_2d[i])
        expected_entropy_1d[i] = expected_entropy
        # Update the minimum expected entropy
        if expected_entropy < min_expected_entropy:
            min_expected_entropy = expected_entropy
            best_intensity = candidate_intensity
    if 0 and n_trials == 3:
        print(expected_entropy_2d)
        print(expected_entropy_1d)
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        X, Y = np.meshgrid(intensities, intensities)

        # Plot the surface.
        surf = ax.plot_surface(X, Y, expected_entropy_2d, cmap=cm.coolwarm,
                               linewidth=0, antialiased=False)

        # Customize the z axis.
        # ax.set_zlim(-1.01, 1.01)
        # ax.zaxis.set_major_locator(LinearLocator(10))
        # A StrMethodFormatter is used automatically
        ax.zaxis.set_major_formatter('{x:.02f}')

        # Add a color bar which maps values to colors.
        fig.colorbar(surf, shrink=0.5, aspect=5)

        plt.title(f"expected entropies for n_trials = {n_trials}")
        plt.show()
        plt.figure(figsize=(10, 6))
        plt.plot(intensities, expected_entropy_1d)
        plt.title(f"expected entropies for n_trials = {n_trials}")
        plt.show()

    return min_expected_entropy, best_intensity, expected_entropy_1d


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
    expected_entropy, best_intensity = expected_entropy_after_n_trials(prior_tuple, intensities_tuple, k_guess,
                                                                       max_prob_guess, min_prob_guess, n_trials)
    print('completed outer')

    return best_intensity


def precompute_optimal_choices(prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials=5, sampling_interval = 20):
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
        best_intensity = choose_next_intensity_fast(current_prior, intensities, k_guess, max_prob_guess, min_prob_guess,
                                                    n_trials - current_trial, sampling_interval)

        # Simulate both possible outcomes (success and failure)
        likelihood_success = logistic_function(best_intensity, k_guess, intensities) * (
                    max_prob_guess - min_prob_guess) + min_prob_guess
        likelihood_failure = 1 - likelihood_success

        # Compute the posterior for success and failure
        posterior_success = bayesian_update_1d(current_prior, intensities, best_intensity, 1, k_guess, max_prob_guess,
                                               min_prob_guess)
        posterior_failure = bayesian_update_1d(current_prior, intensities, best_intensity, 0, k_guess, max_prob_guess,
                                               min_prob_guess)
        print(f'building rest after trial {current_trial}')
        # Recursively build the lookup table for the next trial
        lookup_table[current_sequence] = {
            'intensity': best_intensity
        }
        build_lookup(posterior_success, current_trial + 1, current_sequence + '1')
        build_lookup(posterior_failure, current_trial + 1, current_sequence + '0')

    # Start building the lookup table
    build_lookup(prior, 0, '')

    return lookup_table


def expected_entropy(intensity, prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials):
    """
    Wrapper function to compute expected entropy for a single intensity.
    """
    print(intensity[0])
    likelihood_success = logistic_function(intensity, k_guess, intensities) * (
                max_prob_guess - min_prob_guess) + min_prob_guess
    likelihood_failure = 1 - likelihood_success

    posterior_success = bayesian_update_1d(prior, intensities, intensity, 1, k_guess, max_prob_guess, min_prob_guess)
    posterior_failure = bayesian_update_1d(prior, intensities, intensity, 0, k_guess, max_prob_guess, min_prob_guess)

    expected_entropy_success, _, _ = expected_entropy_after_n_trials(tuple(posterior_success), tuple(intensities),
                                                                     k_guess, max_prob_guess, min_prob_guess,
                                                                     n_trials - 1)
    expected_entropy_failure, _, _ = expected_entropy_after_n_trials(tuple(posterior_failure), tuple(intensities),
                                                                     k_guess, max_prob_guess, min_prob_guess,
                                                                     n_trials - 1)

    return np.sum(likelihood_success * expected_entropy_success + likelihood_failure * expected_entropy_failure)


def sample_intensities_fixed_interval(intensities, interval):
    """
    Sample intensities at fixed intervals.
    """
    return intensities[::interval]


def choose_next_intensity_fast(prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials,
                               sampling_interval):
    """
    Choose the next intensity level using fixed interval sampling and refinement.
    """
    # Step 1: Sample intensities at fixed intervals
    sampled_intensities = sample_intensities_fixed_interval(intensities, sampling_interval)
    sampled_prior = sample_intensities_fixed_interval(prior, sampling_interval)

    # Step 2: Evaluate expected entropy for sampled intensities
    # expected_entropies = [
    #    expected_entropy(intensity, prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials)
    #    for intensity in sampled_intensities
    # ]
    _expected_entropy, best_sampled_intensity, expected_entropies = expected_entropy_after_n_trials(
        tuple(sampled_prior),
        tuple(sampled_intensities),
        k_guess, max_prob_guess,
        min_prob_guess,
        n_trials)

    # Step 3: Find the best candidate intensity
    # best_sampled_intensity = sampled_intensities[np.argmin(expected_entropies)]

    # Step 4: Refine the search using gradient-based or binary search
    best_intensity = gradient_based_search(best_sampled_intensity, prior, intensities, k_guess, max_prob_guess,
                                           min_prob_guess, n_trials,
                                           guess_range = sampling_interval * (intensities[1] - intensities[0]),
                                           )#tol = 0.000000001)
    # Alternatively: best_intensity = binary_search(best_sampled_intensity, prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials)

    return best_intensity


def gradient_based_search(initial_intensity, prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials,
                          guess_range = 40, tol = 0.001):
    """
    Refine the search using gradient-based optimization.
    """
    guess_range_extra_factor = 2
    print(initial_intensity)
    result = minimize(
        expected_entropy,
        x0=initial_intensity,
        args=(prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials),
        bounds=[(max(min(intensities), initial_intensity - guess_range*guess_range_extra_factor),
                 min(max(intensities), initial_intensity + guess_range*guess_range_extra_factor))],
        # Constrain to valid intensity range
        #method='L-BFGS-B'#,# Gradient-based optimization
        # options={'ftol': 0.001}
        method='Nelder-Mead',
        options={'xatol': 0.2, 'fatol':999999}

    )
    return result.x[0]  # Return the optimized intensity


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
# Precompute the optimal choices
# Start the timer for the precomputation
precompute_start_time = time.time()
lookup_table = precompute_optimal_choices(prior, b_values, k_guess, max_prob_guess, min_prob_guess, n_trials=4, sampling_interval=5)
precompute_end_time = time.time()
print(f'precompute_total_time = {precompute_end_time - precompute_start_time}')
# Save the lookup table to a file
with open('optimal_choices_fast_old.pkl', 'wb') as f:
    pickle.dump(lookup_table, f)
