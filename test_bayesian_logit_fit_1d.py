import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy import trapz
from scipy.stats import entropy
from constants import stimuli_dBlevels, stimuli_colors
from bimodal_distribution import generate_bimodal_prior
import pickle

# Logistic function to model probability of success
def logistic_function(I, k, b):
    return 1 - 1 / (1 + np.exp(-(k * (I - b))))


# Simulate observing a success (1) or failure (0) based on light intensity
def observe_result(intensity_index, noisy_probabilities):
    p_success = noisy_probabilities[intensity_index]
    return np.random.binomial(1, p_success)


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


# Bayesian update of the posterior using Bayes' Theorem
def bayesian_all(prior, intensity_levels, intensities, results, k_guess, max_prob_guess, min_prob_guess):
    """
    Update the posterior distribution after each trial using Bayes' theorem.

    Args:
    - prior: The current prior distribution over I_threshold.
    - I: The light intensity being tested.
    - result: The observed result (0 or 1) for that intensity.

    Returns:
    - posterior: Updated posterior distribution.
    """
    # Likelihood of getting 1, given various b guesses
    likelihoods = logistic_function(np.tile(results[:, 0, None], (1, intensity_levels)), k_guess,
                                    np.tile(intensities, (results.shape[0], 1)))  * (max_prob_guess - min_prob_guess) + min_prob_guess

    # Likelihood is 1-likelihood for where results were actually 0, and mask if not tested
    likelihoods[results[:, 1] == 0] = 1 - likelihoods[results[:, 1] == 0]
    likelihoods[np.isnan(results[:, 1])] = np.nan
    likelihoods = np.ma.array(likelihoods, mask=np.isnan(likelihoods))  # Use a mask to mark the NaNs

    # Compute the unnormalized posterior: likelihood * prior
    posterior = np.prod(likelihoods, axis=0) * prior

    # Normalize the posterior to ensure it sums to 1 (since it's a probability distribution)
    posterior /= trapz(posterior, dx=(max(intensities) - min(intensities)) / intensity_levels)
    return posterior


# Function to select the next light intensity based on the current posterior
def choose_next_intensity_max_posterior(posterior, intensities):
    # Find all indices where the posterior is maximized
    max_posterior_value = np.max(posterior)
    max_indices = np.where(posterior == max_posterior_value)[0]

    # Choose the center intensity from those indices
    center_index = (max_indices[0] + max_indices[-1]) // 2
    return intensities[center_index]


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
def expected_entropy_after_n_trials(prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials=5):
    """
    Recursively compute the expected entropy of the posterior after n_trials,
    given the current prior. At each step, choose the best candidate intensity.
    """
    #if n_trials>1:
    #    print(f'expected_entropy_after_n_trials {n_trials}')
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
        expected_entropy_success, _ = expected_entropy_after_n_trials(posterior_success, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials - 1)
        #print(expected_entropy_success)
        #print(f'returned once {n_trials-1}')
        expected_entropy_failure, _ = expected_entropy_after_n_trials(posterior_failure, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials - 1)

        # Compute the expected entropy for this candidate intensity (scalar value)
        expected_entropy = np.sum(
            likelihood_success * expected_entropy_success + likelihood_failure * expected_entropy_failure)

        # Update the minimum expected entropy
        if expected_entropy < min_expected_entropy:
            min_expected_entropy = expected_entropy
            best_intensity = candidate_intensity

    return min_expected_entropy, best_intensity

#from functools import lru_cache
#@lru_cache(maxsize=None)  # Cache all results
def choose_next_intensity_minimize_entropy(prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials=5):
    """
    Choose the next intensity level to minimize the expected entropy after n_trials.
    """
    # Initialize the best intensity and minimum expected entropy
    print(f'choose_next_intensity_minimize_entropy {n_trials}')

    # Compute the expected entropy for this candidate intensity
    expected_entropy, best_intensity = expected_entropy_after_n_trials(prior, intensities, k_guess, max_prob_guess, min_prob_guess, n_trials)
    print('completed outer')

    return best_intensity

# Function to choose the next intensity based on the lookup table
def choose_next_intensity_from_lookup(lookup_table, result_sequence):
    """
    Choose the next intensity based on the precomputed lookup table and the sequence of results so far.
    """

    #print(lookup_table)
    return lookup_table[result_sequence]['intensity']

# Function to compute the 80% confidence interval
def compute_confidence_window(posterior, intensities, confidence_level=0.80):
    cumulative_posterior = np.cumsum(posterior) * (max(intensities) - min(intensities)) / len(intensities)
    lower_bound = np.searchsorted(cumulative_posterior, (1 - confidence_level) / 2)
    upper_bound = np.searchsorted(cumulative_posterior, 1 - (1 - confidence_level) / 2)
    return intensities[lower_bound], intensities[upper_bound]


# Run the Bayesian estimation process with a uniform prior
def run_bayesian_estimation(k=2, b=32, k_guess=2, noise_level=0.01, max_prob_guess=0.95, min_prob_guess=0.05, max_prob=0.95, min_prob=0.05, plot_results = True, num_trials = 5):
    # Load the lookup table from the file
    with open('optimal_choices_fast_old.pkl', 'rb') as f:
        lookup_table = pickle.load(f)
    # Initialize the sequence of results
    result_sequence = ''

    # manual data
    manual_trials = [0.4949, 0.5354, 0.6162, 0.7475, 1.0000]
    manual_trials = []
    trial_confidence_window = []
    trial_max_posterior = []

    # Define possible light intensities to test
    intensity_levels = 75
    max_intensity = 14
    intensities = np.linspace(0, max_intensity, intensity_levels)
    intensities = stimuli_dBlevels
    #intensities = np.linspace(min(stimuli_dBlevels), max(stimuli_dBlevels), intensity_levels)
    intensity_levels = len(intensities)

    # Define possible values for k and b
    #k_values = np.linspace(0.5, 3, 40)  # Example range for k
    b_values = np.linspace(0, 40, 81)  # Example range for b

    # Initialize a uniform prior distribution for the threshold
    prior = np.ones_like(b_values) / (max(b_values)-min(b_values))
    # Define the parameters for the bimodal distribution
    mean_b1 = 35  # Mean of the first Gaussian
    std_b1 = 10  # Standard deviation of the first Gaussian
    mean_b2 = 0  # Mean of the second Gaussian
    std_b2 = 10  # Standard deviation of the second Gaussian

    # Set the weight for each Gaussian
    weight_b1 = 1.1  # First Gaussian has twice the weight of the second
    weight_b2 = 1  # Second Gaussian has normal weight

    # Generate the bimodal prior distribution using the function
    prior = generate_bimodal_prior(b_values, mean_b1, std_b1, mean_b2, std_b2, weight_b1, weight_b2)
    if plot_results:
        print(intensities)
        print(stimuli_colors)
        plt.figure(figsize=(10, 6))
        plt.plot(b_values, prior,
                 label=f'Prior before any trials',
                 color='C0')
    # Model
    true_probabilities = logistic_function(intensities, k, b)
    noisy_probabilities = true_probabilities * (max_prob - min_prob) + min_prob + np.random.normal(0, noise_level,
                                                                                                   intensity_levels)
    # Clip the probabilities to be between 0 and 1
    noisy_probabilities = np.clip(noisy_probabilities, 0, 1)

    # Number of trials
    results = np.zeros((num_trials, 2))

    # Plot the true and fitted logistic curves
    x_range_true = np.linspace(min(intensities), max(intensities), len(intensities))
    x_range = np.linspace(min(intensities), max(intensities), len(b_values))
    if plot_results:
        plt.plot(intensities, logistic_function(intensities, k, b), label='True Logistic Curve', color='k', linestyle='-')
        plt.plot(intensities, noisy_probabilities, label='True Noisy Logistic Curve', color='k', linestyle='--')
    # color_index = ['C0', 'C1']
    # Conduct trials and update posterior
    for i in range(num_trials):
        ## print(f'trial {i+1} of {num_trials}')
        # Choose next intensity based on the current posterior
        # _, next_intensity_refined = choose_next_intensity_1d(prior, b_values, max_prob_guess=0.95, min_prob_guess=0.05)
        #next_intensity_refined = choose_next_intensity_minimize_entropy(prior, b_values, k_guess, max_prob_guess,
        #                                                                min_prob_guess, n_trials=num_trials-i)
        if(i<4):
            next_intensity_refined = choose_next_intensity_from_lookup(lookup_table, result_sequence)
        else:
            _, next_intensity_refined = choose_next_intensity_1d(prior, b_values, max_prob_guess=0.95,
                                                                 min_prob_guess=0.05)
        next_intensity_index = np.argmin(np.abs(intensities - next_intensity_refined))

        next_intensity = intensities[next_intensity_index]
        if manual_trials:
            next_intensity = manual_trials[i]
            next_intensity_index = np.argmin(np.abs(intensities - next_intensity))

        # Observe the result of this trial
        result = observe_result(next_intensity_index, noisy_probabilities)
        if i == 0:
            #result = 0
            a=1
        results[i] = np.asarray([next_intensity, result])
        ## print(f"Trial {i + 1}: prior_max = {b_values[np.argmax(prior)]}, Test intensity = {next_intensity:.4f}, Result = {result}, Intensity index = {next_intensity_index}, Ideal test intensity = {next_intensity_refined}")

        # Update the result sequence
        result_sequence += str(result)
        # Update the posterior distribution after the trial
        prior = bayesian_update_1d(prior, b_values, next_intensity, result,
                                k_guess, max_prob_guess, min_prob_guess)
        #lower_bound, upper_bound = compute_confidence_window(prior_b, intensities_refined, confidence_level=0.90)
        lower_bound, upper_bound = compute_confidence_window(prior, b_values, confidence_level=0.80)
        trial_confidence_window.append(upper_bound - lower_bound)
        mode_value = b_values[np.argmax(prior)]
        mean_value = np.sum(b_values * prior*(b_values[1]-b_values[0]))
        # Compute the cumulative distribution function (CDF)
        cdf = np.cumsum(prior)
        # Find the median
        median_value = b_values[np.searchsorted(cdf, 0.5)]
        trial_max_posterior.append([mode_value, mean_value, median_value])
        if i % 1 == 0 and plot_results:
            # Plot the posterior distribution after each trial
            # Compute and plot the 80% confidence window
            #idealk = k_values[np.argmax(prior[:,np.argmax(prior_b)])]
            #plt.plot(next_intensity, result, label=f'trial {i + 1}', color='C' + str(i % 10), linestyle='None',
            #         marker='o')
            plt.plot(next_intensity, result, color='C' + str(i % 10), linestyle='None',
                     marker='o')
            plt.plot(x_range, logistic_function(x_range, k_guess, b_values[np.argmax(prior)])*(max_prob_guess - min_prob_guess) + min_prob_guess,
                     label=f'Guess after trial {i + 1}', color='C' + str(i % 10), alpha=0.6)
            #plt.plot(x_range, logistic_function(x_range, idealk, b_values[np.argmax(prior_b)])*(max_prob_guess - min_prob_guess) + min_prob_guess,
            #         label=f'Guess after trial {i + 1}, k={round(idealk,3)}, b={round(b_values[np.argmax(prior_b)],3)}', color='C' + str(i % 10))
            #plt.plot(intensities_refined, prior_b,
            #         label=f'Posterior after trial {i + 1}, ({round(trial_confidence_window[-1], 4)})',
            #         color='C' + str(i % 10))
            plt.plot(b_values, prior,
                     label=f'Posterior b after trial {i + 1}, ({round(trial_confidence_window[-1], 4)})',
                     color='C' + str((i+1) % 10))
            #plt.plot(k_values, prior[:,np.argmax(prior_b)],
            #         label=f'Posterior k after trial {i + 1}, ({round(trial_confidence_window[-1], 4)})',
            #         color='C' + str(i % 10))
            plt.fill_betweenx([i / num_trials, (i + 1) / num_trials], lower_bound, upper_bound, color='C' + str((i+1) % 10),
                              alpha=0.2)

    prior = np.ones_like(intensities) / len(intensities)
    # posterior = bayesian_all(prior, intensities, results, k)
    # plt.plot(intensities, posterior, label=f'Posterior after all trials', color='k')
    if plot_results:
        plt.title(f"Posterior Distribution after Trial {i + 1}")
        plt.xlabel("Threshold Estimate (Intensity)")
        plt.ylabel("Posterior Probability")
        plt.legend()
        plt.grid()
    # plt.show()
    if 0:
        max_tests_per = 2
        responses = np.array([[np.nan, np.nan], [np.nan, np.nan],
                              [np.nan, np.nan], [np.nan, np.nan],
                              [1, np.nan], [np.nan, np.nan],
                              [1, 1], [0, np.nan],
                              [0, np.nan]])
        responses = responses.flatten()
        print(f'responses {responses}')
        results = np.zeros((intensity_levels * max_tests_per, 2))
        results[:, 0] = np.repeat(intensities, max_tests_per)
        results[:, 1] = responses
        print(results)
        print(results.shape)
        all_trials_likelihood = bayesian_all(prior, intensity_levels, intensities, results, k_guess=k_guess)

    if 0:  # double check all at once vs separately
        results = np.zeros((num_trials, 2))
        results[:, 0] = np.array([0.5, 0.75, 0.75, 0.875, 1])
        results[:, 1] = np.array([1, 1, np.nan, 0, 0])

        print(results)
        testval = logistic_function(0.5, 10, np.nan)
        print(testval)
        testval2 = logistic_function(0.5, 10, np.nan)
        print(testval * testval2)
        likelihoods = logistic_function(np.tile(results[:, 0, None], (1, intensity_levels)), k_guess,
                                        np.tile(intensities, (results.shape[0], 1)))
        all_trials_likelihood_if_results1 = np.prod(likelihoods, axis=0)
        # Modify rows of b based on values in a

        print(likelihoods)
        likelihoods[results[:, 1] == 0] = 1 - likelihoods[results[:, 1] == 0]

        likelihoods[np.isnan(results[:, 1])] = np.nan
        print(likelihoods)
        likelihoods = np.ma.array(likelihoods, mask=np.isnan(likelihoods))  # Use a mask to mark the NaNs
        all_trials_likelihood = np.prod(likelihoods, axis=0)

        print(all_trials_likelihood)
        all_trials_likelihood /= trapz(all_trials_likelihood,
                                       dx=(max(intensities) - min(intensities)) / intensity_levels)
    # plt.plot(intensities, all_trials_likelihood, 'o', label = "posterior combined trials", color = 'k')
    if plot_results and np.abs(trial_max_posterior[4][0]-b)>6:
        print(trial_max_posterior[4])
        plt.title(f'b = {b}, trial 4 result max posterior: {trial_max_posterior[4]}')
        plt.plot(np.linspace(1, num_trials, num_trials), np.array(trial_confidence_window) / 10)
        plt.show()
    else:
        plt.clf()
        plt.close()
    return np.array(trial_max_posterior)
    # Explanation:
    # a == 0 creates a boolean array where True corresponds to indices where a is 0.
    # b[a == 0] selects the rows of b where a is 0.
    # 1 - b[a == 0] computes 1 minus the original row values for those rows.


# Run the Bayesian estimation process
np.random.seed(0)
#run_bayesian_estimation(b=20,plot_results = True)

if 1:
    bs = np.arange(0, 41,10)  # bs from 0 to 40
    num_tests = 100
    num_trials = 5
    final_results = np.zeros((len(bs),num_tests, num_trials,3)) #mode mean median
    for i, b in enumerate(bs):
        for j in range(num_tests):
            print(f'b = {b} in {bs}, {j+1} of {num_tests}')
            final_results[i,j] = run_bayesian_estimation(b=b,plot_results = False, num_trials = num_trials)
            #run_bayesian_estimation(k=2, b=32, k_guess=2, noise_level=0.01, max_prob_guess=0.95, min_prob_guess=0.05, max_prob=0.95, min_prob=0.05)

    # Save final_results to a file
    np.save(f'final_results_old_bs_{bs}_num_tests_{num_tests}_trials_{num_trials}_precompute4.npy', final_results)
