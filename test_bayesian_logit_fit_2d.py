import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from numpy import trapz
from scipy.stats import entropy


# Logistic function to model probability of success
def logistic_function(I, k, b):
    return 1 - 1 / (1 + np.exp(-(k * (I - b))))


# Simulate observing a success (1) or failure (0) based on light intensity
def observe_result(intensity_index, noisy_probabilities):
    p_success = noisy_probabilities[intensity_index]
    return np.random.binomial(1, p_success)


# Bayesian update of the posterior using Bayes' Theorem
def bayesian_update_1d(prior, intensities, intensity, result, intensity_levels, k_guess, max_prob_guess, min_prob_guess):
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
    posterior /= trapz(posterior, dx=(max(intensities) - min(intensities)) / intensity_levels)
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


def bayesian_update_2d(prior_2d, intensity, result, k_grid, b_grid, max_prob_guess, min_prob_guess):
    """
       Update the 2D posterior distribution over k and b using Bayes' theorem.

       Args:
       - prior_2d: The current 2D prior distribution over k and b.
       - intensity: The intensity being tested.
       - result: The observed result (0 or 1).
       - k_grid: Grid of possible k values.
       - b_grid: Grid of possible b values.
       - max_prob_guess: Maximum probability of success.
       - min_prob_guess: Minimum probability of failure.

       Returns:
       - posterior_2d: Updated 2D posterior distribution.
       """
    # Compute the likelihood for all k and b values
    likelihood = logistic_function(intensity, k_grid, b_grid) * (max_prob_guess - min_prob_guess) + min_prob_guess

    # Adjust likelihood based on the result
    if result == 1:
        likelihood = likelihood
    else:
        likelihood = 1 - likelihood

    # Compute the unnormalized posterior
    posterior_2d = likelihood * prior_2d

    # Normalize the posterior
    posterior_2d /= np.trapz(np.trapz(posterior_2d, b_grid[0], axis=1), k_grid[:,0])

    return posterior_2d


def marginalize_over_k(posterior_2d, k_values, b_values):
    """
    Marginalize over k to obtain the posterior distribution for b.

    Args:
    - posterior_2d: The 2D posterior distribution over k and b.
    - k_values: Array of possible k values.
    - b_values: Array of possible b values.

    Returns:
    - posterior_b: Marginal posterior distribution for b.
    """
    # Integrate over k to get the marginal posterior for b
    posterior_b = np.trapz(posterior_2d, k_values, axis=0)

    # Normalize the marginal posterior
    posterior_b /= np.trapz(posterior_b, b_values)

    return posterior_b


# Function to select the next light intensity based on the current posterior
def choose_next_intensity_b(posterior, intensities):
    # Find all indices where the posterior is maximized
    max_posterior_value = np.max(posterior)
    max_indices = np.where(posterior == max_posterior_value)[0]

    # Choose the center intensity from those indices
    center_index = (max_indices[0] + max_indices[-1]) // 2
    return intensities[center_index]


def choose_next_intensity_2d(posterior, k_values, b_values, k_grid, b_grid, max_prob_guess, min_prob_guess):
    """
    Select the next intensity level to test based on the marginal posterior for b.

    Args:
    - posterior_b: The marginal posterior distribution for b.
    - b_values: Array of possible b values.

    Returns:
    - The intensity level that maximizes the expected information gain.
    """
    # Compute the expected information gain for each intensity
    # (This part is similar to the previous implementation but uses the marginal posterior for b)
    expected_information_gain = np.zeros(len(b_values))

    posterior_b = marginalize_over_k(posterior, k_values, b_values)
    # Compute the entropy of the current marginal posterior
    entropy_prior = entropy(posterior_b)

    # Loop over all possible intensities
    for i, intensity in enumerate(b_values):
        # Compute the likelihood of success and failure at this intensity
        likelihood_success = logistic_function(intensity, k_grid, b_grid) * (max_prob_guess - min_prob_guess) + min_prob_guess
        likelihood_failure = 1 - likelihood_success

        # Compute the expected likelihood by weighting with the prior
        expected_likelihood_success = np.sum(likelihood_success * posterior)
        expected_likelihood_failure = np.sum(likelihood_failure * posterior)


        # Compute the posterior for both possible outcomes (success and failure)
        posterior_success = bayesian_update_2d(posterior, intensity, 1, k_grid, b_grid, max_prob_guess, min_prob_guess)
        posterior_failure = bayesian_update_2d(posterior, intensity, 0, k_grid, b_grid, max_prob_guess, min_prob_guess)

        posterior_success_b = marginalize_over_k(posterior_success, k_values, b_values)
        posterior_failure_b = marginalize_over_k(posterior_failure, k_values, b_values)

        # Compute the entropy of the posterior distributions
        entropy_success = entropy(posterior_success_b)
        entropy_failure = entropy(posterior_failure_b)
        # Expected information gain is the reduction in entropy
        expected_information_gain[i] = entropy_prior - (
            expected_likelihood_success * entropy_success + expected_likelihood_failure * entropy_failure
        )

    # Choose the intensity with the maximum expected information gain
    max_info_gain_index = np.argmax(expected_information_gain)
    return b_values[max_info_gain_index]


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

    # Compute the entropy of the current prior
    entropy_prior = entropy(prior)

    # Loop over all possible intensities to test
    for i, intensity in enumerate(intensities):
        # Compute the likelihood of success and failure for all possible b values
        likelihood_success_all_b = logistic_function(intensity, k_guess, intensities) * (max_prob_guess - min_prob_guess) + min_prob_guess
        likelihood_failure_all_b = 1 - likelihood_success_all_b

        # Compute the expected likelihood by weighting with the prior
        expected_likelihood_success = np.sum(likelihood_success_all_b * prior)
        expected_likelihood_failure = np.sum(likelihood_failure_all_b * prior)

        # Compute the posterior for both possible outcomes (success and failure)
        posterior_success = bayesian_update_1d(prior, intensities, intensity, 1, len(intensities), k_guess, max_prob_guess, min_prob_guess)
        posterior_failure = bayesian_update_1d(prior, intensities, intensity, 0, len(intensities), k_guess, max_prob_guess, min_prob_guess)

        # Compute the entropy of the posterior distributions
        entropy_success = entropy(posterior_success)
        entropy_failure = entropy(posterior_failure)

        # Expected information gain is the reduction in entropy
        expected_information_gain[i] = entropy_prior - (
            expected_likelihood_success * entropy_success + expected_likelihood_failure * entropy_failure
        )

    # Choose the intensity with the maximum expected information gain
    max_info_gain_index = np.argmax(expected_information_gain)
    return intensities[max_info_gain_index]


# Function to compute the 80% confidence interval
def compute_confidence_window(posterior, intensities, confidence_level=0.80):
    cumulative_posterior = np.cumsum(posterior) * (max(intensities) - min(intensities)) / len(intensities)
    lower_bound = np.searchsorted(cumulative_posterior, (1 - confidence_level) / 2)
    upper_bound = np.searchsorted(cumulative_posterior, 1 - (1 - confidence_level) / 2)
    return intensities[lower_bound], intensities[upper_bound]


# Run the Bayesian estimation process with a uniform prior
def run_bayesian_estimation(k=2, b=12, k_guess=2, noise_level=0.01, max_prob_guess=0.95, min_prob_guess=0.05, max_prob=0.95, min_prob=0.05):
    # Store the results of each trial
    trials = []
    # manual data
    manual_trials = [0.4949, 0.5354, 0.6162, 0.7475, 1.0000]
    manual_trials = []
    trial_confidence_window = []

    # Define possible light intensities to test
    intensity_levels = 75
    max_intensity = 14
    intensities = np.linspace(0, max_intensity, intensity_levels)
    refine_factor = 5
    intensities_refined = np.linspace(0, max_intensity, intensity_levels * refine_factor)

    # Define possible values for k and b
    k_values = np.linspace(0.5, 3, 40)  # Example range for k
    b_values = np.linspace(0, 14, 820)  # Example range for b

    # Create a 2D grid for k and b
    k_grid, b_grid = np.meshgrid(k_values, b_values, indexing='ij')

    # Initialize a uniform 2D prior
    prior_2d = np.ones_like(k_grid) / (len(k_values) * len(b_values))
    prior = prior_2d
    prior_b = marginalize_over_k(prior, k_values, b_values)
    # Initialize a uniform prior distribution for the threshold
    #prior = np.ones_like(intensities_refined) / len(intensities) / refine_factor

    # Model
    true_probabilities = logistic_function(intensities, k, b)
    noisy_probabilities = true_probabilities * (max_prob - min_prob) + min_prob + np.random.normal(0, noise_level,
                                                                                                   intensity_levels)
    # Clip the probabilities to be between 0 and 1
    noisy_probabilities = np.clip(noisy_probabilities, 0, 1)

    # Number of trials
    num_trials = 200
    results = np.zeros((num_trials, 2))

    plt.figure(figsize=(10, 6))
    # Plot the true and fitted logistic curves
    x_range_true = np.linspace(min(intensities), max(intensities), intensity_levels)
    x_range = np.linspace(min(intensities), max(intensities), intensity_levels * refine_factor)
    plt.plot(x_range, logistic_function(x_range, k, b), label='True Logistic Curve', color='k', linestyle='-')
    plt.plot(x_range_true, noisy_probabilities, label='True Noisy Logistic Curve', color='k', linestyle='--')
    # color_index = ['C0', 'C1']
    # Conduct trials and update posterior
    for i in range(num_trials):
        # Choose next intensity based on the current posterior
        # next_intensity_refined = choose_next_intensity_1d(prior, intensities_refined, max_prob_guess=0.95, min_prob_guess=0.05)
        next_intensity_refined = choose_next_intensity_2d(prior, k_values, b_values, k_grid, b_grid, max_prob_guess, min_prob_guess)

        next_intensity_index = np.argmin(np.abs(intensities - next_intensity_refined))
        next_intensity = intensities[next_intensity_index]
        if manual_trials:
            next_intensity = manual_trials[i]
            next_intensity_index = np.argmin(np.abs(intensities - next_intensity))

        # Observe the result of this trial
        result = observe_result(next_intensity_index, noisy_probabilities)
        if i == 0:
            result = 0
        results[i] = np.asarray([next_intensity, result])
        print(f"Trial {i + 1}: Test intensity = {next_intensity:.4f}, Result = {result}")

        # Store the trial intensity
        trials.append(next_intensity)
        # Update the posterior distribution after the trial
        #prior = bayesian_update_1d(prior, intensities_refined, next_intensity, result, intensity_levels * refine_factor,
        #                        k_guess, max_prob_guess, min_prob_guess)
        prior = bayesian_update_2d(prior, next_intensity, result, k_grid, b_grid, max_prob_guess=0.95, min_prob_guess=0.05)
        prior_b = marginalize_over_k(prior, k_values, b_values)
        #lower_bound, upper_bound = compute_confidence_window(prior_b, intensities_refined, confidence_level=0.90)
        lower_bound, upper_bound = compute_confidence_window(prior_b, b_values, confidence_level=0.90)
        trial_confidence_window.append(upper_bound - lower_bound)
        if i % 11 == 0:
            # Plot the posterior distribution after each trial
            # Compute and plot the 80% confidence window
            idealk = k_values[np.argmax(prior[:,np.argmax(prior_b)])]
            #plt.plot(next_intensity, result, label=f'trial {i + 1}', color='C' + str(i % 10), linestyle='None',
            #         marker='o')
            plt.plot(next_intensity, result, color='C' + str(i % 10), linestyle='None',
                     marker='o')
            #plt.plot(x_range, logistic_function(x_range, k_guess, intensities_refined[np.argmax(prior)]),
            #         label=f'Guess after trial {i + 1}', color='C' + str(i % 10))
            plt.plot(x_range, logistic_function(x_range, idealk, b_values[np.argmax(prior_b)])*(max_prob_guess - min_prob_guess) + min_prob_guess,
                     label=f'Guess after trial {i + 1}, k={round(idealk,3)}, b={round(b_values[np.argmax(prior_b)],3)}', color='C' + str(i % 10))
            #plt.plot(intensities_refined, prior_b,
            #         label=f'Posterior after trial {i + 1}, ({round(trial_confidence_window[-1], 4)})',
            #         color='C' + str(i % 10))
            plt.plot(b_values, prior_b,
                     label=f'Posterior b after trial {i + 1}, ({round(trial_confidence_window[-1], 4)})',
                     color='C' + str(i % 10))
            plt.plot(k_values, prior[:,np.argmax(prior_b)],
                     label=f'Posterior k after trial {i + 1}, ({round(trial_confidence_window[-1], 4)})',
                     color='C' + str(i % 10))
            plt.fill_betweenx([i / num_trials, (i + 1) / num_trials], lower_bound, upper_bound, color='C' + str(i % 10),
                              alpha=0.2)

    prior = np.ones_like(intensities) / len(intensities)
    # posterior = bayesian_all(prior, intensities, results, k)
    # plt.plot(intensities, posterior, label=f'Posterior after all trials', color='k')
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
    plt.plot(np.linspace(1, num_trials, num_trials), np.array(trial_confidence_window) / 10)
    plt.show()
    # Explanation:
    # a == 0 creates a boolean array where True corresponds to indices where a is 0.
    # b[a == 0] selects the rows of b where a is 0.
    # 1 - b[a == 0] computes 1 minus the original row values for those rows.


# Run the Bayesian estimation process
np.random.seed(0)
run_bayesian_estimation()
