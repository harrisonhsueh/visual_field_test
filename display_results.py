import numpy as np
import matplotlib.pyplot as plt
from utils import bayesian_all, logistic_function
from constants import prior, b_values, k_guess, max_prob_guess, min_prob_guess


def load_data(start_time):
    results = np.load(f'visual_field_test_results_{start_time}.npy')
    thresholds = np.load(f'visual_field_test_thresholds_{start_time}.npy')
    positions = np.load(f'visual_field_test_positions_{start_time}.npy')
    phitheta = np.load(f'visual_field_test_phitheta_{start_time}.npy')
    return results, thresholds, positions, phitheta


def plot_thresholds_heatmap(x, y, thresholds):
    plt.figure(figsize=(8, 6))
    sc = plt.scatter(x, y, c=thresholds, cmap='grey', edgecolor='k', s=100)
    plt.colorbar(sc, label="Threshold Value")
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title("Visual Field Test Thresholds Heatmap")
    plt.grid(True)
    plt.show()


def print_tests_chronologically(results):
    """
    Print test results in chronological order.
    """
    flattened_results = flatten_and_filter_results(results, sort_by_time=True)

    # Subtract the start time from all timestamps
    start_time = flattened_results[0, 0]
    flattened_results[:, 0] -= start_time

    print(flattened_results)
    print(f"Shape of flattened results: {flattened_results.shape}")


def plot_individual_position_results_with_threshold(position_index, results, thresholds, phitheta, posteriors):
    mask = results[position_index, :, 0] != 0
    x_values = results[position_index, mask, 2]
    y_values = results[position_index, mask, 3]

    plt.scatter(x_values, y_values, color='blue', label="Data Points")
    plt.plot(b_values, posteriors[position_index], label="posterior probability")
    plt.plot(b_values, logistic_function(b_values, k_guess, thresholds[position_index]),
             label="fitted logistic function")

    for j, (x, y) in enumerate(zip(x_values, y_values)):
        plt.text(x, y, str(j), fontsize=12, ha='right', va='bottom', color='black')

    plt.axvline(x=thresholds[position_index], color='r', linestyle='--', label="Threshold")
    plt.title(f'Position in degrees: {phitheta[position_index]}, index {position_index}')
    plt.ylim((-0.1, 1.1))
    plt.xlim((12, 40))
    plt.legend()
    plt.show()


def flatten_and_filter_results(results, sort_by_time=False):
    """
    Flatten the results array, filter out unused test slots, and optionally sort by time.

    Args:
        results (np.ndarray): The results array of shape (m, n, p).
        sort_by_time (bool): If True, sort the results by time_of_stimuli.

    Returns:
        np.ndarray: Flattened and filtered results of shape (k, p), where k <= m * n.
    """
    m, n, p = results.shape
    flattened_results = []

    for position_idx in range(m):
        for test_idx in range(n):
            test_data = results[position_idx, test_idx]
            time_of_stimuli, response_time, stimuli_db, test_result = test_data

            # Ignore uninitialized test slots
            if time_of_stimuli == 0 and response_time == 0 and stimuli_db == 0 and test_result == 0:
                continue

            flattened_results.append([time_of_stimuli, response_time, stimuli_db, test_result, position_idx])

    flattened_results = np.array(flattened_results)

    # Sort by time_of_stimuli if requested
    if sort_by_time:
        flattened_results = flattened_results[flattened_results[:, 0].argsort()]

    return flattened_results


def plot_response_time_histogram(results):
    """
    Plot a histogram of response times.
    """
    flattened_results = flatten_and_filter_results(results)
    print(flattened_results)
    mask = flattened_results[:, 3] == 1  # Filter for responses where stimuli_db == 1
    bins = np.linspace(0, 1.5, num=100)
    plt.hist(flattened_results[mask, 1], bins=bins, edgecolor='black')
    plt.xlabel("Response Time")
    plt.ylabel("Frequency")
    plt.title("Histogram of Response Times")
    plt.show()


def main():
    start_time = '2025-02-25_12-04-59'
    results, thresholds, positions, phitheta = load_data(start_time)
    posteriors = bayesian_all(prior, b_values, results[:-1], k_guess, max_prob_guess, min_prob_guess)

    x, y = phitheta[:, 0], phitheta[:, 1]
    plot_thresholds_heatmap(x, y, thresholds)

    print_tests_chronologically(results)

    for i in range(len(thresholds)):
        if i == np.argmin(thresholds) or thresholds[i] < 15:
            plot_individual_position_results_with_threshold(i, results, thresholds, phitheta, posteriors)

    # Option 1: Flatten once and pass to functions
    plot_response_time_histogram(results)


if __name__ == "__main__":
    main()
