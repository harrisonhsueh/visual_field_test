import numpy as np
import matplotlib.pyplot as plt
from utils import bayesian_all, logistic_function, confidence_interval_vectorized
from constants import prior, b_values, k_guess, max_prob_guess, min_prob_guess, stimuli_dBlevels
print(stimuli_dBlevels)

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


def plot_individual_position_results_with_threshold(position_index, results, thresholds, phitheta, posteriors,
                                                    confidence):
    """
        Plot individual position results with threshold, posterior, and confidence interval.

        Args:
            position_index (int): Index of the position to plot.
            results (np.ndarray): The results array of shape (m, n, p).
            thresholds (np.ndarray): The thresholds array of shape (m,).
            phitheta (np.ndarray): The positions array of shape (m, 2).
            posteriors (np.ndarray): The posterior distributions of shape (m, n).
            intensities (np.ndarray): The array of possible intensity values (shape: (n,)).
            confidence (float): The confidence level (e.g., 0.95 for 95% CI).
    """
    # Compute confidence interval
    widths, lowers, uppers = confidence_interval_vectorized(posteriors, b_values, confidence)

    mask = results[position_index, :, 0] != 0
    x_values = results[position_index, mask, 2]
    y_values = results[position_index, mask, 3]

    plt.scatter(x_values, y_values, color='blue', label="Data Points")
    for j, (x, y) in enumerate(zip(x_values, y_values)):
        plt.text(x, y, str(j), fontsize=12, ha='right', va='bottom', color='black')
    plt.plot(b_values, posteriors[position_index], label="posterior probability")
    plt.plot(b_values, logistic_function(b_values, k_guess, thresholds[position_index]),
             label="fitted logistic function")
    # Add shaded confidence interval region
    plt.fill_between(b_values, 0, 1,
                     where=(b_values >= lowers[position_index]) & (b_values <= uppers[position_index]),
                     color='gray', alpha=0.3, label=f"{int(confidence * 100)}% Confidence Interval")
    # Add text for confidence interval width and bounds
    ci_text = f"CI Width: {widths[position_index]:.2f}\nLower: {lowers[position_index]:.2f}, Upper: {uppers[position_index]:.2f}"
    plt.text(0.05, 0.5, ci_text, transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))

    plt.axvline(x=thresholds[position_index], color='r', linestyle='--', label="Threshold")
    plt.xlim((min(stimuli_dBlevels), max(stimuli_dBlevels)))
    plt.ylim((-0.1, 1.1))
    plt.xlabel("Intensity")
    plt.ylabel("Probability")
    plt.title(f'Position in degrees: {phitheta[position_index]}, index {position_index}')
    plt.legend(loc='upper left')
    plt.grid(True)
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
        if i == np.argmin(thresholds) or thresholds[i] < 99:
            plot_individual_position_results_with_threshold(i, results, thresholds, phitheta, posteriors, 0.95)

    # Option 1: Flatten once and pass to functions
    plot_response_time_histogram(results)


if __name__ == "__main__":
    main()
