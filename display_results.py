import numpy as np
import matplotlib.pyplot as plt
from utils import bayesian_all, logistic_function
from constants import prior, b_values, k_guess, max_prob_guess, min_prob_guess
#import seaborn as sns

start_time = '2025-02-25_12-04-59'
results = np.load(f'visual_field_test_results_{start_time}.npy')
# thresholds is shape (m,)
thresholds = np.load(f'visual_field_test_thresholds_{start_time}.npy')
# thresholds is shape (m,2) with x y coordinates in each positions[i]
positions = np.load(f'visual_field_test_positions_{start_time}.npy')
phitheta = np.load(f'visual_field_test_phitheta_{start_time}.npy')

# Extract x and y coordinates
x, y = positions[:, 0], positions[:, 1]
x, y = phitheta[:, 0], phitheta[:, 1]
print(thresholds)
# Create scatter plot
plt.figure(figsize=(8, 6))
sc = plt.scatter(x, y, c=thresholds, cmap='grey', edgecolor='k', s=100)
plt.colorbar(sc, label="Threshold Value")
plt.xlabel("X Position")
plt.ylabel("Y Position")
plt.title("Visual Field Test Thresholds Heatmap")
plt.grid(True)
plt.show()
print(f'shape of thresholds {thresholds.shape}')
posteriors = bayesian_all(prior, b_values, results, k_guess, max_prob_guess, min_prob_guess)
#i=1 is weird
for i, threshold in enumerate(thresholds):
    if i == np.argmin(thresholds) or thresholds[i] < 95:
        mask = results[i, :, 0] != 0  # Create a mask where results[i, j, 0] is not zero
        x_values = results[i, mask, 2]  # X-coordinates
        y_values = results[i, mask, 3]  # Y-coordinates



        plt.scatter(x_values, y_values, color='blue', label="Data Points")  # Consistent color
        plt.plot(b_values, posteriors[i], label="posterior probability")
        plt.plot(b_values, logistic_function(b_values, k_guess, thresholds[i]), label="fitted logistic function")
        # Add labels for each point
        for j, (x, y) in enumerate(zip(x_values, y_values)):
            plt.text(x, y, str(j), fontsize=12, ha='right', va='bottom', color='black')

        plt.axvline(x=thresholds[i], color='r', linestyle='--', label="Threshold")  # Vertical line
        plt.title(f'Position in degrees: {phitheta[i]}')
        plt.ylim((-0.1, 1.1))
        plt.xlim((12,40))
        plt.legend()
        plt.show()

flattened_results = results.reshape((results.shape[0]*results.shape[1],results.shape[2]))
#flattened_results = results[-1]
mask = (flattened_results[:, 0] != 0) & (flattened_results[:, 3] == 1)  # Create a mask where results[i, j, 0] is not zero
print(f' clicked {np.sum(mask)} times')
print(f' shown {np.sum(flattened_results[:, 0] != 0)} stimuli')

bins = np.linspace(0, 1.5, num=100)  # Creates 50 bins from 0 to 1.5 for finer resolution

plt.hist(flattened_results[mask, 1], bins=bins, edgecolor='black')  # Apply mask and finer bins
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Histogram of response times")
plt.show()