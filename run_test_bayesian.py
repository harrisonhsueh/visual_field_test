import numpy as np
import matplotlib.pyplot as plt

# Load final_results from the file
bs = np.arange(0, 41,10)  # bs from 0 to 40
num_tests = 100
final_results = np.load(f'final_results_bs_{bs}_num_tests_{num_tests}_trials_{5}_precompute4.npy')
final_results[:,:,:,1] = final_results[:,:,:,1]/2
# Subtract bs from all columns of final_results
#error = final_results - bs[:, np.newaxis, np.newaxis, np.newaxis]
# Index for the p dimension
test_index = 4

# List of i indices to plot
b_indices = [0, 1, 2, 3, 4] # Replace with your desired i indices
mode_mean_median = [0, 1, 2]
mode_mean_median_index = 2
# Compute global bin edges that cover all i indices
all_data = np.concatenate([final_results[i, :, test_index,mode_mean_median_index] for i in b_indices])
global_counts, global_bin_edges = np.histogram(all_data, bins=20)

# Compute histograms for each i index using the global bin edges
hist_data = []
for b_index in b_indices:
    counts, _ = np.histogram(final_results[b_index, :, test_index,mode_mean_median_index], bins=global_bin_edges)
    hist_data.append(counts)

# Plot histograms with offsets and narrower bars
plt.figure(figsize=(10, 6))
bar_width = (global_bin_edges[1] - global_bin_edges[0]) / (len(b_indices) + 1)  # Adjust bar width
for idx, b_index in enumerate(b_indices):
    # Offset each histogram by shifting the bin edges
    offset = idx * bar_width
    plt.bar(global_bin_edges[:-1] + offset, hist_data[idx], width=bar_width, alpha=0.7, label=f'i={b_index}')

# Add labels and legend
plt.xlabel('Error Value')
plt.ylabel('Frequency')
plt.title(f'Histogram of Error for p={test_index} (Offset and Narrower Bars)')
plt.legend()

b_index = 2
for b_index in b_indices:
    # Compute global bin edges that cover all mode_mean_median indices
    all_data = np.concatenate([final_results[b_index, :, test_index,i] for i in mode_mean_median])

    global_counts, global_bin_edges = np.histogram(all_data, bins=20)

    # Compute histograms for each i index using the global bin edges
    hist_data = []
    for mode_mean_median_index in mode_mean_median:
        counts, _ = np.histogram(final_results[b_index, :, test_index,mode_mean_median_index], bins=global_bin_edges)
        hist_data.append(counts)

    # Plot histograms with offsets and narrower bars
    plt.figure(figsize=(10, 6))
    bar_width = (global_bin_edges[1] - global_bin_edges[0]) / (len(b_indices) + 1)  # Adjust bar width
    for idx, mode_mean_median_index in enumerate(mode_mean_median):
        # Offset each histogram by shifting the bin edges
        offset = idx * bar_width
        plt.bar(global_bin_edges[:-1] + offset, hist_data[idx], width=bar_width, alpha=0.7, label=f'mode_mean_median={mode_mean_median}')

    # Add labels and legend
    plt.xlabel('Error Value')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Error for b={b_index} (Offset and Narrower Bars)')
    plt.legend()

plt.show()

# Define the range of delta values
deltas = np.arange(2, 11)
import sys
np.set_printoptions(threshold=sys.maxsize)
print(final_results)

# Initialize dictionaries to store the results
max_results = {i: [] for i in range(len(bs))}
mean_results = {i: [] for i in range(len(bs))}
percentile_results90 = {i: [] for i in range(len(bs))}
percentile_results95 = {i: [] for i in range(len(bs))}

# Iterate over each b (i.e., each final_results[i])
for i in range(len(bs)):
    b = bs[i]
    for delta in deltas:
        indices = []
        for row in final_results[i]:
            # Find the index where all subsequent values are within delta of b
            valid_index = None
            for idx in range(len(row)):
                if all(np.abs(row[idx:] - b) <= delta):
                    valid_index = idx
                    break
            # Append the valid index or np.nan if no valid index is found
            indices.append(valid_index if valid_index is not None else np.nan)
        # Handle empty indices
        if len(indices) > 0:
            max_results[i].append(np.max(indices))
            mean_results[i].append(np.mean(indices))
            percentile_results90[i].append(np.percentile(indices, 90))
            percentile_results95[i].append(np.percentile(indices, 95))
        else:
            # If no indices satisfy the condition, append NaN
            max_results[i].append(np.nan)
            mean_results[i].append(np.nan)
            percentile_results90[i].append(np.nan)
            percentile_results95[i].append(np.nan)
print(max_results)
# Plot the results
plt.figure(figsize=(12, 8))
for i in range(len(bs)):
    # Plot mean index vs delta
    plt.plot(deltas, max_results[i], label=f'b = {bs[i]} (Max)', color = f'C{i%10}', alpha = 0.5)
    # Plot 95th percentile index vs delta
    plt.plot(deltas, percentile_results95[i], '--', label=f'b = {bs[i]} (95th Percentile)', color = f'C{i%10}', alpha = 0.5)
    # Plot 90th percentile index vs delta
    plt.plot(deltas, percentile_results90[i], ':', label=f'b = {bs[i]} (90th Percentile)', color = f'C{i%10}', alpha = 0.5)

plt.xlabel('Delta')
plt.ylabel('Index')
plt.title('Index vs Delta for each b (Mean and 90th Percentile)')
plt.legend()
plt.grid(True)
plt.show()