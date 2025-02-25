import numpy as np
import matplotlib.pyplot as plt
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
print(thresholds.shape)
#i=1 is weird
for i, threshold in enumerate(thresholds):
    if i == np.argmin(thresholds) or thresholds[i] < 25:
        mask = results[i, :, 0] != 0  # Create a mask where results[i, j, 0] is not zero
        plt.plot(results[i, mask, 2], results[i, mask, 3], 'o')  # Apply mask
        plt.plot(thresholds[i], 0.5, 'o', label="threshold")
        plt.title(f'{phitheta[i]}')
        plt.show()
