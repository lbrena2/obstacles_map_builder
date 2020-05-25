from pandas import DataFrame
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

obstacle_map_coords = np.stack(np.meshgrid(
        np.linspace(-1, 1, 20),
        np.linspace(-1, 1, 20),
    )).reshape([2, -1]).T

poses = np.array([[x+1, 0] for x in range(0, 10)])

obstacle_map_values_mean = [[0] for _ in range(obstacle_map_coords.shape[0])]

obstacle_map_values_median = [np.nanmedian(prediction_values) for prediction_values in obstacle_map_values_mean]
print(obstacle_map_values_median)

ax1 = plt.subplot(3, 1, 1)

sns.heatmap(np.array(obstacle_map_values_mean).reshape((20, 20)),
                 linewidth=0.5,
                 vmin=0,
                 vmax=1,
                 center=0.5,
                 ax=ax1,
                 cmap='RdYlGn')

ax2 = plt.subplot(3, 1, 2, sharex=ax1)

sns.heatmap(np.array(obstacle_map_values_median).reshape((20, 20)),
                 linewidth=0.5,
                 vmin=0,
                 vmax=1,
                 center=0.5,
                 ax=ax2,
                 cmap='RdYlGn')

ax3 = plt.subplot(3, 1, 3)
ax3.plot(poses[:, 0], poses[:, 1], 'b.',
             alpha=0.1,
             markersize=7)

plt.show()


