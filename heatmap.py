import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

csv_data = pd.read_csv('coordinates.csv')
csv_data

img = plt.imread('thumbnail.png')
fig, ax = plt.subplots(figsize = (16, 9))
ax.imshow(img)

plot = sns.kdeplot(
        x='X_coord',
        y='Y_coord', 
        data = csv_data,
        fill = True,
        thresh = 0.2,
        alpha = 0.7,
        n_levels = 10,
        cmap = 'plasma',
        ax = ax
        )
plot = sns.histplot(
        x='X_coord',
        y='Y_coord', 
        data = csv_data,
        bins = 50,
        alpha = 0.4,
        cmap = 'plasma',
        ax = ax
        )
"""plot = sns.scatterplot(
        x='X_coord',
        y='Y_coord', 
        data = csv_data,
        legend = False,
        c = 'blue',
        edgecolor = 'blue',
        ax = ax
        )"""
fig = plot.get_figure()
fig.savefig('heatmap.png')

#plt.xlim(0,600)
#plt.ylim(338,0)
plt.show()