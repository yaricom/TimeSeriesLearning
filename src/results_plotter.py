# -*- coding: utf-8 -*-
"""
Created on Thu Aug  4 21:17:05 2016

Renders output

@author: yaric
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

file_name = 'vp_10_08_11_45' # 'vp_31_07_00_21' # 'vp_04_08_16_40' # 
# the path to look for files
results_path = '../results/best/{}.{}'
# the number of bins
bins_count=100

# read predictions
y_pred_df = pd.read_csv(results_path.format(file_name, 'csv'))

print 'Results:\n%s\n' % y_pred_df.describe()

y_pred = np.array(y_pred_df)

# make histograms
y1_hist, _ = np.histogram(y_pred[:,0], bins=bins_count)
y2_hist, _ = np.histogram(y_pred[:,1], bins=bins_count)
y3_hist, _ = np.histogram(y_pred[:,2], bins=bins_count)

# draw scatter
x = np.arange(bins_count)

y1_plot = plt.scatter(x, np.log10(y1_hist), marker='o', color='b')
y2_plot = plt.scatter(x, np.log10(y2_hist), marker='o', color='r')
y3_plot = plt.scatter(x, np.log10(y3_hist), marker='o', color='g')

plt.grid(color='black', linestyle='-')
plt.title(file_name)
plt.legend((y1_plot, y2_plot, y3_plot), ('y1','y2','y3'),
           scatterpoints=1, loc='upper right')
# save figure
plt.savefig(results_path.format(file_name, 'png'), dpi=72)
# show figure
plt.show()