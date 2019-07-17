import plotly.plotly as py
import plotly.graph_objs as go
import plotly.figure_factory as ff

import numpy as np
import pandas as pd

val = pd.read_csv('model/val_data.csv')
roc = pd.read_csv('model/roc_data.csv')

hist_data = [val['y_pred_proba1']]
group_labels = ['Distribution of predictions by probability.']

fig = ff.create_distplot(hist_data, group_labels, bin_size=.0001)
py.plot(fig)
