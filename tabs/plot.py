import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
from sklearn.metrics import confusion_matrix


class PlotFig(object):
    def __init__(self):
        self.fig = self.create_fig()

    def create_fig(self):
        print("creating figure...")
        val = pd.read_csv(open('/assets/val_data.csv', 'rb'))
        roc = pd.read_csv(open('/assets/roc_data.csv', 'rb'))
        hist_data = val['y_pred_proba1']
        kde = gaussian_kde(hist_data)
        y = kde.pdf(np.linspace(min(hist_data), max(hist_data)))
        y_true = val['y_val'].astype(int)

        # Generate list of probabilities.
        line_space = np.linspace(min(hist_data), max(hist_data), 100)

        # List all confusion matrix outputs for annotating heatmap with slider.
        al, bl, cl, dl = [], [], [], []
        for j in line_space:
            y_pred = np.array(hist_data) > j
            confusion = confusion_matrix(y_true, y_pred)
            al.append(confusion[0][1])
            bl.append(confusion[0][0])
            cl.append(confusion[1][0])
            dl.append(confusion[1][1])

        fig = make_subplots(1, 3)

        fig.update_layout(go.Layout(height=475,
                                    width=1050,
                                    showlegend=False),
                          annotations=(go.layout.Annotation(text='{}'.format(al[0]),
                                                            x=.565,
                                                            y=.775,
                                                            showarrow=False,
                                                            font=dict(color='black'),
                                                            xref='paper',
                                                            yref='paper',
                                                            ),
                                       go.layout.Annotation(text='{}'.format(bl[0]),
                                                            x=.425,
                                                            y=.775,
                                                            showarrow=False,
                                                            font=dict(color='black'),
                                                            xref='paper',
                                                            yref='paper',
                                                            ),
                                       go.layout.Annotation(text='{}'.format(cl[0]),
                                                            x=.425,
                                                            y=.225,
                                                            showarrow=False,
                                                            font=dict(color='black'),
                                                            xref='paper',
                                                            yref='paper',
                                                            ),
                                       go.layout.Annotation(text='{}'.format(dl[0]),
                                                            x=.565,
                                                            y=.225,
                                                            showarrow=False,
                                                            font=dict(color='black'),
                                                            xref='paper',
                                                            yref='paper',
                                                            ),
                                       go.layout.Annotation(text='Distribution of probabilities',
                                                            x=0.025,
                                                            y=1.11,
                                                            showarrow=False,
                                                            font=dict(color='black',
                                                                      size=16),
                                                            xref='paper',
                                                            yref='paper'
                                                            ),
                                       go.layout.Annotation(text='Confusion matrix',
                                                            x=.5,
                                                            y=1.11,
                                                            showarrow=False,
                                                            font=dict(color='black',
                                                                      size=16),
                                                            xref='paper',
                                                            yref='paper'
                                                            ),
                                       go.layout.Annotation(text='ROC curve',
                                                            x=.9,
                                                            y=1.11,
                                                            showarrow=False,
                                                            font=dict(color='black',
                                                                      size=16),
                                                            xref='paper',
                                                            yref='paper'
                                                            ),
                                       go.layout.Annotation(
                                           text='Manipulating Probability Threshold to Optimize True Positive Rate.',
                                           x=.5,
                                           y=1.325,
                                           showarrow=False,
                                           font=dict(color='black',
                                                     size=18),
                                           xref='paper',
                                           yref='paper'
                                           ),
                                       go.layout.Annotation(
                                           text='Move the slider to the right to classify more people as non diabetic.',
                                           x=.5,
                                           y=-.25,
                                           showarrow=False,
                                           font=dict(color='black',
                                                     size=12),
                                           xref='paper',
                                           yref='paper'
                                           ),
                                       go.layout.Annotation(
                                           text='The green bar on the left frame indicates the probability threshold for being diabetic. See how many people are correctly classified in the middle frame.',
                                           x=.5,
                                           y=-.35,
                                           showarrow=False,
                                           font=dict(color='black',
                                                     size=12),
                                           xref='paper',
                                           yref='paper'
                                           ),
                                       go.layout.Annotation(
                                           text='The right frame plots the false positive rate on the x-axis vs the true positive rate on the y-axis',
                                           x=.5,
                                           y=-.45,
                                           showarrow=False,
                                           font=dict(color='black',
                                                     size=12),
                                           xref='paper',
                                           yref='paper'
                                           ),
                                       )
                          )

        # Make histogram portion of distplot.
        hist = go.Histogram(x=hist_data,
                            xbins={'start': min(hist_data),
                                   'size': .0001,
                                   'end': max(hist_data)})

        # Make probability distribution function portion of distplot.
        dist = go.Scatter(x=np.linspace(min(hist_data),
                                        max(hist_data)),
                          y=y)

        # Add them to fig3.
        fig.add_trace(hist,
                      row=1,
                      col=1)

        fig.add_trace(dist,
                      row=1,
                      col=1)

        # Add traces for ROC curve
        fig.add_trace(go.Scatter(x=roc['fpr'],
                                 y=roc['tpr'],
                                 visible=True),
                      row=1,
                      col=3)

        # Add trace for baseline ROC curve.
        x = np.linspace(0, 1, 100)
        fig.add_trace(go.Scatter(x=x,
                                 y=x,
                                 visible=True),
                      row=1,
                      col=3)

        # Generate traces for slider animation.
        for i, step in enumerate(line_space):
            # Add virticle line to distplot at each probability point.
            fig.add_trace(go.Scatter(mode='lines',
                                     line=dict(color='#0fff10',
                                               width=1.5),
                                     x=[step, step],
                                     y=[0, 1250],
                                     connectgaps=True,
                                     visible=False),
                          row=1,
                          col=1)

            # Create new confusion matrix with current probability threshold.
            y_pred = np.array(hist_data) > step
            confusion = confusion_matrix(y_true, y_pred)

            # Add new heatmap for each probability threshold.
            fig.add_trace(go.Heatmap(z=confusion,
                                     x=['Predicted Non Diab', 'Predicted Diab'],
                                     y=['Diab', 'Non Diab'],
                                     colorscale='earth',
                                     visible=False,
                                     showscale=False),
                          row=1,
                          col=2)

            # Add a point along ROC curve for each probability threshold.
            # len(roc) is 12 times longer than len(line_space).
            # Plot every 12th value in roc for each step.
            j = i * 12
            fig.add_trace(go.Scatter(x=[roc['fpr'].iloc[j]],
                                     y=[roc['tpr'].iloc[j]],
                                     visible=False),
                          row=1,
                          col=3)

        # Make starting traces visible.
        fig.data[4].visible = True
        fig.data[5].visible = True
        fig.data[6].visible = True

        # Make steps for slider.
        steps = []
        for i, x, a, b, c, d in zip(range(1, 101), line_space, al, bl, cl, dl):
            step = dict(method='update',
                        value=x,
                        label='{:.4f}'.format(x),
                        # We want to make visible three traces per step.
                        # Traces we want are from index 4 - 304.
                        # The traces are grouped in threes, i.e. 4-6, 7-9.
                        # For each step, create a binary list with `True` at the indexes we want to be visible.
                        args=[{'visible': [True if t in range(4) or t in range(i * 3, (i * 3) + 3) else False for t in
                                           range(len(fig.data))]},
                              {'annotations[0].text': a,
                               'annotations[1].text': b,
                               'annotations[2].text': c,
                               'annotations[3].text': d}
                              ])
            steps.append(step)

        # Add slider to layout.
        sliders = [go.layout.Slider(active=0,
                                    steps=steps,
                                    y=-.4,
                                    x=.25,
                                    currentvalue=dict(prefix="Probability: "),
                                    lenmode='fraction',
                                    len=.5)]

        fig.layout.sliders = sliders

        # Return chart.
        return fig
