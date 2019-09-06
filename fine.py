"""For some reason XGBRFClassifier cannot be found from a jupyter env, so I did it elsewhere. Here."""

import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
# import seaborn as sns
from category_encoders import OrdinalEncoder
from eli5.sklearn import PermutationImportance
from ipywidgets import interact, fixed
from sklearn.metrics import roc_auc_score, roc_curve, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.pipeline import make_pipeline
# from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.utils.multiclass import unique_labels
from xgboost.sklearn import XGBRFClassifier, XGBClassifier
import chart_studio
import chart_studio.plotly as py
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import gaussian_kde
chart_studio.tools.set_credentials_file(username='pkutrich', api_key='HdAHLVwvk6R5gE6tp9td')

df = pd.read_csv('model/model_data.csv').set_index('index')


def train_val_test_split(X, y):
    trainval, test, y_trainval, y_test = train_test_split(X,
                                                          y,
                                                          stratify=y,
                                                          random_state=42,
                                                          )
    train, val, y_train, y_val = train_test_split(trainval,
                                                  y_trainval,
                                                  stratify=y_trainval,
                                                  random_state=42)
    return train, val, test, y_train, y_val, y_test

target = 'Diabetic'
y = df[target].astype(bool)
X = df.drop(columns=target)

train, val, test, y_train, y_val, y_test = train_val_test_split(X, y)
train.shape, val.shape, test.shape, y_train.shape, y_val.shape, y_test.shape

# Predict majority class as baseline.
y.value_counts(normalize=True)

roc_auc_score(y_train, np.full(y_train.shape, 0))
pipeline = make_pipeline(OrdinalEncoder(),
                         SimpleImputer())

X_train = pipeline.fit_transform(train)
X_val = pipeline.transform(val)
X_test = pipeline.transform(test)
X_train.shape, X_val.shape, X_test.shape

eval_set = [(X_train, y_train),
            (X_val, y_val)]

model = XGBRFClassifier(n_jobs=-1,
                        n_estimators=5000,
                        early_stopping_rounds=100,
                        random_state=42,
                        scale_pos_weight=15,
                        learning_rate=.005,
                        reg_lambda=.01,
                        verbosity=1)
print('fitting...')
model.fit(X_train,
          y_train,
          eval_set=eval_set,
          eval_metric='auc',
          verbose=True)

y_pred_proba = model.predict_proba(X_val)[:, 1]
print(f'Validation ROC AUC score: {roc_auc_score(y_val, y_pred_proba)}')
# curves = roc_curve(y_val, y_pred_proba)
# curve_x, curve_y = curves[0], curves[1]
# x_ = np.linspace(0, 1, len(curve_x))
# fig = plt.figure(figsize=(6, 6))
# ax = fig.gca()
# y_ = x_
# plt.plot(curve_x, curve_y, label='Model Preditions')
# plt.plot(x_, y_, color='red', label='Baseline')
# plt.title('ROC Curve for initial model.',
#           fontsize=16,
#           color='white')
# ax.set_ylabel('True Positive Rate')
# ax.set_xlabel('False Positive Rate')
# plt.legend()
# plt.savefig('roc.png');

# val = pd.read_csv('assets/val_data.csv')
# roc = pd.read_csv('assets/roc_data.csv')
#
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=roc['fpr'],
#                          y=roc['tpr'],
#                          visible=True))
#
# # Add trace for baseline ROC curve.
# x=np.linspace(0, 1, 100)
# fig.add_trace(go.Scatter(x=x,
#                          y=x,
#                          visible=True))
# # Save to chart_studio.
# py.iplot(fig, filename='roc')
print('permuting...')
permuter = PermutationImportance(model,
                                 cv='prefit',
                                 n_iter=5,
                                 scoring='roc_auc',
                                 random_state=42)
permuter.fit(X_val, y_val)
features_of_import = pd.Series(permuter.feature_importances_, val.columns).sort_values(ascending=True)
print('importance', features_of_import)
# plt.figure(figsize=(8, len(val.columns) / 2))
# features_of_import.sort_values(ascending=True).plot.barh()
# plt.savefig('pretrim.png');

print('plotting...')
fig1 = go.Figure()
fig1.add_trace(go.Bar(x=features_of_import, y=val.columns))
py.iplot(fig1, filename='features1')
# fig1

mask = features_of_import > 0
trimmed_columns = train.columns[mask]
train_trimmed = train[trimmed_columns]
val_trimmed = val[trimmed_columns]
test_trimmed = test[trimmed_columns]

pipeline1 = make_pipeline(OrdinalEncoder(),
                          SimpleImputer())

X_train_ = pipeline1.fit_transform(train_trimmed)
X_val_ = pipeline1.transform(val_trimmed)
X_test_ = pipeline1.transform(test_trimmed)

eval_set1 = [(X_train_, y_train),
            (X_val_, y_val)]

print('refitting...')
model.fit(X_train_, y_train, eval_set=eval_set1, eval_metric='auc')
y_pred_proba = model.predict_proba(X_val_)[:, 1]
print(f'XGBRFClassifier validation ROC AUC: {roc_auc_score(y_val, y_pred_proba)}')

importance = pd.Series(model.feature_importances_, val_trimmed.columns).sort_values(ascending=True)
# plt.figure(figsize=(8, len(val_trimmed.columns) / 2))
# plt.savefig('posttrim.png')
fig2 = go.Figure()
fig2.add_trace(go.Bar(x=importance, y=val_trimmed.columns))
py.iplot(fig2, filename='features2')
# fig2

