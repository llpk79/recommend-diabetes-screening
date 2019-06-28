from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

from pickle import load, loads
import numpy as np
import pandas as pd

from app import app

pickled_column_values = load(open('model/models/column_values.p', 'rb'))
column_values = loads(pickled_column_values)

columns = list(column_values.keys())
style = {'padding': '1.5em'}

layout = html.Div([
    dcc.Markdown("""
        ### Predict

        Select from the dropdown menus to see if a medical screening for diabetes might is recommended.
    
    """), 

    html.Div([
        dcc.Markdown(f'###### {columns[0]}'), 
        dcc.Dropdown(
            id=columns[0], 
            options=[{'label': purpose, 'value': purpose} for purpose in column_values[columns[0]]], 
            value=column_values[columns[0]][0]
        ), 
    ], style=style), 

    html.Div([
        dcc.Markdown(f'###### {columns[1]}'), 
        dcc.Dropdown(
            id=columns[1], 
            options=[{'label': purpose, 'value': purpose} for purpose in column_values[columns[1]]], 
            value=column_values[columns[1]][0]
        ), 
    ], style=style), 

    html.Div([
        dcc.Markdown(f'###### {columns[2]}'), 
        dcc.Dropdown(
            id=columns[2], 
            options=[{'label': purpose, 'value': purpose} for purpose in column_values[columns[2]]], 
            value=column_values[columns[2]][0]
        ), 
    ], style=style),

    html.Div([
        dcc.Markdown(f'###### {columns[3]}'), 
        dcc.Dropdown(
            id=columns[3], 
            options=[{'label': purpose, 'value': purpose} for purpose in column_values[columns[3]]], 
            value=column_values[columns[3]][0]
        ), 
    ], style=style),

    html.Div([
        dcc.Markdown(f'###### {columns[4]}'), 
        dcc.Dropdown(
            id=columns[4], 
            options=[{'label': purpose, 'value': purpose} for purpose in column_values[columns[4]]], 
            value=column_values[columns[4]][0]
        ), 
    ], style=style),

    html.Div([
        dcc.Markdown(f'###### {columns[5]}'), 
        dcc.Dropdown(
            id=columns[5], 
            options=[{'label': purpose, 'value': purpose} for purpose in column_values[columns[5]]], 
            value=column_values[columns[5]][0]
        ), 
    ], style=style),

    html.Div([
        dcc.Markdown(f'###### {columns[6]}'), 
        dcc.Dropdown(
            id=columns[6], 
            options=[{'label': purpose, 'value': purpose} for purpose in column_values[columns[6]]], 
            value=column_values[columns[6]][0]
        ), 
    ], style=style),

    html.Div([
        dcc.Markdown(f'###### {columns[7]}'), 
        dcc.Dropdown(
            id=columns[7], 
            options=[{'label': purpose, 'value': purpose} for purpose in column_values[columns[7]]], 
            value=column_values[columns[7]][0]
        ), 
    ], style=style),

    html.Div([
        dcc.Markdown(f'###### {columns[8]}'), 
        dcc.Dropdown(
            id=columns[8], 
            options=[{'label': purpose, 'value': purpose} for purpose in column_values[columns[8]]], 
            value=column_values[columns[8]][0]
        ), 
    ], style=style),

    html.Div([
        dcc.Markdown(f'###### {columns[9]}'), 
        dcc.Dropdown(
            id=columns[9],
            options=[{'label': purpose, 'value': purpose} for purpose in column_values[columns[9]]], 
            value=column_values[columns[9]][0]
        ), 
    ], style=style),

    html.Div([
        dcc.Markdown(f'###### {columns[10]}'), 
        dcc.Dropdown(
            id=columns[10], 
            options=[{'label': purpose, 'value': purpose} for purpose in column_values[columns[10]]], 
            value=column_values[columns[10]][0]
        ), 
    ], style=style),

    html.Div([
        dcc.Markdown(f'###### {columns[11]}'), 
        dcc.Dropdown(
            id=columns[11], 
            options=[{'label': purpose, 'value': purpose} for purpose in column_values[columns[11]]], 
            value=column_values[columns[11]][0]
        ), 
    ], style=style),

    dcc.Markdown('### Prediction'), 
    html.Div(id='prediction-content', style={'marginBottom': '5em'}), 
])

@app.callback(
    Output('prediction-content', 'children'),
    [Input('Age', 'value'),
     Input('Income', 'value'),
     Input('Over Median Income', 'value'),
     Input('Total Household', 'value'),
     Input('Overweight', 'value'),
     Input('Good Health', 'value'),
     Input('Fruit', 'value'),
     Input('Sleep Hrs', 'value'),
     Input('Insurance', 'value'),
     Input('Recent Dr Visit', 'value'),
     Input('Smoker', 'value'),
     Input('Alcohol', 'value')])

def predict(Age, 
            Income,
            Over_median_income,
            Total_Household, 
            Overweight, 
            Good_Health, 
            Fruit, 
            Sleep_Hrs, 
            Insurance, 
            Recent_Dr_Visit, 
            Smoker, 
            Alcohol
           ):
    print('in function')

    df = pd.DataFrame(
        columns=columns, 
        data=[[Age, 
               Income, 
               Total_Household, 
               Over_median_income,
               Overweight, 
               Good_Health, 
               Fruit, 
               Sleep_Hrs, 
               Insurance, 
               Recent_Dr_Visit, 
               Smoker, 
               Alcohol
               ]])
    
    pickled_pipeline = load(open('model/models/pipeline.p', 'rb'))
    pipeline = loads(pickled_pipeline)
    
    pickled_model = load(open('model/models/estimator.p', 'rb'))
    model = loads(pickled_model)
    print(df)

    df_ = pipeline.transform(df)
#     df_ = pipeline.named_steps['ordinalencoder'].transform(df)
#     df_ = pipeline.named_steps['simpleimputer'].transform(df_)
    y_pred_proba = model.predict_proba(df_)[:, 1] > .4808
    print(f'Prediction: {y_pred_proba}')
#     if y_pred_proba:
    return 'You may benefit from a medical diabetic screening.'
#     else:
#         return 'The model does not recommend diabetic screening.'
