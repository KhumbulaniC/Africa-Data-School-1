# Dash_App.py
# Import Packages
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc
import numpy as np
import pandas as pd
import pickle


# Setup
app = dash.Dash(__name__)
app.title = 'Machine Learning Model Deployment'
server = app.server

# load ML model
with open('churn_model.pickle', 'rb') as f:
    clf = pickle.load(f)
    
# App Layout 
app.layout = html.Div([
    dbc.Row([html.H3(children='Predict Banking Customer Churn')]),
    dbc.Row([
        dbc.Col(html.Label(children='Credit Score:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='CreditScore', type='number', value=619, min=0, max=10000, step=1)) 
    ]),
    dbc.Row([
        dbc.Col(html.Label("Geography:")),
        dbc.Col(dcc.Checklist(
            options=[{"label": "Germany", "value": 1},{"label": "France", "value": 2},{"label": "Spain", "value": 3},{"label": "unknown", "value": 4},],
            value='value_3',
            id="Geography",
            inline=True,
        ))
    ]),
    dbc.Row([
        dbc.Col(html.Label("Gender:")),
        dbc.Col(dcc.Checklist(
            options=[{"label": "Male", "value": 1},{"label": "France", "value": 2},],
            value='value_2',
            id="Gender",
            inline=True,
        ))
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Age:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='Age', type='number', value=42, min=0, max=100, step=1)) 
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Tenure:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='Tenure', type='text', value=8))  
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Balance:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='Balance', type='text', value=159660.80))  
    ]),  
    dbc.Row([
        dbc.Col(html.Label(children='Number of Products:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='NumOfProducts', type='text', value=3)) 
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Has Credit Card:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='HasCrCard', type='text', value=1)) 
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Is Active Member:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='IsActiveMember', type='text', value=0))  
    ]),
    dbc.Row([
        dbc.Col(html.Label(children='Estimated Salary:'), width={"order": "first"}),
        dbc.Col(dcc.Input(id='EstimatedSalary', type='text', value=113931.57))  
    ]),
    dbc.Row([dbc.Button('Submit', id='submit-val', n_clicks=0, color="primary")]),
    html.Br(),
    dbc.Row([html.Div(id='prediction output')])
    
    ], style = {'padding': '0px 0px 0px 150px', 'width': '50%'})
# Callback to produce the prediction 
@app.callback(
    Output('prediction output', 'children'),
    Input('submit-val', 'n_clicks'),
    State('CreditScore', 'value'),
    State('Geography', 'value'),
    State('Gender', 'value'),
    State('Age', 'value'),
    State('Tenure', 'value'), 
    State('Balance', 'value'),
    State('NumOfProducts', 'value'),
    State('HasCrCard', 'value'),
    State('IsActiveMember', 'value'), 
    State('EstimatedSalary', 'value')
)
   
def update_output(n_clicks, CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary):    
    x = np.array([[float(CreditScore), str(Geography), str(Gender), float(Age), float(Tenure), float(Balance), float(NumOfProducts), float(HasCrCard), float(IsActiveMember), float(EstimatedSalary)]])
    df=pd.DataFrame(data=x[0:,0:], index=[i for i in range(x.shape[0])], columns=[str(i) for i in ['CreditScore', 'Geography', 'Gender', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'HasCrCard', 'IsActiveMember', 'EstimatedSalary']])
    prediction = clf.predict(df)
    if prediction == 1:
        output = 'churn'
    elif prediction == 0:
        output = 'remain'
    return f'The prediction is that the customer is like to {output}.'
# Run the App 
if __name__ == '__main__':
    app.run_server()
