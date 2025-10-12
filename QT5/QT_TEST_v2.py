import pandas as pd    
import plotly.express as px
from dash import  Dash, dcc, html, Input, Output
 
df = pd.read_csv('C:/Users/Zyad Diab/OneDrive/Desktop/projects/data/Dash.csv')
df.head()
app =  Dash()
app.title = "Interactive Dashboard "
num_cols = df.select_dtypes(include='number').columns
app.layout = html.Div([ html.H1("Interactive Dashboard With Pie Chart"),
                       html.Label("select a value to show in pie plot"),
            dcc.Dropdown(id = 'column-dropdown',
                         options=[{'label':col,'value':col} for col in num_cols],
                         value = num_cols[0]),
            dcc.Graph(id = 'pie-chart')
                         ])
@app.callback(Output('pie-chart','figure'),
              Input ('column-dropdown','value')
              )
def update_pie(select_col):
    grouped = df.groupby('Area')[select_col].sum().reset_index()
    fig = px.pie(grouped,
                 names='Area',
                 values = select_col,
                 title=f"Distribution of {select_col} By Area",
                 hole = 0.4
 
    )
    return fig
 
if __name__ == "__main__":
    app.run(debug=True)