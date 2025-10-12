import pandas as pd
import plotly.express as px
from dash import Dash,dcc,Input,Output,html
 
df = pd.read_csv(r"C:\Users\Zyad Diab\OneDrive\Desktop\projects\Dash.csv")
df.head()
 
app = Dash()
app.title = "Interactive dashboard"
num_cols = df.select_dtypes(include='number').columns
app.layout = html.Div([
    html.H1("Interactive Dashboard With Pie Chart"),
    html.Label("select a value to show in pie plot"),
 
])
 
if __name__ == "__main__":
    app.run(debug=True)