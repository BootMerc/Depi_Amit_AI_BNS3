# dash_app_with_filters.py
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output

# ---- Load dataset ----
df = pd.read_csv("C:/Users/6y6yy/Desktop/DEPI_ONL_AI/test/cleaned_data.csv")



# Age bins
df['age_bin'] = pd.cut(df['age'], bins=range(0, 100, 10), right=False)
labels = [str(interval) for interval in df['age_bin'].cat.categories]

# ---- Initialize Dash App ----
app = Dash(__name__)

# ---- Layout ----
app.layout = html.Div([
    html.H1("Ford GoBike Dashboard", style={'textAlign': 'center'}),

    # --- Global Filters ---
    html.Div([
        html.Label("Filter by User Type:"),
        dcc.Dropdown(
            id="user_filter",
            options=[{"label": u, "value": u} for u in df["user_type"].dropna().unique()],
            value=None,
            placeholder="Select user type"
        ),
        html.Label("Filter by Gender:"),
        dcc.Dropdown(
            id="gender_filter",
            options=[{"label": g, "value": g} for g in df["member_gender"].dropna().unique()],
            value=None,
            placeholder="Select gender"
        ),
    ], style={"width": "40%", "margin": "auto"}),

    html.Hr(),

    dcc.Tabs([
        # --- Tab 1: Users ---
        dcc.Tab(label="Users", children=[
            dcc.Graph(id="user_type_chart"),
            dcc.Graph(id="bike_share_chart"),
            dcc.Graph(id="gender_chart"),
            dcc.Graph(id="age_bin_chart"),
            dcc.Graph(id="age_box"),
            
        ]),

        # --- Tab 2: Trips ---
        dcc.Tab(label="Trips", children=[
            dcc.Graph(id="trip_minutes"),
            dcc.Graph(id="trip_hours"),
        ]),

        # --- Tab 3: Insights ---
        dcc.Tab(label="Insights", children=[
            dcc.Graph(id="age_gender_box"),
            dcc.Graph(id="age_user_type_box"),
        ]),
    ])
])



# ---- Callbacks (Update all graphs based on filters) ----
@app.callback(
    [Output("user_type_chart", "figure"),
     Output("bike_share_chart", "figure"),
     Output("age_bin_chart", "figure"),
     Output("gender_chart", "figure"),
     Output("age_gender_box", "figure"),
     Output("trip_minutes", "figure"),
     Output("trip_hours", "figure"),
     Output("age_box", "figure"),
     Output("age_user_type_box", "figure")],
    [Input("user_filter", "value"),
     Input("gender_filter", "value")]
)
def update_all(selected_user, selected_gender):
    # Filter data
    dff = df.copy()
    if selected_user:
        dff = dff[dff["user_type"] == selected_user]
    if selected_gender:
        dff = dff[dff["member_gender"] == selected_gender]

    # 1. User type histogram
    fig1 = px.histogram(
        dff, x="user_type", color="user_type",
        title="Distribution of User Types"
    )
    fig1.update_layout(bargap=0.2, template="plotly_white")

    # 2. Bike share program
    fig2 = px.histogram(
        dff, x="bike_share_for_all_trip", color="bike_share_for_all_trip",
        title="Users in the Bike Share for All Program"
    )
    fig2.update_layout(bargap=0.2, template="plotly_white")

    # 3. Age bins
    fig3 = px.histogram(
        dff, x="age", color="age_bin",
        category_orders={"age_bin": labels},
        title="Age Distribution by Age Groups"
    )

    # 4. Gender distribution
    fig4 = px.histogram(
        dff, x="member_gender",
        title="Distribution by Gender"
    )
    fig4.update_layout(template="plotly_white")

    # 5. Age vs gender (box plot)
    fig5 = px.box(
        dff, x="member_gender", y="age", color="member_gender",
        title="Age Distribution by Gender"
    )

    # 6. Trip duration in minutes
    fig6 = px.histogram(
        dff, x="duration_min", nbins=145,
        title="Trip Duration in Minutes"
    )
    fig6.update_xaxes(range=[0, 60], title="Duration (minutes)")
    fig6.update_yaxes(title="Frequency")
    fig6.update_layout(template="plotly_white", bargap=0.05)

    # 7. Trip duration in hours (custom bins)
    bin_require = [0, 0.25, 0.5, 0.75, 1]
    counts, edges = np.histogram(dff["duration_hour"], bins=bin_require)
    fig7 = go.Figure()
    fig7.add_trace(go.Bar(
        x=[f"{edges[i]:.2f} - {edges[i+1]:.2f}" for i in range(len(edges)-1)],
        y=counts,
        text=counts,
        textposition="outside",
        marker_color="skyblue"
    ))
    fig7.update_layout(
        title="Trip Duration in Hours",
        xaxis_title="Duration in Hours",
        yaxis_title="Number of Customers",
        template="plotly_white"
    )

    # 8. Age box plot
    fig8 = px.box(
        dff, x="age",
        title="Age Distribution (Box Plot)"
    )
    fig8.update_layout(template="plotly_white")

    # 9. Age vs user type (box plot)
    fig9 = px.box(
        dff, x="user_type", y="age", color="user_type",
        title="Age by User Type"
    )

    return fig1, fig2, fig3, fig4, fig5, fig6, fig7, fig8, fig9


# ---- Run App ----
if __name__ == "__main__":
    app.run(debug=True)
