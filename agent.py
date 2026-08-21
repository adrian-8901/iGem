import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Dash, dcc, html, Input, Output

app = Dash(__name__)

# ---  PARAMETERS ---
total_distance = 20  
dx = 1.0              
dt = 0.4        
D = 0.8               
decay_rate = 0.02     
max_time = 20.0
total_steps = int(max_time / dt) 
M_0 = 1000.0                
initial_concentration = 10.0

pK_a = 6.0    
K_0 = 0.03    

app.layout = html.Div(style={'backgroundColor': '#000000', 'color': '#ffffff', 'padding': '20px', 'fontFamily': 'sans-serif', 'minHeight': '100vh'}, children=[
    html.H1("Simulation", style={'textAlign': 'center', 'marginBottom': '30px', 'color': '#ffffff'}),
    
    
    html.Div(style={'display': 'flex', 'justifyContent': 'space-around', 'alignItems': 'center', 'backgroundColor': '#111111', 'padding': '20px', 'borderRadius': '8px', 'marginBottom': '20px', 'gap': '10px', 'border': '1px solid #333333'}, children=[
        
        html.Div(style={"flex": "1"}, children=[
            html.Label("1. Medium pH", style={'fontWeight': 'bold', 'color': '#ff4d4d', 'marginBottom': '10px', 'display': 'block'}),
            dcc.Slider(
                id='ph-slider', min=1.0, max=8.0, step=0.1, value=7.4,
                marks={
                    2: {'label': 'Stomach (2.0)', 'style': {'color': '#ffffff'}},
                    5.5: {'label': 'Duodenum (5.5)', 'style': {'color': '#ffffff'}},
                    7.4: {'label': 'Colon (7.4)', 'style': {'color': '#ffffff'}},
                },
                allow_direct_input=False,
                tooltip={"placement": "bottom", "style": {"backgroundColor": "#1e1e1e", "border": "0", "outline": "none"}},
                updatemode="drag"

            )
        ]),
        
        html.Div(style={"flex": "1"}, children=[
            html.Label("2. Surface Area-to-Volume Ratio", style={'fontWeight': 'bold', 'color': '#3399ff', 'marginBottom': '10px', 'display': 'block'}),
            dcc.Slider(
                id='particles-slider', min=1, max=1000, step=10, value=1,
                marks={
                    1: {'label': 'Capsule (1)', 'style': {'color': '#ffffff'}},
                    250: {'label': '250', 'style': {'color': '#ffffff'}},
                    500: {'label': '500', 'style': {'color': '#ffffff'}},
                    750: {'label': '750', 'style': {'color': '#ffffff'}},
                    1000: {'label': '1,000', 'style': {'color': '#ffffff'}}
                },
                allow_direct_input=False,
                updatemode="drag"

            )
        ]),
        
        html.Div(style={"flex": "1"}, children=[
            html.Label("3. Temperature", style={'fontWeight': 'bold', 'color': "#ffee33", 'marginBottom': '10px', 'display': 'block'}),
            dcc.Slider(
                id="temp-slider", min=0, max=100, step=1, value=37,
                marks={i: {'label': f"{i}°C", "style": {"color": "#ffffff", "margin-left": "5px"}} for i in range(0, 101, 50)},
                allow_direct_input=False,
                updatemode="drag"
            )
        ]
        )
    
    ]),
    
    html.Div([
        dcc.Graph(id='simulation-graphs')
    ])
])



@app.callback(
    Output('simulation-graphs', 'figure'),
    Input('ph-slider', 'value'),
    Input('particles-slider', 'value'),
)
def run_realtime_loop(medium_ph, n_particles):
    alpha = 2.0 / 3.0
        
    K_dissolution = K_0 * (1 + 10 ** (medium_ph - pK_a)) * (n_particles ** (1/3))
    
    concentration = np.zeros(total_distance)
    time_history = []
    mass_history = []
    
    pill_mass = M_0
    
    for step in range(total_steps + 1):
        t = round(step * dt, 1)
        
        if pill_mass > 0:
            dM_dt = K_dissolution * (pill_mass ** alpha)
            pill_mass = max(0.0, pill_mass - dM_dt * dt)
            concentration[0] = initial_concentration * (pill_mass / M_0)
        else:
            concentration[0] = 0.0
            
        time_history.append(t)
        mass_history.append(pill_mass)
        
        new_concentration = concentration.copy()
        for x in range(1, total_distance - 1):
            rate_of_spread = D * (concentration[x+1] - 2*concentration[x] + concentration[x-1]) / (dx**2)
            new_concentration[x] += (rate_of_spread - (decay_rate * concentration[x])) * dt
        concentration = new_concentration

    fig = make_subplots(
        rows=1, cols=2, 
        subplot_titles=("Medication Diffusion Profile", "Pill Mass Decay Over Time")
    )
    
    fig.add_trace(
        go.Scatter(x=np.arange(total_distance), y=concentration, mode='lines', 
                   name='Diffusion Curve', line=dict(color='crimson', width=3)), 
        row=1, col=1
    )
    
    fig.add_trace(
        go.Scatter(x=time_history, y=mass_history, mode='lines', 
                   name='Mass Decay', line=dict(color='royalblue', width=3)), 
        row=1, col=1
    )
    
    fig.update_layout(
        template="plotly_dark", height=550, showlegend=False,
        paper_bgcolor='#000000', plot_bgcolor='#000000',
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    fig.update_yaxes(range=[-0.5, 11], row=1, col=1, title_text="Medication Concentration")
    fig.update_xaxes(range=[0, total_distance], row=1, col=1, title_text="Distance into Tissue")
    fig.update_yaxes(range=[-50, M_0 + 50], row=1, col=2, title_text="Remaining Pill Mass (mg)")
    
    return fig

if __name__ == '__main__':
    app.run(debug=True)