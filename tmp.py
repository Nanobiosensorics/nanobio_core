import pandas as pd
import plotly.express as px

def display_active_coords(data):
    fig =  px.scatter_3d(data, x='X', y='T', z='Y')
    fig.update_traces(marker=dict(size=1,
                            line=dict(width=5,
                            color='Black')),
                selector=dict(mode='markers'))
    fig.show()