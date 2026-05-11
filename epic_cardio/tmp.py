def display_active_coords(data):
    try:
        import plotly.express as px
    except ImportError as exc:
        raise RuntimeError(
            "display_active_coords requires the optional 'plotly' package. "
            "Install it manually before using this scratch helper."
        ) from exc
    fig =  px.scatter_3d(data, x='X', y='T', z='Y')
    fig.update_traces(marker=dict(size=1,
                            line=dict(width=5,
                            color='Black')),
                selector=dict(mode='markers'))
    fig.show()
