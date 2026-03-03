import numpy as np
import plotly.graph_objects as go

from otocs.style import COLORS


def plot_2d(
    x: np.ndarray,
    y: np.ndarray,
    title: str = "2d plot",
    name: str = "data",
    showlegend: bool = True,
    width: int = 600,
    height: int = 300,
    xaxis_title: str = "x",
    yaxis_title: str = "y",
    xaxes_automargin: bool = True,
    yaxes_automargin: bool = True,
    color: str = COLORS[0],
    plot: bool = True,
    return_figure: bool = False,
):

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=name,
            marker=dict(color=color),
            showlegend=showlegend,
        )
    )
    fig.update_xaxes(
        automargin=xaxes_automargin,
    )
    fig.update_yaxes(
        automargin=yaxes_automargin,
    )
    fig.update_layout(
        width=width,
        height=height,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )
    if plot:
        fig.show()
    if return_figure:
        return fig
