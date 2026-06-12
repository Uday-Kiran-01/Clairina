# clairina/visualization.py
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_spend_trends(df, channels=None, window=7):
    if channels is None:
        channels = [c for c in df.columns if c.startswith("spend_") and "_lag" not in c]
    df2 = df.set_index("date")[channels].rolling(window).mean().reset_index()
    fig = px.line(df2, x="date", y=channels, title=f"{window}-day rolling spend by channel")
    fig.update_layout(legend_title_text="Channel")
    return fig

def plot_coeff_bar(params_dict):
    # params_dict keys like 'log1p_spend_google'
    items = {k: v for k,v in params_dict.items() if k.startswith("log1p_spend_")}
    df = pd.DataFrame([{"channel": k.replace("log1p_",""), "beta": v} for k,v in items.items()])
    df = df.sort_values("beta", ascending=False)
    fig = px.bar(df, x="channel", y="beta", title="Model coefficients (log1p features)")
    return fig

def plot_marginal_bar(marginals):
    # marginals: dict channel -> {mean_spend, marginal_rev_per_$}
    import pandas as pd
    df = pd.DataFrame.from_dict(marginals, orient="index").reset_index().rename(columns={"index":"channel"})
    df = df.sort_values("marginal_rev_per_$", ascending=False)
    fig = px.bar(df, x="channel", y="marginal_rev_per_$", title="Marginal revenue per $ (at mean spend)")
    return fig

def plot_allocation_pie(allocation_dict, title="Allocation"):
    df = pd.DataFrame(list(allocation_dict.items()), columns=["channel","spend"])
    fig = px.pie(df, names="channel", values="spend", title=title)
    return fig
