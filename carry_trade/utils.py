import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import quandl
import requests
from canvasapi import Canvas
from plotly import colors
from plotly.subplots import make_subplots
from scipy import stats

# =============================================================================
# Credentials
# =============================================================================

quandl.ApiConfig.api_key = os.getenv("QUANDL_API_KEY")


# =============================================================================
# Canvas
# =============================================================================


def download_files(filename_frag: str):
    """Downloads files from Canvas with `filename_frag` in filename."""

    url = os.getenv("CANVAS_URL")
    token = os.getenv("CANVAS_TOKEN")

    course_id = 33395
    canvas = Canvas(url, token)
    course = canvas.get_course(course_id)

    existing_files = [p.name for p in Path().iterdir() if filename_frag in p.name]

    for f in course.get_files():
        if filename_frag in f.filename and f.filename not in existing_files:
            try:
                print(f.filename, f.id)
                file = course.get_file(f.id)
                file.download(file.filename)
            except:
                print(f.filename, f.id, "not downloaded")


# =============================================================================
# Fetch Data
# =============================================================================


def fetch_ticker(
    dataset_code: str, query_params: Dict = None, database_code: str = "EOD"
):
    """Fetches price data for a single ticker."""

    url = f"https://www.quandl.com/api/v3/datasets/{database_code}/{dataset_code}.json"

    params = dict(api_key=os.getenv("QUANDL_API_KEY"))
    if query_params is not None:
        params = dict(**params, **query_params)

    r = requests.get(url, params=params)

    if r.status_code != 200:
        print(r.text)

    else:
        dataset = r.json()["dataset"]

        df = pd.DataFrame(
            dataset["data"], columns=[c.lower() for c in dataset["column_names"]]
        )
        df["ticker"] = dataset["dataset_code"]

        return df.sort_values("date")


def get_yc(ticker):
    """Fetches yield curves and saves to csv."""

    fetch_ticker(ticker, database_code="YC").to_csv(
        f"data/df_{ticker}.csv", index=False
    )


def get_fx(currency: str, start_date: str):
    """Fetches fx rates and saves to csv."""

    fetch_ticker(currency, database_code="CUR").to_csv(
        f"data/df_fx_{currency}.csv", index=False
    )


# =============================================================================
# Load Data
# =============================================================================


def reindex_and_interpolate(df, yc: bool = False):
    """Reindexes and linearly interpolates to have data on every date in the
    sequence.
    """
    df.date = pd.to_datetime(df.date)
    df = df.set_index("date")
    df = df.reindex(pd.date_range(df.index.min(), df.index.max())).iloc[:, :-1]
    df = df.interpolate(axis=0)
    if yc:
        df = df.interpolate(axis=1)

    return df


def load_yc(tickers: List) -> Dict:
    """Loads yield curves."""

    dfs_yc = {t: pd.read_csv(f"s3://finm33150/carry-trade/df_{t}.csv") for t in tickers}

    for k in dfs_yc:
        dfs_yc[k] = reindex_and_interpolate(dfs_yc[k], yc=True)

    return dfs_yc


def load_fx(currencies: List) -> Dict:
    """Load foreign exchange rates."""

    dfs_fx = {
        c: pd.read_csv(f"s3://finm33150/carry-trade/df_fx_{c}.csv") for c in currencies
    }

    for k in dfs_fx:
        dfs_fx[k] = reindex_and_interpolate(dfs_fx[k])

    return dfs_fx


def load_libor(libors: List) -> Dict:
    """Loads 3 month libors."""

    dfs_libor = {
        l: pd.read_csv(f"s3://finm33150/carry-trade/df_libor_{l}.csv") for l in libors
    }
    dfs_libor["JPY3MTD156N"].value = (
        dfs_libor["JPY3MTD156N"].value.replace(".", None).astype(float)
    )

    for k in dfs_libor:
        dfs_libor[k] = reindex_and_interpolate(dfs_libor[k])

    return dfs_libor


# =============================================================================
# Curve Calculations
# =============================================================================


def get_spot_curve(spot_rates_curve: pd.Series):
    tenors = []
    for i in spot_rates_curve.index:
        n, per = i.split("-")
        n = int(n)
        tenor = n / 12 if per == "month" else n
        tenors.append(tenor)

    spot_rates_curve.index = tenors

    spot_rates_curve.name = "rate"
    spot_rates_curve = spot_rates_curve.astype(float) / 100

    return spot_rates_curve.to_frame()


def compute_zcb_curve(spot_rates_curve):
    zcb_rates = spot_rates_curve.copy()
    for curve in spot_rates_curve.columns:
        spot = spot_rates_curve[curve]
        for tenor, spot_rate in spot.iteritems():
            if tenor > 0.001:
                times = np.arange(tenor - 0.25, 0, step=-0.25)[::-1]
                coupon_half_yr = 0.25 * spot_rate
                z = np.interp(
                    times, zcb_rates[curve].index.values, zcb_rates[curve].values
                )  # Linear interpolation
                preceding_coupons_val = (coupon_half_yr * np.exp(-z * times)).sum()
                zcb_rates[curve][tenor] = (
                    -np.log((1 - preceding_coupons_val) / (1 + coupon_half_yr)) / tenor
                )
    return zcb_rates


def get_zcb_interp(zcb: pd.Series, n_days: int = 7):

    times = zcb.index.values - n_days / 365
    zcb_interp = np.interp(times, zcb.index.values, zcb.rate)

    zcb_interp = pd.Series(zcb_interp, index=times)
    zcb_interp.name = "rate"

    return zcb_interp.to_frame()


def bond_price(zcb, coupon_rate, tenor):
    times = np.arange(tenor, 0, step=-0.25)[::-1]
    if times.shape[0] == 0:
        p = 1.0
    else:
        r = np.interp(times, zcb.index.values, zcb.rate.values)  # Linear interpolation
        p = np.exp(-tenor * r[-1]) + 0.25 * coupon_rate * np.exp(-r * times).sum()
    return p


# =============================================================================
# Strategy
# =============================================================================


def run_strategy(
    yc_L: str,
    fx_B: str,
    fx_L: str,
    libor: str,
    leverage: float,
    date_range: pd.date_range,
    dfs_yc: Dict,
    dfs_fx: Dict,
    dfs_libor: Dict,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Runs carry trade strategy.

    Args:
        yc: yield curve
        fx: foreign exchange rate
        libor: 3 month libor rate
        H: Home market - USD
        B: Borrowing market
        L: Lending market
        Lev: Leverage
        K: Capital
        0: Beginning of period
        1: End of period

    Returns:
        Returns dataframe, profit components dataframe
    """

    yc_slice = dict(THA=slice(1, 8), IDN=slice(0, 5), ROU=slice(0, 4))[yc_L]

    returns = []
    profits = []
    for d0, d1 in zip(date_range[:-1], date_range[1:]):
        fx_B0 = dfs_fx[fx_B].loc[d0].rate
        fx_B1 = dfs_fx[fx_B].loc[d1].rate
        fx_L0 = dfs_fx[fx_L].loc[d0].rate
        fx_L1 = dfs_fx[fx_L].loc[d1].rate
        yc_L0 = dfs_yc[yc_L].loc[d0]
        yc_L1 = dfs_yc[yc_L].loc[d1]
        libor_L0 = dfs_libor[libor].loc[d0].value + 0.50

        src_L0 = get_spot_curve(yc_L0[yc_slice])
        zcb_L0 = compute_zcb_curve(src_L0)

        src_L1 = get_spot_curve(yc_L1[yc_slice])
        zcb_L1 = compute_zcb_curve(src_L1)

        zcb_L1_interp = get_zcb_interp(zcb_L1, (d1 - d0).days)

        K_H0 = float(2e6)
        K_B0 = K_H0 * fx_B0
        Lev_B0 = K_B0 / (1 - leverage) * leverage
        I_B1 = Lev_B0 * libor_L0 / 100 * (d1 - d0).days / 360

        V_B0 = K_B0 + Lev_B0
        V_L0 = (V_B0 * fx_L0) / fx_B0
        N_L0 = V_L0 / np.exp(-zcb_L0.rate.values[-1] * zcb_L0.index.values[-1])
        V_L1 = N_L0 * np.exp(
            -zcb_L1_interp.rate.values[-1] * zcb_L1_interp.index.values[-1]
        )

        V_B1 = V_L1 / fx_L1 * fx_B1
        K_B1 = V_B1 - Lev_B0 - I_B1
        K_H1 = K_B1 / fx_B1
        r_H1 = np.log(K_H1 / K_H0)

        returns.append((d1, r_H1))

        profits.append(
            (
                d1,
                fx_B0,
                fx_B1,
                fx_L0,
                fx_L1,
                zcb_L0.rate.values[-1],
                zcb_L0.index.values[-1],
                zcb_L1_interp.rate.values[-1],
                zcb_L1_interp.index.values[-1],
                V_L0,
                N_L0,
                V_L1,
                V_B0,
                V_B1,
                K_H0,
                K_H1,
                r_H1,
            )
        )

    df_ret = pd.DataFrame(returns, columns=["date", "per_return"]).set_index("date")
    df_profit = pd.DataFrame(
        profits,
        columns=[
            "d1",
            "fx_B0",
            "fx_B1",
            "fx_L0",
            "fx_L1",
            "zcb_L0",
            "zcb_L0t",
            "zcb_L1",
            "zcb_L1t",
            "V_L0",
            "N_L0",
            "V_L1",
            "V_B0",
            "V_B1",
            "K_H0",
            "K_H1",
            "r_H1",
        ],
    ).set_index("d1")

    df_ret.name = f"{fx_B},{yc_L}"

    return df_ret, df_profit


# =============================================================================
# Charts
# =============================================================================

COLORS = colors.qualitative.T10

IS_labels = [
    ("obs", lambda x: f"{x:>7d}"),
    ("min:max", lambda x: f"{x[0]:>0.4f}:{x[1]:>0.3f}"),
    ("mean", lambda x: f"{x:>7.4f}"),
    ("std", lambda x: f"{x:>7.4f}"),
    ("skewness", lambda x: f"{x:>7.4f}"),
    ("kurtosis", lambda x: f"{x:>7.4f}"),
]


def get_moments_annotation(
    s: pd.Series,
    xref: str,
    yref: str,
    x: float,
    y: float,
    xanchor: str,
    title: str,
    labels: List,
) -> go.layout.Annotation:
    """Calculates summary statistics for a series and returns and
    Annotation object.
    """
    moments = list(stats.describe(s.to_numpy()))
    moments[3] = np.sqrt(moments[3])

    sharpe = s.mean() / s.std()

    return go.layout.Annotation(
        text=(
            f"<b>sharpe: {sharpe:>8.4f}</b><br>"
            + ("<br>").join(
                [f"{k[0]:<9}{k[1](moments[i])}" for i, k in enumerate(labels)]
            )
        ),
        align="left",
        showarrow=False,
        xref=xref,
        yref=yref,
        x=x,
        y=y,
        bordercolor="black",
        borderwidth=0.5,
        borderpad=2,
        bgcolor="white",
        xanchor=xanchor,
        yanchor="top",
    )


def make_components_chart(
    yc_L: str,
    fx_B: str,
    fx_L: str,
    libor: str,
    leverage: float,
    date_range: pd.date_range,
    dfs_yc: Dict,
    dfs_fx: Dict,
    dfs_libor: Dict,
) -> go.Figure:

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"5-Year Yield: {yc_L}",
            f"FX Rate: {fx_L}:{fx_B}",
            f"3 Month Libor: {libor}",
            f"FX Rate: {fx_B}:USD",
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
        specs=[
            [{"secondary_y": True}, {"secondary_y": True}],
            [{"secondary_y": False}, {"secondary_y": True}],
        ],
    )

    # Lend market yield
    # =================
    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_yc[yc_L].loc[date_range]["5-year"],
            line=dict(width=1, color=COLORS[0]),
            name=yc_L,
        ),
        row=1,
        col=1,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_yc[yc_L].loc[date_range]["5-year"].pct_change() * 100,
            line=dict(width=1, color=COLORS[1], dash="dot"),
            name=yc_L,
        ),
        row=1,
        col=1,
        secondary_y=True,
    )

    # Borrow market fx
    # =================
    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_fx[fx_B].loc[date_range].rate,
            line=dict(width=1, color=COLORS[0]),
            name=fx_B,
        ),
        row=2,
        col=2,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_fx[fx_B].loc[date_range].rate.pct_change() * 100,
            line=dict(width=1, color=COLORS[1], dash="dot"),
            name=fx_B,
        ),
        row=2,
        col=2,
        secondary_y=True,
    )

    # Borrow market funding cost
    # =================
    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_libor[libor].loc[date_range].value,
            line=dict(width=1, color=COLORS[0]),
            name=libor,
        ),
        row=2,
        col=1,
    )

    # Lend market fx cost
    # =================
    fx_BL = (
        dfs_fx[fx_L].loc[date_range].loc[date_range].rate
        / dfs_fx[fx_B].loc[date_range].rate
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=fx_BL,
            line=dict(width=1, color=COLORS[0]),
            name=fx_L,
        ),
        row=1,
        col=2,
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=fx_BL.pct_change() * 100,
            line=dict(width=1, color=COLORS[1], dash="dot"),
            name=fx_L,
        ),
        row=1,
        col=2,
        secondary_y=True,
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor="grey", mirror=True)
    fig.update_yaxes(
        showline=True, linewidth=1, linecolor="grey", mirror=True, tickformat="0.1f"
    )

    fig.update_layout(
        title_text=(
            f"Weekly Carry Trade: Borrow {fx_B}, Lend {yc_L}"
            "<br>Underlying Securities: "
            f"{date_range.min().strftime('%Y-%m-%d')}"
            f" - {date_range.max().strftime('%Y-%m-%d')}"
        ),
        showlegend=False,
        height=600,
        font=dict(size=10),
        margin=dict(l=50, r=10, b=40, t=90),
        yaxis3=dict(tickformat="0.3f"),
    )

    for i in fig["layout"]["annotations"]:
        i["font"]["size"] = 12

    return fig


def make_returns_chart(df_ret: pd.DataFrame) -> go.Figure:

    fx_B, yc_L = df_ret.name.split(",")

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"Weekly Returns",
            f"Returns Distribution",
            f"Cumulative Returns",
            f"Q/Q Plot",
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    # Returns Distribution
    returns = pd.cut(df_ret.per_return, 50).value_counts().sort_index()
    midpoints = returns.index.map(lambda interval: interval.right).to_numpy()
    norm_dist = stats.norm.pdf(
        midpoints, loc=df_ret.per_return.mean(), scale=df_ret.per_return.std()
    )

    fig.add_trace(
        go.Scatter(
            x=df_ret.index,
            y=df_ret.per_return * 100,
            line=dict(width=1, color=COLORS[0]),
            name="return",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=df_ret.index,
            y=df_ret.per_return.cumsum() * 100,
            line=dict(width=1, color=COLORS[0]),
            name="cum. return",
        ),
        row=2,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=[interval.mid for interval in returns.index],
            y=returns / returns.sum() * 100,
            name="pct. of returns",
            marker=dict(color=COLORS[0]),
        ),
        row=1,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=[interval.mid for interval in returns.index],
            y=norm_dist / norm_dist.sum() * 100,
            name="normal",
            line=dict(width=1, color=COLORS[1]),
        ),
        row=1,
        col=2,
    )

    # Q/Q Data
    returns_norm = (
        (df_ret.per_return - df_ret.per_return.mean()) / df_ret.per_return.std()
    ).sort_values()
    norm_dist = pd.Series(
        list(map(stats.norm.ppf, np.linspace(0.001, 0.999, len(df_ret.per_return)))),
        name="normal",
    )

    fig.append_trace(
        go.Scatter(
            x=norm_dist,
            y=returns_norm,
            name="return norm.",
            mode="markers",
            marker=dict(color=COLORS[0], size=3),
        ),
        row=2,
        col=2,
    )

    fig.add_trace(
        go.Scatter(
            x=norm_dist,
            y=norm_dist,
            name="norm.",
            line=dict(width=1, color=COLORS[1]),
        ),
        row=2,
        col=2,
    )

    fig.add_annotation(
        text=(f"{df_ret.per_return.cumsum()[-1] * 100:0.2f}"),
        xref="paper",
        yref="y3",
        x=0.465,
        y=df_ret.per_return.cumsum()[-1] * 100,
        xanchor="left",
        showarrow=False,
        align="left",
    )

    fig.add_annotation(
        get_moments_annotation(
            df_ret.per_return,
            xref="paper",
            yref="paper",
            x=0.81,
            y=0.23,
            xanchor="left",
            title="Returns",
            labels=IS_labels,
        ),
        font=dict(size=6, family="Courier New, monospace"),
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

    fig.update_layout(
        title_text=(
            f"Weekly Carry Trade: Borrow {fx_B}, Lend {yc_L}"
            "<br>Returns: "
            f"{df_ret.index.min().strftime('%Y-%m-%d')}"
            f" - {df_ret.index.max().strftime('%Y-%m-%d')}"
        ),
        showlegend=False,
        height=600,
        font=dict(size=10),
        margin=dict(l=50, r=50, b=50, t=100),
        yaxis=dict(tickformat="0.1f"),
        yaxis3=dict(tickformat="0.1f"),
        yaxis2=dict(tickformat="0.1f"),
        yaxis4=dict(tickformat="0.1f"),
        xaxis2=dict(tickformat="0.1f"),
        xaxis4=dict(tickformat="0.1f"),
    )

    for i in fig["layout"]["annotations"]:
        i["font"]["size"] = 12

    fig.update_annotations(font=dict(size=10))

    return fig
