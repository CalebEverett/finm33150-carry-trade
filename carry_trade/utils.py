import gzip
import os
from typing import Dict, List

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

    for f in course.get_files():
        if filename_frag in f.filename:
            print(f.filename, f.id)
            file = course.get_file(f.id)
            file.download(file.filename)


# =============================================================================
# Fetching Data
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


# =============================================================================
# Data Preparation
# =============================================================================


def compute_zcb_curve(spot_rates_curve):
    zcb_rates = spot_rates_curve.copy()
    for curve in spot_rates_curve.columns:
        spot = spot_rates_curve[curve]
        for tenor, spot_rate in spot.iteritems():
            if tenor > 0.001:
                times = np.arange(tenor - 0.5, 0, step=-0.5)[::-1]
                coupon_half_yr = 0.5 * spot_rate
                z = np.interp(
                    times, zcb_rates[curve].index.values, zcb_rates[curve].values
                )  # Linear interpolation
                preceding_coupons_val = (coupon_half_yr * np.exp(-z * times)).sum()
                zcb_rates[curve][tenor] = (
                    -np.log((1 - preceding_coupons_val) / (1 + coupon_half_yr)) / tenor
                )
    return zcb_rates


# =============================================================================
# Reading Data
# =============================================================================


def get_trade_data(pair: str, year: str, path: str = "accumulation_opportunity/data"):
    """Reads local gzipped trade data file and return dataframe."""

    dtypes = {
        "PriceMillionths": int,
        "Side": int,
        "SizeBillionths": int,
        "timestamp_utc_nanoseconds": int,
    }

    filename = f"trades_narrow_{pair}_{year}.delim.gz"
    delimiter = {"2018": "|", "2021": "\t"}[year]

    with gzip.open(f"{path}/{filename}") as f:
        df = pd.read_csv(f, delimiter=delimiter, usecols=dtypes.keys(), dtype=dtypes)

    df.timestamp_utc_nanoseconds = pd.to_datetime(df.timestamp_utc_nanoseconds)

    return df.set_index("timestamp_utc_nanoseconds")


# =============================================================================
# Strategy
# =============================================================================


def get_accum_df(
    df: pd.DataFrame,
    arrival_time: str = "2018-04-08 22:05",
    quantity=3.25e9,
    side: int = 1,
    participation=0.050,
    max_trade_participation=0.10,
    chunk_size=6.5e9,
    price_window_ms=200,
):
    """Creates accumulation data frame that trades can be calculated from."""

    # Create accumulation dataframe
    df_accum = df.loc[arrival_time:].copy()
    df_accum["CumSizeBillionths"] = df_accum.SizeBillionths.cumsum()

    df_accum = df_accum[df_accum.Side == side].copy()

    price_agg = {1: "max", -1: "min"}[side]
    df_accum = (
        df_accum.reset_index()
        .groupby("timestamp_utc_nanoseconds")
        .agg(
            {
                "CumSizeBillionths": "max",
                "SizeBillionths": "sum",
                "PriceMillionths": price_agg,
                "Side": "first",
            }
        )
    )

    df_accum["CumChunks"] = df_accum.CumSizeBillionths.floordiv(chunk_size)
    df_accum["CumParticipation"] = (
        (
            (
                df_accum.CumChunks.map(
                    df_accum.groupby("CumChunks").min()["CumSizeBillionths"].iloc[1:]
                )
                * participation
            )
            .fillna(0)
            .apply(lambda x: min(x, quantity))
        )
        .round()
        .astype(int)
    )

    df_accum["TradePrice"] = (
        df_accum.PriceMillionths.sort_index(ascending=False)
        .rolling(f"{price_window_ms}ms")
        .agg(price_agg)
        .sort_index()
    )

    df_accum["QualifiedTrade"] = df_accum["TradePrice"] == df_accum["PriceMillionths"]

    df_accum["MaxTradeSize"] = round(
        df_accum.SizeBillionths * max_trade_participation, 0
    )

    # Calculate trades
    df_trades = get_trades_df(df_accum)
    assert (
        df_trades.StratTradeSize.sum() == quantity
    ), "Sum of trades does not equal quantity."

    # Prepare result record
    S0 = df_accum.iloc[0].PriceMillionths
    VWAP = round(
        df_trades.StratTradePrice.astype(object).dot(
            df_trades.StratTradeSize.astype(object)
        )
        / quantity
    )
    IS = VWAP / S0 - 1 if side else 1 - VWAP / S0

    completion_time = df_trades.index.max()

    result = dict(
        quantity=int(quantity),
        side=side,
        S0=S0,
        VWAP=VWAP,
        IS=IS,
        n_trades=len(df_trades),
        mean_trade_size=int(df_trades.StratTradeSize.mean()),
        arrival_time=df_accum.index.min(),
        completion_time=completion_time,
        execution_time=completion_time - df_accum.index.min(),
        participation=participation,
        max_trade_participation=max_trade_participation,
        chunk_size=int(chunk_size),
        price_window_ms=price_window_ms,
    )

    return df_accum.loc[: df_trades.index.max()], df_trades, result


def get_trades_df(df_accum: pd.DataFrame) -> pd.DataFrame:
    """Calculates trades from accumulation dataframe."""

    trades = []
    cum_trades = 0
    tick_idx = 0
    while cum_trades < df_accum.CumParticipation.max():
        tick = df_accum.iloc[tick_idx]

        if tick.CumParticipation - cum_trades > 0 and tick.QualifiedTrade:
            trade_size = min(tick.CumParticipation - cum_trades, tick.MaxTradeSize)
            trades.append((tick.name, trade_size, tick.TradePrice))
            cum_trades += trade_size

        tick_idx += 1

    df_trades = (
        pd.DataFrame(
            trades,
            columns=["timestamp_utc_nanoseconds", "StratTradeSize", "StratTradePrice"],
        )
        .set_index("timestamp_utc_nanoseconds")
        .astype(int)
    )

    return df_trades


def get_results_df(df: pd.DataFrame, params: Dict, nobs: int = 100) -> pd.DataFrame:
    """Runs strategy for given number of observations and returns results
    dataframe.
    """

    results = []
    while len(results) < nobs:
        params["arrival_time"] = np.random.choice(df.index.unique())
        try:
            results.append(get_accum_df(df, **params)[-1])
        except:
            pass

    return pd.DataFrame(results)


# =============================================================================
# Charts
# =============================================================================

COLORS = colors.qualitative.D3

IS_labels = [
    ("obs", lambda x: f"{x:>16d}"),
    ("min:max", lambda x: f"{x[0]:>0.3f}:{x[1]:>0.3f}"),
    ("mean", lambda x: f"{x:>13.4f}"),
    ("std", lambda x: f"{x:>15.4f}"),
    ("skewness", lambda x: f"{x:>11.4f}"),
    ("kurtosis", lambda x: f"{x:>13.4f}"),
]

ET_labels = [
    ("obs", lambda x: f"{x:>16d}"),
    ("min:max", lambda x: f"{x[0]:>0.0f}:{x[1]:>0.0f}"),
    ("mean", lambda x: f"{x:>13.2f}"),
    ("std", lambda x: f"{x:>15.2f}"),
    ("skewness", lambda x: f"{x:>11.4f}"),
    ("kurtosis", lambda x: f"{x:>13.4f}"),
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

    return go.layout.Annotation(
        text=(
            f"{title}:<br>"
            + ("<br>").join(
                [f"{k[0]:<10}{k[1](moments[i])}" for i, k in enumerate(labels)]
            )
        ),
        align="left",
        showarrow=False,
        xref=xref,
        yref=yref,
        x=x,
        y=y,
        bordercolor="black",
        borderwidth=1,
        borderpad=2,
        bgcolor="white",
        font=dict(size=10),
        xanchor=xanchor,
        yanchor="top",
    )


def make_trade_prices_chart(
    df: pd.DataFrame,
    df_accum: pd.DataFrame,
    df_trades: pd.DataFrame,
) -> go.Figure:
    df_result = df[df_accum.index.min() : df_accum.index.max()]

    fig = go.Figure()

    fig.add_scatter(
        x=df_result[df_result.Side == 1].index,
        y=df_result[df_result.Side == 1].PriceMillionths,
        mode="markers",
        name="Buy",
        marker=dict(size=7, color=COLORS[0]),
    )
    fig.add_scatter(
        x=df_result[df_result.Side == -1].index,
        y=df_result[df_result.Side == -1].PriceMillionths,
        mode="markers",
        name="Sell",
        marker=dict(size=7, color=COLORS[4]),
    )
    fig.add_scatter(
        x=df_trades.index,
        y=df_trades.StratTradePrice,
        mode="markers",
        name="Trade",
        marker=dict(size=4, color=COLORS[1]),
    )
    fig.update_layout(
        title=(
            f"Trade Prices: {df.name}<br>{str(df_accum.index.min())[:19]} "
            f"- {str(df_accum.index.max())[:19]}"
        ),
        xaxis_title="timestamp_utc_nanoseconds",
        yaxis_title="PriceMillionths",
    )

    return fig


def make_trade_sizes_chart(
    df: pd.DataFrame,
    df_accum: pd.DataFrame,
    df_trades: pd.DataFrame,
    bar_width: int = 3000,
) -> go.Figure:
    df_result = df[df_accum.index.min() : df_accum.index.max()]

    fig = go.Figure()

    fig.add_bar(
        x=df_result[df_result.Side == 1].index,
        y=df_result[df_result.Side == 1].SizeBillionths,
        name="Buy",
        marker_color=COLORS[0],
        width=bar_width,
    )
    fig.add_bar(
        x=df_result[df_result.Side == -1].index,
        y=df_result[df_result.Side == -1].SizeBillionths * -1,
        name="Sell",
        marker_color=COLORS[4],
        width=bar_width,
    )
    fig.add_bar(
        x=df_trades.index,
        y=df_trades.StratTradeSize * df_accum.iloc[0].Side,
        name="Trade",
        marker_color=COLORS[1],
        width=bar_width,
    )
    fig.update_layout(
        title=(
            f"Trade Sizes: {df.name}<br>{str(df_accum.index.min())[:19]} "
            f"- {str(df_accum.index.max())[:19]}"
        ),
        xaxis_title="timestamp_utc_nanoseconds",
        yaxis_title="SizeBillionths",
    )

    return fig


def make_participation_chart(
    df_accum: pd.DataFrame, df_trades: pd.DataFrame, name: str = "BTC-USD"
) -> go.Figure:
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    fig.add_trace(
        go.Scatter(
            x=df_accum.index,
            y=df_accum.CumSizeBillionths,
            name="Cumulative Total Volume",
            line=dict(color=COLORS[2]),
        ),
        secondary_y=True,
    )

    fig.add_trace(
        go.Scatter(
            x=df_accum.index,
            y=df_accum.CumParticipation,
            name="Target Participation",
            line=dict(color=COLORS[3]),
        ),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(
            x=df_trades.index,
            y=df_trades.StratTradeSize.cumsum(),
            name="Actual Participation",
            line=dict(color=COLORS[1]),
        ),
        secondary_y=False,
    )

    fig.update_layout(
        title=(
            f"Target vs. Actual Participation: {name}"
            f"<br>{str(df_accum.index.min())[:19]} - {str(df_accum.index.max())[:19]}"
        ),
        xaxis_title="timestamp_utc_nanoseconds",
        yaxis_title="SizeBillionths - Target and Actual",
        yaxis2_title="SizeBillionths - Total",
    )

    return fig


def get_result_hist(
    df_results: pd.DataFrame, title_text: str = "IS and Time Distributions"
) -> go.Figure:

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Implementation Shortfall", "Execution Duration"],
    )

    fig.add_trace(
        go.Histogram(x=df_results.IS, name="IS", histnorm="percent"), row=1, col=1
    )

    fig.add_trace(
        go.Histogram(
            x=df_results.execution_time.dt.seconds.divide(60),
            name="execution_time",
            histnorm="percent",
        ),
        row=1,
        col=2,
    )

    IS_mean_line = dict(
        type="line",
        yref="paper",
        y0=0.02,
        y1=0.98,
        xref="x",
        line_dash="dot",
        line_width=3,
        x0=df_results.IS.mean(),
        x1=df_results.IS.mean(),
        line=dict(color=COLORS[3]),
    )

    ET_mean_line = dict(
        type="line",
        yref="paper",
        y0=0.02,
        y1=0.98,
        xref="x2",
        line_dash="dot",
        line_width=3,
        x0=df_results.execution_time.dt.seconds.divide(60).mean(),
        x1=df_results.execution_time.dt.seconds.divide(60).mean(),
        line=dict(color=COLORS[3]),
    )

    fig.update_layout(shapes=[IS_mean_line, ET_mean_line], title_text=title_text)

    fig.add_annotation(
        get_moments_annotation(
            df_results.IS, "paper", "paper", 0.4, 0.98, "right", "IS", IS_labels
        )
    )

    fig.add_annotation(
        get_moments_annotation(
            df_results.execution_time.dt.seconds.divide(60),
            "paper",
            "paper",
            0.98,
            0.98,
            "right",
            "Time",
            ET_labels,
        )
    )

    return fig


def make_shortfall_time_scatter(
    df_results: pd.DataFrame, n_trend_obs: int = 200
) -> go.Figure:

    ols_result = stats.linregress(
        df_results.execution_time.dt.seconds.divide(60), df_results.IS
    )
    print(ols_result)

    fig = px.scatter(
        y=df_results.IS,
        x=df_results.execution_time.dt.seconds.divide(60),
        title="Implementation Shortfall vs. Execution Duration",
        labels=dict(y="IS", x="execution_time"),
    )

    fig.add_scatter(
        x=np.arange(n_trend_obs),
        y=[ols_result.intercept + ols_result.slope * x for x in np.arange(n_trend_obs)],
    )

    fig.update_layout(showlegend=False)

    return fig
