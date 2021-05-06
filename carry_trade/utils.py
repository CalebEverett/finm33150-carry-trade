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
# Data Fetching
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

    dfs_yc = {t: pd.read_csv(f"data/df_{t}.csv") for t in tickers}

    for k in dfs_yc:
        dfs_yc[k] = reindex_and_interpolate(dfs_yc[k], yc=True)

    return dfs_yc


def load_fx(currencies: List) -> Dict:
    """Load foreign exchange rates."""

    dfs_fx = {c: pd.read_csv(f"data/df_fx_{c}.csv") for c in currencies}

    for k in dfs_fx:
        dfs_fx[k] = reindex_and_interpolate(dfs_fx[k])

    return dfs_fx


def load_libor(libors: List) -> Dict:
    """Loads 3 month libors."""

    dfs_libor = {l: pd.read_csv(f"data/df_libor_{l}.csv") for l in libors}
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

        # Approximate components of home currency profit split between
        # (i) change in bond value, (ii) lending market fx, (iii) borrowing
        # market fx and (iv) interest expense.
        profits.append(
            (
                d1,
                (V_L1 - V_L0) / fx_L0,
                (V_L0 / fx_L1 - V_L0 / fx_L0) * (1 - leverage),
                K_B0 / fx_B1 - K_B0 / fx_B0,
                -I_B1 / fx_B0,
                K_H1 - K_H0,
            )
        )

    df_ret = pd.DataFrame(returns, columns=["date", "weekly_return"]).set_index("date")
    df_profit = pd.DataFrame(
        profits, columns=["date", "lend", "fx_LB", "fx_BH", "interest", "total"]
    ).set_index("date")

    return df_ret, df_profit


# =============================================================================
# Charts
# =============================================================================

COLORS = colors.qualitative.T10

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
):

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"Change in 5-Year Yield: {yc_L}",
            f"Change in FX Rate: {fx_B}:USD",
            f"3 Month Libor: {libor}",
            f"Change in FX Rate: {fx_L}:USD",
        ],
        vertical_spacing=0.09,
        horizontal_spacing=0.08,
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_yc[yc_L].loc[date_range]["5-year"].pct_change(),
            line=dict(width=1, color=COLORS[0]),
            name=yc_L,
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_fx[fx_B].loc[date_range].rate.pct_change(),
            line=dict(width=1, color=COLORS[0]),
            name=fx_B,
        ),
        row=1,
        col=2,
    )

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

    fig.add_trace(
        go.Scatter(
            x=date_range,
            y=dfs_fx[fx_L].loc[date_range].loc[date_range].rate.pct_change(),
            line=dict(width=1, color=COLORS[0]),
            name=fx_L,
        ),
        row=2,
        col=2,
    )

    fig.update_xaxes(showline=True, linewidth=1, linecolor="black", mirror=True)
    fig.update_yaxes(showline=True, linewidth=1, linecolor="black", mirror=True)

    fig.update_layout(
        title_text=(
            f"Weekly Carry Trade: Borrow {fx_B} and Lend {fx_L}"
            f"<br>{date_range.min().strftime('%Y-%m-%d')}"
            f" - {date_range.max().strftime('%Y-%m-%d')}"
        ),
        showlegend=False,
        height=600,
        font=dict(size=10),
        margin=dict(l=50, r=50, b=50, t=80),
    )

    for i in fig["layout"]["annotations"]:
        i["font"]["size"] = 12

    return fig


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
