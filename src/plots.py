"""Pure Plotly figure builders for the Streamlit dashboard.

Every function takes data (and plain-language labels) in and returns a plotly.graph_objects.Figure.
None of them reads a file or config.yaml or calls st.*; app.py only hands the Figure to st.plotly_chart.

House style, so a reader who has not seen the code can use every chart:
- one plain-sentence title saying what is shown, no raw variable names in it;
- labelled axes with units; hover text with the exact value;
- the Okabe-Ito colour-blind-safe palette, and colour is never the only cue (hatching, dashes,
  a legend entry and a tick-label suffix back it up).
Written without plotly installed in the project venv, so every call here is from memory of the
plotly API and unverified until the tests have run.
"""
import re

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import norm

from src.correlogram import acf_pacf_band

# Okabe-Ito palette (colour-blind safe).
BLUE, ORANGE, SKY, GREEN = "#0072B2", "#E69F00", "#56B4E9", "#009E73"
VERMILLION, PURPLE, YELLOW, GREY = "#D55E00", "#CC79A7", "#F0E442", "#8C8C8C"

SERIES_COLOURS = [BLUE, ORANGE, GREEN, PURPLE, SKY]            # one per selector, in a fixed order
SERIES_DASHES = ["solid", "dash", "dot", "dashdot", "longdash"]  # second cue next to the colour
SOURCE_STYLES = [(BLUE, ""), (ORANGE, "/")]                      # (colour, hatch) of saved run, live run
BAND_FILL = "rgba(140,140,140,0.30)"


def _style(fig: go.Figure, title: str, xtitle: str | None, ytitle: str | None, height: int) -> go.Figure:
    """Apply the shared layout: title, axis titles, height, margins and a horizontal legend.

    Input: a Figure, the plain-sentence title, x and y axis titles (None = leave the axes alone,
    for subplot grids that title each axis themselves), the height in pixels. Output: the same Figure.
    """
    fig.update_layout(title=dict(text=title), height=height, margin=dict(l=60, r=30, t=90, b=60),
                      legend=dict(orientation="h", yanchor="top", y=-0.18, x=0))

    if xtitle is not None:
        fig.update_xaxes(title_text=xtitle)
    if ytitle is not None:
        fig.update_yaxes(title_text=ytitle)

    return fig


def _marker(colour: str, hatch: str) -> dict:
    """Bar marker: a solid colour, or (hatch non-empty) a hatched fill in that colour.

    Input: hex colour and a plotly pattern shape such as "/" ("" = solid). Output: marker dict.
    The hatch is the second cue, so a bar is told apart even without seeing its colour.
    """
    if not hatch:
        return dict(color=colour)

    # Translucent background of the same colour behind the hatch lines.
    r, g, b = (int(colour[i:i + 2], 16) for i in (1, 3, 5))

    return dict(color=colour, pattern=dict(shape=hatch, fgcolor=colour, bgcolor=f"rgba({r},{g},{b},0.25)"))


def _colour_index(name: str, order: list[str] | None, fallback: int) -> int:
    """Stable palette position of a selector: its place in `order` if given, else `fallback`."""
    return (order.index(name) if order and name in order else fallback) % len(SERIES_COLOURS)


def _order_key(text: str) -> tuple[int, ...]:
    """Sort key for order labels like "(2, 1)": the numbers inside, so (1, 1) < (1, 2) < (2, 0)."""
    return tuple(int(v) for v in re.findall(r"\d+", text))


# ---------------------------------------------------------------------------------------
# Data tab
# ---------------------------------------------------------------------------------------

def price_line(price: pd.Series, label: str, unit: str) -> go.Figure:
    """The daily exchange rate as a line with a range slider underneath.

    Input: price Series (DatetimeIndex), label (pair name such as "EUR/USD"), unit (what one unit of
    the rate is, e.g. "US dollars per euro"). Output: Figure. Drag the slider to zoom into a period.
    """
    fig = go.Figure(go.Scatter(
        x=price.index, y=price.to_numpy(), mode="lines", line=dict(color=BLUE, width=1.3),
        name="Daily rate", hovertemplate="%{x|%d %b %Y}<br>Rate: %{y:.4f}<extra></extra>"))
    fig.update_xaxes(rangeslider_visible=True)

    return _style(fig, f"Daily exchange rate, {label}: drag the bar underneath to zoom into a period",
                  "Date", f"Exchange rate ({unit})", 440)


def return_line(returns: pd.Series, label: str) -> go.Figure:
    """The daily log return in percent as a line, with a zero line.

    Input: returns Series (percent, DatetimeIndex), label. Output: Figure. Calm and wild stretches
    (volatility clustering) are what the rest of the study models.
    """
    fig = go.Figure(go.Scatter(
        x=returns.index, y=returns.to_numpy(), mode="lines", line=dict(color=BLUE, width=0.8),
        name="Daily return", hovertemplate="%{x|%d %b %Y}<br>Return: %{y:.3f}%<extra></extra>"))
    fig.add_hline(y=0, line_width=1, line_color=GREY)

    return _style(fig, f"Daily percentage change in the {label} rate: calm and wild stretches alternate",
                  "Date", "Daily log return (% change per day)", 340)


def return_histogram(returns: pd.Series, label: str) -> go.Figure:
    """Histogram of the returns with a normal curve of the same mean and spread on top.

    Input: returns Series (percent), label. Output: Figure with 2 traces (bars, curve) and two buttons
    that switch the vertical axis between linear and logarithmic. On the log axis the heavy tails show up
    as bars far above the dashed curve.
    """
    x = returns.to_numpy(dtype=float)
    mu, sd = x.mean(), x.std(ddof=1)
    grid = np.linspace(x.min(), x.max(), 400)

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=x, histnorm="probability density", nbinsx=80, marker_color=SKY, opacity=0.9,
        name="Observed daily returns",
        hovertemplate="Return around %{x:.2f}%<br>Density: %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=grid, y=norm.pdf(grid, mu, sd), mode="lines", line=dict(color=ORANGE, width=2.5, dash="dash"),
        name="Normal curve with the same mean and spread",
        hovertemplate="Return %{x:.2f}%<br>Normal density: %{y:.4f}<extra></extra>"))
    fig.update_layout(updatemenus=[dict(
        type="buttons", direction="right", x=0, y=1.14, xanchor="left",
        buttons=[dict(label="Linear scale", method="relayout", args=[{"yaxis.type": "linear"}]),
                 dict(label="Log scale (shows the tails)", method="relayout", args=[{"yaxis.type": "log"}])])])

    return _style(fig, f"How daily changes in the {label} rate are spread, compared with a normal bell curve",
                  "Daily log return (%)", "Density (share of days per percentage point)", 420)


# ---------------------------------------------------------------------------------------
# Diagnose tab
# ---------------------------------------------------------------------------------------

def correlogram_grid(returns, nlags: int, confidence: float = 0.95) -> go.Figure:
    """2x2 grid of interactive ACF and PACF bars: top row returns, bottom row squared returns.

    Input: returns array-like, nlags (largest lag), confidence (width of the grey band). Output: Figure.
    Idea: the grey band holds what pure noise would produce about `confidence` of the time; bars
    outside it (hatched, orange) are spikes. Hover shows lag, value and "Outside band: yes/no".
    Squares use the demeaned returns, the same convention as the rules. Each panel has the traces
    band, inside-band bars and outside-band bars; only the first panel adds legend entries.
    """
    x = np.asarray(returns, dtype=float)
    rows = [("Returns", x), ("Squared returns", (x - x.mean()) ** 2)]
    kinds = [("acf", "autocorrelation (ACF)"), ("pacf", "partial autocorrelation (PACF)")]
    lags = np.arange(1, nlags + 1)

    fig = make_subplots(rows=2, cols=2, vertical_spacing=0.2, horizontal_spacing=0.08,
                        subplot_titles=[f"{name}: {kname}" for name, _ in rows for _, kname in kinds])

    for r, (name, series) in enumerate(rows, start=1):
        cor = acf_pacf_band(series, nlags, confidence)
        band = float(cor["band"])

        for c, (key, kname) in enumerate(kinds, start=1):
            vals, first = cor[key], (r == 1 and c == 1)
            outside = np.abs(vals) > band

            # The shaded band as a filled rectangle (a trace, so it can have a legend entry).
            fig.add_trace(go.Scatter(
                x=[0.5, nlags + 0.5, nlags + 0.5, 0.5], y=[-band, -band, band, band], fill="toself",
                mode="lines", line=dict(width=0), fillcolor=BAND_FILL, hoverinfo="skip",
                name=f"{confidence:.0%} band (what pure noise stays inside)", legendgroup="band",
                showlegend=first), row=r, col=c)

            # Bars inside the band (solid blue) and outside it (hatched orange): colour AND pattern.
            for mask, nm, colour, hatch, flag in [(~outside, "Inside the band", BLUE, "", "no"),
                                                  (outside, "Outside the band (spike)", VERMILLION, "/", "yes")]:
                fig.add_trace(go.Bar(
                    x=lags[mask], y=vals[mask], name=nm, legendgroup=nm, showlegend=first,
                    marker=_marker(colour, hatch), customdata=[flag] * int(mask.sum()),
                    hovertemplate=(f"{name}, {kname}<br>Lag %{{x}} days<br>Value %{{y:.3f}}"
                                   "<br>Outside band: %{customdata}<extra></extra>")), row=r, col=c)

            fig.update_xaxes(title_text="Lag (days)", row=r, col=c)
            fig.update_yaxes(title_text="Correlation (-1 to 1)", row=r, col=c)

    fig.update_layout(barmode="overlay")

    return _style(fig, "How each day's return and squared return relate to earlier days: "
                       "bars outside the grey band are unlikely to be chance", None, None, 720)


# ---------------------------------------------------------------------------------------
# Select tab
# ---------------------------------------------------------------------------------------

def gate_pvalue_bars(gate: pd.DataFrame, alpha: float) -> go.Figure:
    """Bar chart of the gate p-values with the pass threshold as a dashed line.

    Input: gate DataFrame with columns test (text), p_value, passes (bool) as made by
    dashboard.selection_tables; alpha = gate.alpha. Output: Figure. A bar must stand above the line to
    pass; passing bars are solid blue, failing bars hatched orange (colour AND pattern).
    An empty table (no valid fit) gives an empty chart with the same title and axes.
    """
    fig = go.Figure()

    for passed, nm, colour, hatch in [(True, "Passes (above the line)", BLUE, ""),
                                      (False, "Fails (below the line)", VERMILLION, "/")]:
        sub = gate[gate["passes"] == passed]
        fig.add_trace(go.Bar(
            x=sub["test"], y=sub["p_value"], name=nm, marker=_marker(colour, hatch),
            hovertemplate="%{x}<br>p-value: %{y:.4f}<br>" + ("Passes" if passed else "Fails") + "<extra></extra>"))

    fig.add_hline(y=alpha, line_dash="dash", line_color="black", line_width=2,
                  annotation_text=f"Threshold {alpha}", annotation_position="top left")
    fig.update_yaxes(range=[0, 1.05])

    return _style(fig, "Diagnostic gate: every test needs a p-value above the dashed line for the model to pass",
                  "Diagnostic test", "p-value (higher is better)", 420)


# ---------------------------------------------------------------------------------------
# Forecast tab
# ---------------------------------------------------------------------------------------

def _fallback_runs(fc: pd.DataFrame, selector: str) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Stretches of consecutive days on which `selector` had no valid fit (fit_status == no_valid_fit).

    Input: fc forecast table with date, selector, fit_status. Output: list of (first_day, day_after_last)
    pairs; the end is one day late so a single-day run still has width. Empty if the column is missing.
    """
    if "fit_status" not in fc.columns:
        return []

    sub = fc[fc["selector"] == selector].sort_values("date")
    if sub.empty:
        return []

    bad = (sub["fit_status"] == "no_valid_fit").to_numpy()
    dates = sub["date"].reset_index(drop=True)

    # Run id changes whenever the flag flips; keep the runs where the flag is True.
    run_id = np.cumsum(np.r_[True, bad[1:] != bad[:-1]])
    runs = []

    for k in np.unique(run_id):
        pos = np.flatnonzero(run_id == k)   # positions of this run
        if bad[pos[0]]:
            runs.append((dates[pos[0]], dates[pos[-1]] + pd.Timedelta(days=1)))

    return runs


def forecast_lines(fc: pd.DataFrame, selectors: list[str], label: str, names: dict[str, str],
                   fallback_selector: str | None = None, order: list[str] | None = None) -> go.Figure:
    """Forecast volatility sqrt(h) of the chosen selectors against the realised |return|.

    Input: fc = one pair's forecast table (date, selector, h, r2 and optionally fit_status), selectors
    to draw, label (pair name), names (selector id -> plain name), fallback_selector (shade the days
    it used the rolling-variance fallback, or None), order (all selector ids, to keep colours stable).
    Output: Figure with a range slider; click legend entries to hide or show a line.
    sqrt(h) is a forecast standard deviation in percent per day; |return| = sqrt(r2) is one noisy draw.
    Raises ValueError if a requested selector is not in the table.
    """
    fig = go.Figure()

    # Realised |return| per target day (r2 is the same for every selector, so take it once).
    realised = fc.groupby("date")["r2"].first().pipe(np.sqrt)
    fig.add_trace(go.Scatter(
        x=realised.index, y=realised.to_numpy(), mode="lines", line=dict(color=GREY, width=1),
        name="Realised size of the day's move (|return|)",
        hovertemplate="%{x|%d %b %Y}<br>|return|: %{y:.3f}%<extra></extra>"))

    for i, sel in enumerate(selectors):
        sub = fc[fc["selector"] == sel].sort_values("date")
        if sub.empty:
            raise ValueError(f"selector {sel!r} not in the forecast table")

        k, nm = _colour_index(sel, order, i), names.get(sel, sel)
        fig.add_trace(go.Scatter(
            x=sub["date"], y=np.sqrt(sub["h"].to_numpy()), mode="lines",
            line=dict(color=SERIES_COLOURS[k], dash=SERIES_DASHES[k], width=1.8), name=nm,
            hovertemplate="%{x|%d %b %Y}<br>" + nm + ": %{y:.3f}%<extra></extra>"))

    # Shade the days the chosen selector used the fallback, and give the shading a legend entry.
    runs = _fallback_runs(fc, fallback_selector) if fallback_selector else []
    for start, end in runs:
        fig.add_vrect(x0=start, x1=end, fillcolor=YELLOW, opacity=0.3, line_width=0, layer="below")
    if runs:
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers", marker=dict(symbol="square", size=14, color=YELLOW),
            name=f"Days the {names.get(fallback_selector, fallback_selector)} used the rolling-variance fallback"))

    fig.update_xaxes(rangeslider_visible=True)

    return _style(fig, f"Volatility forecast of each selector against how large each day's move really was, {label}",
                  "Date of the forecast day", "Volatility (daily standard deviation, % per day)", 520)


def cumulative_qlike_lines(diff: pd.DataFrame, baseline_name: str, label: str, names: dict[str, str],
                           order: list[str] | None = None) -> go.Figure:
    """Cumulative QLIKE difference of every selector against a baseline, one line each.

    Input: diff = evaluate.cumulative_qlike_diff output (date index, one column per selector), the
    plain name of the baseline, label (pair name), names (selector id -> plain name), order (all selector
    ids, for stable colours). Output: Figure. A line that falls is a selector doing better than the
    baseline; its end value is the total difference over the test days.
    """
    fig = go.Figure()

    for i, sel in enumerate(diff.columns):
        k, nm = _colour_index(sel, order, i), names.get(sel, sel)
        fig.add_trace(go.Scatter(
            x=diff.index, y=diff[sel].to_numpy(), mode="lines",
            line=dict(color=SERIES_COLOURS[k], dash=SERIES_DASHES[k], width=1.8), name=nm,
            hovertemplate="%{x|%d %b %Y}<br>" + nm + ": %{y:.3f}<extra></extra>"))

    fig.add_hline(y=0, line_width=1, line_color="black")
    fig.update_xaxes(rangeslider_visible=True)

    return _style(fig, f"Running total of how much better (below 0) or worse (above 0) each selector forecasts "
                       f"than {baseline_name}, {label}",
                  "Date", "Cumulative QLIKE difference (loss units added up over days)", 460)


# ---------------------------------------------------------------------------------------
# Simulate tab
# ---------------------------------------------------------------------------------------

def recovery_bars(live: dict | None, saved: dict | None, rates: list[tuple[str, str, str]],
                  live_R: int, saved_R: int | None) -> go.Figure:
    """Grouped bars comparing the saved large run with the live run, with Monte Carlo SE error bars.

    Input: live and saved = recovery_metrics-style dicts (a share in 0..1 under each rate key and its
    SE under the se key) or None; rates = list of (rate key, SE key, plain label); live_R and saved_R =
    number of simulated series behind each. Output: Figure; values in percent. A NaN rate is left out.
    The saved run is drawn first (solid blue), the live run second (hatched orange).
    """
    fig = go.Figure()
    sources = [(f"Saved reference run ({saved_R} simulated series)", saved),
               (f"Live run ({live_R} simulated series)", live)]

    for (nm, data), (colour, hatch) in zip(sources, SOURCE_STYLES):
        if data is None:
            continue

        pts = [(lab, 100 * data[k], 100 * data[se]) for k, se, lab in rates if not np.isnan(data[k])]
        fig.add_trace(go.Bar(
            x=[p[0] for p in pts], y=[p[1] for p in pts], name=nm,
            error_y=dict(type="data", array=[p[2] for p in pts], visible=True),
            customdata=[p[2] for p in pts], marker=_marker(colour, hatch),
            hovertemplate="%{x}<br>" + nm + ": %{y:.1f}%<br>Monte Carlo SE: +/- %{customdata:.1f} points<extra></extra>"))

    fig.update_layout(barmode="group")
    fig.update_yaxes(range=[0, 108])

    return _style(fig, "How often the selector gets the order right: the small live run next to the large saved run",
                  "Measure", "Share of simulated series (%)", 440)


def order_choice_bars(saved: pd.Series | None, live: pd.Series | None, saved_R: int | None, live_R: int,
                      true_order: str, max_orders: int) -> go.Figure:
    """Bars of which (p, q) the selector chose, in percent of runs, with the true order highlighted.

    Input: saved and live = Series (index = order text like "(1, 1)", values = percent of runs) or None
    (at least one must be given); saved_R and live_R = number of series behind each; true_order text;
    max_orders = most bars shown (the most frequent ones; the true order is always kept).
    Output: Figure. Saved run is solid blue, live run hatched orange. The true order is marked by a green
    band, a "(true)" suffix on its tick label and a legend entry "True order".
    Raises ValueError if both series are None.
    """
    sources = [(f"Saved reference run ({saved_R} simulated series)", saved),
               (f"Live run ({live_R} simulated series)", live)]
    active = [(nm, s, style) for (nm, s), style in zip(sources, SOURCE_STYLES) if s is not None]
    if not active:
        raise ValueError("order_choice_bars needs a saved or a live run")

    # Orders to show: best-supported first, true order forced in, then sorted by (p, q) for reading.
    best = pd.concat([s for _, s, _ in active]).groupby(level=0).max().sort_values(ascending=False)
    keep = [o for o in best.index if o != true_order][:max(max_orders - 1, 0)] + [true_order]
    orders = sorted(set(keep), key=_order_key)

    fig = go.Figure()
    for nm, s, (colour, hatch) in active:
        fig.add_trace(go.Bar(
            x=orders, y=[float(s.get(o, 0.0)) for o in orders], name=nm, marker=_marker(colour, hatch),
            hovertemplate="%{x}<br>" + nm + ": chosen in %{y:.1f}% of runs<extra></extra>"))

    # True order: a green band behind its bars, a tick-label suffix, and a legend entry.
    idx = orders.index(true_order)
    fig.add_vrect(x0=idx - 0.5, x1=idx + 0.5, fillcolor=GREEN, opacity=0.2, line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers",
                             marker=dict(symbol="square", size=14, color=GREEN), name="True order"))
    fig.update_xaxes(tickmode="array", tickvals=orders,
                     ticktext=[o + " (true)" if o == true_order else o for o in orders])
    fig.update_layout(barmode="group")

    return _style(fig, "Which variance order the selector chose, as a share of simulated series "
                       "(the true order is shaded green)",
                  "Order chosen (p, q): lags of past shocks, lags of past variances", "Share of simulated series (%)", 440)


def order_heatmap(orders: pd.DataFrame, n: int, selector: str, model_names: dict[str, str],
                  true_orders: dict[str, str]) -> go.Figure:
    """Heatmap: for every true model, which order the selector chose, in percent of its runs.

    Input: orders = recovery_orders.csv-style table (true_model, n, selector, final_order, count),
    n and selector to show, model_names (model id -> plain name; its order sets the row order),
    true_orders (model id -> "(p, q)"). Output: Figure; rows sum to 100. Brighter = chosen more often;
    outlined squares ("True order") mark each model's true order.
    """
    sub = orders[(orders["n"] == n) & (orders["selector"] == selector)]
    table = sub.pivot_table(index="true_model", columns="final_order", values="count",
                            aggfunc="sum", fill_value=0)
    table = table.reindex([m for m in model_names if m in table.index])
    cols = sorted(table.columns, key=_order_key)
    pct = 100 * table[cols].div(table[cols].sum(axis=1), axis=0)
    rows = [model_names[m] for m in pct.index]

    fig = go.Figure(go.Heatmap(
        z=pct.to_numpy(), x=cols, y=rows, colorscale="Viridis", zmin=0, zmax=100,
        colorbar=dict(title="Share of series (%)"),
        hovertemplate="True model: %{y}<br>Order chosen: %{x}<br>Chosen in %{z:.1f}% of series<extra></extra>"))

    # Outlined squares at each model's true order (only where that order appears as a column).
    marks = [(true_orders[m], model_names[m]) for m in pct.index if true_orders[m] in cols]
    fig.add_trace(go.Scatter(
        x=[a for a, _ in marks], y=[b for _, b in marks], mode="markers", name="True order",
        marker=dict(symbol="square-open", size=26, color=VERMILLION, line=dict(width=3, color=VERMILLION)),
        hovertemplate="True order of %{y}: %{x}<extra></extra>"))
    fig.update_yaxes(autorange="reversed")

    return _style(fig, "Which order the selector chose for each true model, as a share of its simulated series "
                       "(outlined squares mark the true order)",
                  "Order chosen (p, q)", "True model", 460)
