from numbers import Number
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Dash, Input, Output, State, ctx, dcc, html
from dash.exceptions import PreventUpdate
from flask import send_from_directory


# -------------------------------------------
# Data Preparation 
# -------------------------------------------

# Set up file paths for the project
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "global_leader_ideologies.csv"
FONT_DIR = (BASE_DIR / "fonts" / "monument-grotesk-font-family-1764226824-0").resolve()

# Read the CSV file containing all the ideology data
raw_df = pd.read_csv(DATA_FILE)


def normalize_democracy(series: pd.Series) -> pd.Series:
    """
    Clean up democracy values in the dataset.
    Converts all values to lowercase and standardizes them to 'yes', 'no', or 'no data'.
    """
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.where(normalized.isin(["yes", "no"]), "no data")


# Define which columns we need for displaying summary information
SUMMARY_COLUMNS = [
    "hog",
    "hog_party",
    "hog_party_eng",
    "leader",
    "leader_party",
    "leader_party_eng",
    "hog_left",
    "hog_center",
    "hog_right",
]

# Create the main dataframe with selected columns
df = raw_df.reindex(columns=["year", "hog_ideology", "region", "democracy", *SUMMARY_COLUMNS]).copy()
df["hog_ideology"] = df["hog_ideology"].str.lower()  # Make ideology values lowercase
df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")  # Convert year to numeric
df["region"] = df["region"].fillna("Unknown")  # Replace missing regions with 'Unknown'
df["democracy_flag"] = normalize_democracy(df["democracy"])  # Normalize democracy values

# Define the valid ideology types and their colors
valid_ideologies = ["leftist", "centrist", "rightist"]
color_map = {
    "leftist": "#1d76db",  # Blue for leftist
    "centrist": "#b094b0",  # Purple for centrist
    "rightist": "#db231d",  # Red for rightist
}
# Greyscale colors for different filter stages
GREY_STAGE_COLORS = {
    1: "#dcdcdc",
    2: "#b7b6b6",
    3: "#8c8b8b",
}

# -------------------------------------------
# Styling and Configuration Constants
# -------------------------------------------
FONT_FAMILY = "ABCMonumentGrotesk, Arial, sans-serif"
CHOICE_LABEL_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "gap": "8px",
    "padding": "4px 0",
    "lineHeight": "1.3",
}
SECTION_LABEL_STYLE = {"fontSize": 16, "fontWeight": 600}
HOVER_TEMPLATE = "<b>%{location}</b><br>Click for political summary<extra></extra>"
HOVER_LABEL_STYLE = {
    "bgcolor": "#ffffff",
    "bordercolor": "#d7d7d7",
    "font": {"family": FONT_FAMILY, "color": "#111", "size": 12},
}

# Project metadata and information sections
INFO_SECTIONS = {
    "Project name": "Ideology Atlas",
    "Research questions": "How do geography, regime type, and ideology intersect across modern leadership history?",
    "Sources": "https://www.ippapublicpolicy.org/file/paper/60c247759f1df.pdf",
    "Made by": "William Kosse, Márton Berettyán, Nóra Balogh",
}
DATASET_SOURCE_URL = "https://github.com/bastianherre/global-leader-ideologies"

# Prepare the map dataframe - specific columns for world map visualization
map_df = raw_df.reindex(
    columns=["country_name", "hog_ideology", "year", "region", "democracy", *SUMMARY_COLUMNS]
).copy()
map_df["hog_ideology"] = map_df["hog_ideology"].str.lower()
map_df["year"] = pd.to_numeric(map_df["year"], errors="coerce").astype("Int64")
map_df = map_df[map_df["hog_ideology"].isin(valid_ideologies)]  # Keep only valid ideologies
map_df["democracy_flag"] = normalize_democracy(map_df["democracy"])
# Remove duplicate country-year entries, keeping the most recent
map_df = map_df.drop_duplicates(subset=["country_name", "year"], keep="last")

# Get available years for the slider
available_years = sorted(map_df["year"].dropna().unique())
min_year = int(available_years[0]) if available_years else None
max_year = int(available_years[-1]) if available_years else None


def _year_marks(years):
    """
    Create year labels for the slider.
    Only shows first year, last year, and every 10 years to avoid crowding.
    """
    if not years:
        return {}
    first, last = years[0], years[-1]
    return {
        int(year): str(int(year))
        for year in years
        if year in (first, last) or int(year) % 10 == 0
    }


year_marks = _year_marks(available_years)


# -------------------------------------------
# Helper Functions
# -------------------------------------------
# These functions help clean, validate, and format data for the dashboard

def _resolve_regions(selection):
    """Filter regions - returns None if 'all' is selected (no filtering)."""
    if not selection or "all" in selection:
        return None
    return selection


def _resolve_ideologies(selection):
    """Filter ideologies - only keep valid ones from the list."""
    return [ide for ide in selection or [] if ide in valid_ideologies]


def _apply_multi_filter(frame, column, values):
    """
    Filter a dataframe based on column values.
    Returns empty dataframe if values is empty list, original if None.
    """
    if values is None:
        return frame
    if not values:
        return frame.iloc[0:0]
    return frame[frame[column].isin(values)]


def _is_one(value) -> bool:
    """Check if a value equals 1.0 (handles NaN and type conversion)."""
    if pd.isna(value):
        return False
    try:
        return float(value) == 1.0
    except (TypeError, ValueError):
        return bool(value)


def _safe_text(value, fallback="Unknown"):
    """
    Safely convert any value to text.
    Returns fallback string for None, empty strings, or NaN values.
    """
    if value is None:
        return fallback
    if isinstance(value, str):
        return value.strip() or fallback
    if isinstance(value, Number):
        if pd.isna(value):
            return fallback
        if float(value).is_integer():
            return str(int(value))
        return str(value)
    if pd.isna(value):
        return fallback
    return str(value)


def _format_democracy(row):
    """Format democracy field for display - converts to 'Democracy' or 'Non-democracy'."""
    value = row.get("democracy_flag")
    if value in (None, "no data"):
        value = row.get("democracy")
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "Unknown"
    normalized = str(value).strip().lower()
    if normalized in {"1", "yes", "democracy", "true"}:
        return "Democracy"
    if normalized in {"0", "no", "non-democracy", "false"}:
        return "Non-democracy"
    return "Unknown"


def _political_leaning(row):
    """Determine the political leaning based on left/center/right flags."""
    if _is_one(row.get("hog_left")):
        return "Left"
    if _is_one(row.get("hog_center")):
        return "Center"
    if _is_one(row.get("hog_right")):
        return "Right"
    ideology = row.get("hog_ideology")
    if isinstance(ideology, str) and ideology:
        return ideology.capitalize()
    return "Unknown"


def _pref_value(row, primary, fallback):
    """Get preferred value from a row - use primary field, fallback to alternative if empty."""
    value = row.get(primary)
    if value is None or (isinstance(value, str) and not value.strip()):
        value = row.get(fallback)
    return _safe_text(value)


def _extract_summary_row(country, year):
    """Find and return the data row for a specific country and year."""
    if not country or year is None:
        return None
    country_lower = str(country).strip().lower()
    matches = map_df[
        map_df["country_name"].fillna("").str.lower().eq(country_lower)
        & map_df["year"].eq(year)
    ]
    if matches.empty:
        return None
    return matches.iloc[0]


def _build_summary_card(country, year, row):
    if row is None:
        message = f"No political summary available for {country or 'the selected country'} in {year or 'this year'}."
        return html.Div(
            className="summary-card",
            children=[html.Div(message, className="summary-empty")],
        )

    fields = [
        ("Country", _safe_text(country)),
        ("Year", _safe_text(year)),
        ("Democracy", _format_democracy(row)),
        ("Political leaning", _political_leaning(row)),
        ("Head of Government", _safe_text(row.get("hog"))),
        ("HoG Party", _pref_value(row, "hog_party_eng", "hog_party")),
        ("Leader", _safe_text(row.get("leader"))),
        ("Leader Party", _pref_value(row, "leader_party_eng", "leader_party")),
        ("Region", _safe_text(row.get("region"))),
    ]

    return html.Div(
        className="summary-card",
        children=[
            html.H3("Political Snapshot", className="summary-title"),
            html.Div(
                [
                    html.Div(
                        className="summary-field",
                        children=[
                            html.Span(f"{label}:", className="summary-label"),
                            html.Span(value, className="summary-value"),
                        ],
                    )
                    for label, value in fields
                ]
            ),
        ],
    )


def _build_info_card():
    instruction_steps = [
        "Pick one or more regions to activate the workflow.",
        "Choose a regime type to focus the dataset.",
        "Select at least one ideology to reveal ideological splits.",
        "Lock a specific year on the slider to unlock the full-color map view.",
    ]

    info_fields = [
        ("Project name", INFO_SECTIONS["Project name"]),
        ("Research questions", INFO_SECTIONS["Research questions"]),
        (
            "Sources",
            html.Span(
                [
                    html.A(
                        "Identifying Ideologues: A Global Dataset on Political Leaders, 1945-2019, Bastian Herre",
                        href=INFO_SECTIONS["Sources"],
                        target="_blank",
                        className="summary-link",
                    ),
                    html.Span(" — "),
                    html.A(
                        "Dataset (GitHub)",
                        href=DATASET_SOURCE_URL,
                        target="_blank",
                        className="summary-link",
                    ),
                ]
            ),
        ),
        ("Made by", INFO_SECTIONS["Made by"]),
    ]

    return html.Div(
        className="summary-card",
        children=[
            html.H3("Project Information", className="summary-title"),
            html.Div(
                [
                    html.Div(
                        className="summary-field",
                        children=[
                            html.Span(f"{label}:", className="summary-label"),
                            html.Span(value, className="summary-value"),
                        ],
                    )
                    for label, value in info_fields
                ]
            ),
            html.H4("Website instructions", className="summary-subtitle"),
            html.Ul(
                [html.Li(step, className="summary-instruction") for step in instruction_steps],
                className="summary-list",
            ),
        ],
    )


def _compute_stage(has_region, regimes, ideologies, year_selected):
    stage = 0
    for idx, ready in enumerate([has_region, regimes, ideologies, year_selected], start=1):
        if not ready:
            return stage
        stage = idx
    return 4


def _prepare_stage_highlight(stage, regions, regimes, ideologies, has_region_selection):
    if stage == 0 or not has_region_selection:
        return pd.DataFrame()

    subset = map_df
    if regions:
        subset = subset[subset["region"].isin(regions)]
    if stage >= 2:
        subset = _apply_multi_filter(subset, "democracy_flag", regimes)
    if stage >= 3:
        subset = _apply_multi_filter(subset, "hog_ideology", ideologies)
    return subset.drop_duplicates(subset=["country_name"])


# --------------------------------------
# Figure factories
# --------------------------------------
def make_world_map(
    stage,
    selected_regions=None,
    selected_year=None,
    democracy_filters=None,
    ideology_filters=None,
    has_region_selection=False,
):
    filtered = map_df
    if selected_regions:
        filtered = filtered[filtered["region"].isin(selected_regions)]

    if stage == 4:
        filtered = _apply_multi_filter(filtered, "democracy_flag", democracy_filters)
        filtered = _apply_multi_filter(filtered, "hog_ideology", ideology_filters)
        if selected_year is not None:
            filtered = filtered[filtered["year"] == selected_year]

    if stage == 4 and selected_year is not None and not filtered.empty:
        fig = px.choropleth(
            filtered,
            locations="country_name",
            locationmode="country names",
            color="hog_ideology",
            color_discrete_map=color_map,
        )
    else:
        highlight_df = _prepare_stage_highlight(
            stage,
            selected_regions,
            democracy_filters,
            ideology_filters,
            has_region_selection,
        )
        if highlight_df.empty:
            fig = go.Figure()
            fig.add_trace(
                go.Choropleth(locations=[], z=[], showscale=False, hoverinfo="skip")
            )
        else:
            stage_label = f"stage_{stage}"
            highlight_df = highlight_df.assign(stage_label=stage_label)
            fig = px.choropleth(
                highlight_df,
                locations="country_name",
                locationmode="country names",
                color="stage_label",
                color_discrete_map={stage_label: GREY_STAGE_COLORS.get(stage, "#dddddd")},
            )

    fig.update_geos(
        showland=True,
        landcolor="#F0F0F0",
        showcountries=True,
        countrycolor="#ffffff",
        showframe=False,
    )
    if selected_regions:
        fig.update_geos(fitbounds="locations")
    else:
        fig.update_geos(fitbounds=None)
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=False,
        font=dict(family=FONT_FAMILY),
    )
    if fig.data:
        fig.update_traces(hovertemplate=HOVER_TEMPLATE, hoverlabel=HOVER_LABEL_STYLE)
    return fig


def make_trend_chart(filtered_df, selected_ideologies):
    ideologies = _resolve_ideologies(selected_ideologies)
    if not ideologies:
        fig = go.Figure()
    elif len(ideologies) == 1:
        ideology = ideologies[0]
        ideology_df = filtered_df[filtered_df["hog_ideology"] == ideology]
        yearly_counts = ideology_df.groupby("year").size().reset_index(name="count")
        fig = px.bar(
            yearly_counts,
            x="year",
            y="count",
            color_discrete_sequence=[color_map[ideology]],
            opacity=0.75,
        )
    else:
        grouped = filtered_df[filtered_df["hog_ideology"].isin(ideologies)]
        grouped = grouped.groupby(["year", "hog_ideology"]).size().reset_index(name="count")
        fig = px.bar(
            grouped,
            x="year",
            y="count",
            color="hog_ideology",
            barmode="group",
            color_discrete_map=color_map,
        )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=30, r=20, t=5, b=5),
        xaxis_title=None,
        yaxis_title=None,
        legend_title_text="Ideology" if len(ideologies) > 1 else None,
        showlegend=False,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family=FONT_FAMILY),
        bargap=0,
        bargroupgap=0,
    )
    fig.update_xaxes(showticklabels=False, fixedrange=True, showgrid=False)
    fig.update_yaxes(showticklabels=False, fixedrange=True, showgrid=False, zeroline=False)
    return fig


# --------------------------------------
# Layout helpers
# --------------------------------------
def build_ideology_options():
    return [
        {
            "label": html.Span(
                [
                    html.Span(
                        style={
                            "display": "inline-block",
                            "width": "12px",
                            "height": "12px",
                            "borderRadius": "2px",
                            "backgroundColor": color_map[ide],
                            "marginRight": "8px",
                        }
                    ),
                    html.Span(ide.capitalize()),
                ],
                style={"display": "flex", "alignItems": "center"},
            ),
            "value": ide,
        }
        for ide in valid_ideologies
    ]


def build_sidebar():
    region_values = sorted(map_df["region"].dropna().unique())
    region_options = [{"label": "All", "value": "all"}] + [
        {"label": region.title(), "value": region} for region in region_values
    ]
    democracy_options = [
        {"label": "Democracies", "value": "yes"},
        {"label": "Non-democracies", "value": "no"},
    ]

    return html.Div(
        id="sidebar",
        style={
            "width": "15%",
            "minWidth": "230px",
            "padding": "20px",
            "boxShadow": "2px 0 12px rgba(0,0,0,0.08)",
            "display": "flex",
            "flexDirection": "column",
            "gap": "18px",
            "overflowY": "auto",
            "overflowX": "hidden",
            "maxHeight": "100vh",
            "boxSizing": "border-box",
        },
        children=[
            html.H3("Ideology Atlas", style={"margin": "0"}),
            html.Div([
                html.Label("Region", style=SECTION_LABEL_STYLE),
                dcc.Checklist(
                    id="region_selector",
                    options=region_options,
                    value=[],
                    labelStyle=CHOICE_LABEL_STYLE,
                    inputStyle={"marginRight": "4px"},
                ),
            ]),
            html.Div([
                html.Label("Regime Type", style=SECTION_LABEL_STYLE),
                dcc.Checklist(
                    id="democracy_selector",
                    options=democracy_options,
                    value=[],
                    labelStyle=CHOICE_LABEL_STYLE,
                    inputStyle={"marginRight": "4px"},
                ),
            ]),
            html.Div([
                html.Label("Ideology", style=SECTION_LABEL_STYLE),
                dcc.Checklist(
                    id="ideology_selector",
                    options=build_ideology_options(),
                    value=[],
                    labelStyle={**CHOICE_LABEL_STYLE, "gap": "12px"},
                    inputStyle={"marginRight": "4px"},
                    className="ideology-checklist",
                ),
            ]),
        ],
    )


def build_overlay(overlay_id, backdrop_id, close_id, content, modal_id=None):
    return html.Div(
        id=overlay_id,
        className="summary-overlay hidden",
        children=[
            html.Div(id=backdrop_id, className="summary-backdrop", n_clicks=0),
            html.Div(
                id=modal_id,
                className="summary-modal",
                children=[
                    html.Button("×", id=close_id, className="summary-close", n_clicks=0),
                    content,
                ],
            ),
        ],
    )


# -------------------------------------------
# Dash Application Setup
# -------------------------------------------
# Configure Plotly graphs and initialize the Dash web application

# Graph configuration settings 
MAP_CONFIG = {
    "displaylogo": False,
    "displayModeBar": False,
    "responsive": True,
    "scrollZoom": False,
}
TREND_CONFIG = {"displayModeBar": False, "staticPlot": True, "responsive": True}
GRAPH_FULL_STYLE = {"width": "100%", "height": "100%"}

# Create the Dash app instance
app = Dash(__name__)


# Route for serving custom fonts
@app.server.route("/fonts/<path:filename>")
def serve_font(filename):
    """Serve custom font files from the fonts directory."""
    return send_from_directory(FONT_DIR, filename)

# Custom HTML template for the app - defines the structure of the index page
app.index_string = """
<!DOCTYPE html>
<html lang=\"en\" style=\"height:100%; overflow:hidden;\">
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            /* Import custom fonts */
            @font-face {
                font-family: "ABCMonumentGrotesk";
                src: url("/fonts/ABCMonumentGrotesk-Regular-Trial.otf") format("opentype");
                font-weight: 400;
                font-style: normal;
                font-display: swap;
            }

            @font-face {
                font-family: "ABCMonumentGrotesk";
                src: url("/fonts/ABCMonumentGrotesk-Bold-Trial.otf") format("opentype");
                font-weight: 600;
                font-style: normal;
                font-display: swap;
            }

            /* Styling for overlay modals (summary and info) */
            .summary-overlay {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                background: transparent;
                display: none;
                align-items: center;
                justify-content: center;
                padding: 20px;
                z-index: 1000;
                gap: 0;
            }

            .summary-overlay.hidden {
                display: none;
            }

            .summary-overlay.visible {
                display: flex;
            }

            .summary-backdrop {
                position: absolute;
                top: 0;
                left: 0;
                width: 100%;
                height: 100%;
                background: rgba(0, 0, 0, 0.35);
                cursor: pointer;
            }

            .summary-modal {
                background: #ffffff;
                border-radius: 0;
                max-width: 420px;
                width: 100%;
                padding: 24px;
                box-shadow: 0 18px 40px rgba(0, 0, 0, 0.25);
                position: relative;
                z-index: 1;
                font-family: "ABCMonumentGrotesk", Arial, sans-serif;
            }

            .summary-close {
                position: absolute;
                top: 12px;
                right: 12px;
                border: none;
                background: transparent;
                font-size: 20px;
                cursor: pointer;
                padding: 4px 8px;
                line-height: 1;
                border-radius: 0;
            }

            .summary-title {
                margin-top: 0;
                margin-bottom: 12px;
                font-size: 20px;
            }

            .summary-subtitle {
                margin: 18px 0 8px 0;
                font-size: 16px;
                font-weight: 600;
            }

            .summary-list {
                margin: 0;
                padding-left: 20px;
                color: #111;
                font-size: 13px;
            }

            .summary-instruction {
                margin-bottom: 6px;
            }

            .summary-field {
                display: flex;
                justify-content: space-between;
                font-size: 14px;
                margin-bottom: 6px;
                gap: 12px;
            }

            .summary-label {
                font-weight: 600;
                color: #333;
            }

            .summary-value {
                color: #111;
                text-align: right;
                flex: 1;
            }

            .summary-link {
                color: #0b62d6;
                text-decoration: underline;
            }

            .summary-link:hover {
                color: #063a7f;
            }

            .summary-empty {
                font-size: 14px;
                color: #555;
            }

            .info-button {
                position: absolute;
                top: 12px;
                right: 16px;
                background: #ffffff;
                border: 1px solid #c7c7c7;
                padding: 6px 10px;
                font-size: 13px;
                cursor: pointer;
                letter-spacing: 0.05em;
                box-shadow: 0 4px 10px rgba(0,0,0,0.15);
                border-radius: 0;
            }

            .info-button:hover {
                background: #f5f5f5;
            }

            #sidebar {
                box-shadow: none !important;
            }

            #histogram_container {
                box-shadow: none !important;
            }
        </style>
    </head>
    <body style=\"margin:0; height:100%; overflow:hidden;\">
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
"""

default_world_map_fig = make_world_map(stage=0)
default_trend_fig = make_trend_chart(df, valid_ideologies)

app.layout = html.Div(
    style={
        "display": "flex",
        "height": "100vh",
        "width": "100vw",
        "margin": 0,
        "overflow": "hidden",
        "backgroundColor": "#ffffff00",
        "fontFamily": FONT_FAMILY,
    },
    children=[
        dcc.Store(id="year_confirmed", data=False),
        build_overlay(
            overlay_id="summary_overlay",
            backdrop_id="summary_backdrop",
            close_id="summary_close",
            content=html.Div(id="summary_modal_content", className="summary-modal-content"),
            modal_id="summary_modal",
        ),
        build_overlay(
            overlay_id="info_overlay",
            backdrop_id="info_backdrop",
            close_id="info_close",
            content=_build_info_card(),
            modal_id="info_modal",
        ),
        build_sidebar(),
        html.Div(
            id="main_panel",
            style={
                "flex": "1 1 auto",
                "height": "100vh",
                "display": "flex",
                "flexDirection": "column",
                "overflow": "hidden",
            },
            children=[
                html.Div(
                    id="map_container",
                    style={
                        "flex": "1 1 85%",
                        "width": "100%",
                        "height": "100%",
                        "position": "relative",
                    },
                    children=[
                        dcc.Graph(
                            id="world_map",
                            figure=default_world_map_fig,
                            config=MAP_CONFIG,
                            style=GRAPH_FULL_STYLE,
                        ),
                        html.Button("info", id="info_button", className="info-button", n_clicks=0),
                    ],
                ),
                html.Div(
                    id="histogram_container",
                    style={
                        "flex": "0 0 25%",
                        "width": "100%",
                        "padding": "10px 30px 15px 30px",
                        "boxSizing": "border-box",
                        "backgroundColor": "transparent",
                        "boxShadow": "none",
                        "display": "flex",
                        "flexDirection": "column",
                    },
                    children=[
                        dcc.Graph(
                            id="trend_chart",
                            figure=default_trend_fig,
                            config=TREND_CONFIG,
                            style={"flex": "1 1 auto", **GRAPH_FULL_STYLE},
                        ),
                        html.Div(
                            style={"paddingTop": "6px"},
                            children=[
                                dcc.Slider(
                                    id="year_slider",
                                    min=min_year if min_year is not None else 0,
                                    max=max_year if max_year is not None else 0,
                                    value=max_year if max_year is not None else 0,
                                    included=False,
                                    marks=year_marks if year_marks else {},
                                    step=1,
                                    tooltip={"always_visible": False, "placement": "bottom"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# -------------------------------------------
# Dash Callbacks - Interactivity 
# -------------------------------------------
# These callbacks respond to user interactions and update the dashboard

# Callback 1: Track when user has selected a year
@app.callback(Output("year_confirmed", "data"), Input("year_slider", "value"))
def flag_year_confirmation(selected_year):
    """Set flag to True once user has interacted with the year slider."""
    if ctx.triggered_id is None:
        return False
    return True


# Callback 2: Toggle the information modal overlay
@app.callback(
    Output("info_overlay", "className"),
    Input("info_button", "n_clicks"),
    Input("info_close", "n_clicks"),
    Input("info_backdrop", "n_clicks"),
    prevent_initial_call=True,
)
def toggle_info_modal(info_click, close_click, backdrop_click):
    """Open/close info modal when user clicks info button, close button, or backdrop."""
    trigger = ctx.triggered_id
    if trigger in {"info_close", "info_backdrop"}:
        return "summary-overlay hidden"
    if trigger == "info_button":
        return "summary-overlay visible"
    raise PreventUpdate


# Callback 3: Show summary modal when user clicks on a country in the map
@app.callback(
    Output("summary_modal_content", "children"),
    Output("summary_overlay", "className"),
    Input("world_map", "clickData"),
    Input("summary_close", "n_clicks"),
    Input("summary_backdrop", "n_clicks"),
    State("year_slider", "value"),
    prevent_initial_call=True,
)
def toggle_summary_modal(click_data, close_clicks, backdrop_clicks, selected_year):
    """Display country information when clicked on map, or close modal when user clicks close/backdrop."""
    trigger = ctx.triggered_id

    if trigger in {"summary_close", "summary_backdrop"}:
        return [], "summary-overlay hidden"

    if trigger == "world_map" and click_data:
        # Extract country name and year from click event
        point = (click_data.get("points") or [{}])[0]
        country = point.get("location") or point.get("hovertext")
        year_value = int(selected_year) if selected_year is not None else None
        row = _extract_summary_row(country, year_value)
        content = _build_summary_card(country, year_value, row)
        return content, "summary-overlay visible"

    raise PreventUpdate


# Callback 4: Update world map based on all filter selections
@app.callback(
    Output("world_map", "figure"),
    Input("region_selector", "value"),
    Input("democracy_selector", "value"),
    Input("year_slider", "value"),
    Input("ideology_selector", "value"),
    Input("year_confirmed", "data"),
)
def update_world_map(selected_regions, selected_democracy, selected_year, selected_ideologies, year_confirmed):
    """
    Regenerate world map visualization when filters change.
    Only shows full color map when user has locked a year (year_confirmed=True).
    """
    regions = _resolve_regions(selected_regions)
    has_region_selection = bool(selected_regions)
    ideology_filters = _resolve_ideologies(selected_ideologies)
    
    # Determine completion stage (1-4) based on what user has selected
    stage = _compute_stage(has_region_selection, selected_democracy, ideology_filters, year_confirmed)
    
    # Only apply year filter when at final stage and year is confirmed
    year_value = int(selected_year) if (stage == 4 and selected_year is not None) else None
    
    return make_world_map(
        stage,
        regions,
        year_value,
        selected_democracy,
        ideology_filters,
        has_region_selection,
    )


# Callback 5: Update trend chart based on region, democracy, and ideology filters
@app.callback(
    Output("trend_chart", "figure"),
    Input("region_selector", "value"),
    Input("democracy_selector", "value"),
    Input("ideology_selector", "value"),
)
def update_chart(selected_regions, selected_democracy, selected_ideologies):
    """
    Regenerate trend chart showing ideology distribution over time.
    Filters data by selected regions and democracy status.
    """
    filtered = df
    regions = _resolve_regions(selected_regions)
    
    # Apply region filter
    if regions:
        filtered = filtered[filtered["region"].isin(regions)]
    
    # Apply democracy filter
    filtered = _apply_multi_filter(filtered, "democracy_flag", selected_democracy)

    fig = make_trend_chart(filtered, selected_ideologies)
    return fig


# -------------------------------------------
# Start the app
# -------------------------------------------
if __name__ == "__main__":
    # Run the Dash app with debug mode enabled
    app.run(debug=True)

