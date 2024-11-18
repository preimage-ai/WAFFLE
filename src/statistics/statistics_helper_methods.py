import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go


PLOTS_PATH = "src/statistics/plots/"


def get_sorted_counted_values(df, column_name):
    return sorted(
        Counter(df[column_name].dropna().str.lower()).items(),
        key=lambda x: x[1],
        reverse=True,
    )


def count_and_save_bars_plot_for_column(
    df, column_name, plot_name, output_path=PLOTS_PATH
):
    size = len(df[column_name].unique())
    save_text_frequency_bar_plot(
        get_sorted_counted_values(df, column_name),
        plot_name + f" {size} in total",
        output_path,
    )


def save_text_frequency_bar_plot(text_frequencies, plot_name, output_path=PLOTS_PATH):
    count = sum([frequency for _, frequency in text_frequencies])
    sorted_items = sorted(text_frequencies, key=lambda x: x[1], reverse=True)
    top_30_items = dict(sorted_items[:30])
    # top_30_items = {key: value / count * 100 for key, value in top_30_items.items()}
    df = pd.DataFrame(top_30_items.items(), columns=["label", "frequency"])
    fig, axes = plt.subplots(1, 1, figsize=(15, 20))
    axes.set_title(plot_name, fontsize=20)
    axes.tick_params(axis="both", labelsize=16)
    axes.set_xlabel("Frequency", fontsize=18)  # set xlabel fontsize to 18
    axes.set_ylabel("Label", fontsize=18)  # set ylabel fontsize to 18
    fig.subplots_adjust(left=0.5)
    sns.set(rc={"axes.xmargin": 0.05})
    sns.barplot(ax=axes, x="frequency", y="label", data=df, width=0.5)
    for i, v in enumerate(df["frequency"]):
        axes.text(
            v,
            i + 0.25,
            f"{(v / count *100):.2f}%, {v}",
            color="black",
            fontweight="bold",
            fontsize=16,
        )
    fig.savefig(output_path + f"/{plot_name}.png")
    plt.clf()
    plt.close()


def plot_map(df, output):
    countries = sorted(
        Counter(df["country"].dropna().str.lower()).items(),
        key=lambda x: x[1],
        reverse=True,
    )
    plotly_df = pd.read_csv(
        "https://raw.githubusercontent.com/plotly/datasets/master/fips-unemp-16.csv",
        dtype={"fips": str},
    )
    fig = px.choropleth_mapbox(
        plotly_df,
        geojson=[country[0] for country in countries],
        locations="fips",
        color="unemp",
        color_continuous_scale="Viridis",
        range_color=(0, 12),
        mapbox_style="carto-positron",
        zoom=2,
        # center = {"lat": 37.0902, "lon": -95.7129},
        opacity=0.5,
        labels={"count": "building count"},
    )
    fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
    fig.show()
    fig.write_image(output)

def get_column_counts(df, column_name, column_vals):
    column_counts = sorted(
        Counter(df[column_name]).items(),
        key=lambda x: x[1],
        reverse=True,
    )
    res = []
    for val in column_vals:
        for column_count in column_counts:
            if column_count[0] == val:
                res.append((column_count[0], column_count[1] / len(df)))
                break
    return res

def plot_bars(df, column_name, top_k=60, font_size=20):
    # Count the occurrences of each building type and sort in descending order
    column_counts = (
        df[column_name].value_counts().sort_values(ascending=False).head(top_k)
    )

    colors = px.colors.qualitative.Plotly * (top_k // 10)

    # Create a Figure
    fig = go.Figure(
        data=[
            go.Bar(
                x=column_counts.index,
                y=column_counts.values,
                marker_color=colors,  # different color for each bar
            )
        ]
    )

    fig.update_layout(
        barmode="group",
        xaxis_tickangle=-70,
        font=dict(size=font_size),
        plot_bgcolor="white",
        height=800,
    )
    fig.update_yaxes(type="log",  tickmode='linear', dtick=1)
    fig.show()
    pass

import pycountry
def get_iso_country(country):
    iso = pycountry.countries.get(name=country)
    unknowns_to_iso = {
        'United States of America': 'USA',
        'Iran': 'IRN',
        'Taiwan': 'TWN',
        'Syria': 'SYR',
        'Czech Republic': 'CZE',
        'Northern Cyprus': 'CYP',
        'Palestine': 'ISR',
        'Russia': 'RUS',
        'Jammu & Kashmir': 'IND',
        'South Korea': 'KOR',
        'Venezuela': 'VEN',
        'Bolivia': 'BOL',
        'Wales': 'GBR',
        'Channel Islands': 'GBR',
        'Vietnam': 'VNM',
    }
    if iso and isinstance(iso.alpha_3, str):
        return iso.alpha_3
    elif country in unknowns_to_iso:
        return unknowns_to_iso[country]
    return 'Unknown'


def plot_countries(df, name):
    df['ISO_country'] = df['country'].apply(get_iso_country)
    country_counts = df['ISO_country'].value_counts()
    log_country_counts = np.log(country_counts)
    colorscale = 'Blues' if name == 'Training' else 'Oranges'
    colorbar_x = 0.95 if name == 'Training' else 1.0
    
    trace = go.Choropleth(
        locations=log_country_counts.index,  # country codes
        z=log_country_counts.values,  # counts
        text=log_country_counts.index,  # country codes
        colorscale=colorscale,
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar=dict(title=name,
                      x=colorbar_x),
        name=f'{name} Set'
    )
    return trace


def plot_countries_combined(df_train, df_test):
    trace_train = plot_countries(df_train, 'Training')
    trace_test = plot_countries(df_test, 'Test')

    # Create a Figure with both Choropleth maps
    fig = go.Figure(data=[trace_train, trace_test])

    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=False,
            projection_type='equirectangular'
        ),
        font=dict(size=30),
    )

    fig.show()


def plot_histogram(df, column_name, threshold, log=False):
    # Create a copy of the DataFrame to avoid modifying the original one
    df_copy = df.copy()

    # Add up all the long tail numbers
    tail_mask = df_copy[column_name] > threshold
    df_copy.loc[tail_mask, column_name] = threshold

    # Create a Histogram
    fig = go.Figure(
        data=[
            go.Histogram(
                x=df_copy[column_name],
                nbinsx=50,  # change this value to adjust the number of bins
                marker_color='indianred',
            )
        ]
    )

    fig.update_layout(
        title_text='Histogram of ' + column_name,  # title of plot
        xaxis_title_text=column_name,  # xaxis label
        yaxis_title_text='Count',  # yaxis label
        bargap=0.1,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1,  # gap between bars of the same location coordinates
        plot_bgcolor="white",
        font=dict(size=30),
    )
    if log:
        fig.update_yaxes(type="log",  tickmode='linear', dtick=1)

    fig.show()


if __name__ == "__main__":
    import pandas as pd
    import json
    
    df = pd.read_csv("data/clean_csv/dataset.csv")

    df_grounded = df[df["grounded_unified_fn"].notna()]
    labels = []

    df_grounded["grounded_unified_fn"].apply(
        lambda x: labels.extend(json.load(open(x)).keys())
    )
    
    labels_df = pd.DataFrame(labels, columns=['label'])
    plot_bars(labels_df, 'label', top_k=40)

    df_all = pd.read_csv("data/csvs_v2/large_dataset_with_country.csv")
    df_train = pd.read_csv("data/csvs_v2/large_dataset_train.csv")
    df_test = pd.read_csv("data/csvs_v2/large_dataset_test.csv")
    plot_countries_combined(df_train, df_test)



    plot_bars(df_all, df_train, df_test, "building_type", top_k=40)

    columns = ["building_type"]

    plot_map(df_all, "a.png")

    for column in columns:
        count_and_save_bars_plot_for_column(
            df_all, column, f"{column}_all", output_path=PLOTS_PATH
        )
        count_and_save_bars_plot_for_column(
            df_train, column, f"{column}_train", output_path=PLOTS_PATH
        )
        count_and_save_bars_plot_for_column(
            df_test, column, f"{column}_test", output_path=PLOTS_PATH
        )
