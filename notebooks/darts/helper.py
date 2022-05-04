import pandas as pd
import matplotlib.pyplot as plt
from darts.metrics import mape, r2_score
import os


def find_gaps(df, Date_col_name="DateTime"):
    from datetime import datetime, timedelta

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    deltas = df["DateTime"].diff()[1:]
    gaps = deltas[deltas > timedelta(minutes=30)]
    # Print results
    if not gaps.empty:
        # print(f'{len(gaps)} gaps with median gap duration: {gaps.median()}')
        # print(gaps)
        return gaps
    return pd.DataFrame()


def display_forecast(
    pred_series,
    actual_series,
    forecast_type,
    start_date=None,
    save=False,
    fig_size=(10, 5),
    fig_name=None,
    model_name=None,
):
    """
    Helper function to plot the forecast and save the forecast in a single function for easier use.
    Args: pred_series (darts.TimeSeries): The forecasted series.
          actual_series (darts.TimeSeries): The original series.
          forecast_type (str): the the of forercasting that has been done IE: "1 Day".
          start_date (str): The start date of the forecast.
          save (bool): Whether to save the figure.
          fig_size (tuple): The size of the figure.
          fig_name (str): The name of the figure.
          model_name (str): The name of the model.
    """

    plt.figure(figsize=fig_size)
    plt.ylim(0, 3)
    plt.rcParams["figure.facecolor"] = "white"

    if start_date is not None:
        actual_series = actual_series.drop_before(start_date)
    actual_series.univariate_component(0).plot(label="actual")
    pred_series.plot(label=("historic " + forecast_type + " forecasts"))
    if fig_name is not None:
        plt.title(f"{fig_name} MSE: {r2_score(actual_series.univariate_component(0), pred_series)}")
    else:
        plt.title(f"MSE: {r2_score(actual_series.univariate_component(0), pred_series)}")
    plt.legend()
    if save:
        if fig_name and model_name:
            # check if directory exists and create if not
            if not os.path.exists(f"../../Plots/{model_name}"):
                os.makedirs(f"../../Plots/{model_name}")
            plt.savefig(f"../../Plots/{model_name}/{fig_name}.png")
        elif fig_name:
            plt.savefig(f"../../Plots/{fig_name}.png")
        else:
            plt.savefig(f"../../Plots/{forecast_type}_forecast.png")
