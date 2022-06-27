import imp
import pandas as pd
import matplotlib.pyplot as plt
from darts.metrics import mape, r2_score, mse
from darts.dataprocessing.transformers import Scaler
import os

import glob
from darts import TimeSeries
from tqdm.contrib.concurrent import process_map
import wandb
import numpy as np
from tqdm import tqdm
import torch
import time


AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()


def find_gaps(df, Date_col_name="DateTime", delta=15):
    from datetime import datetime, timedelta
    if isinstance(df.index, pd.DatetimeIndex):
        df["DateTime"] = pd.to_datetime(df.index)
    # df["DateTime"] = pd.to_datetime(df)
    deltas = df[Date_col_name].diff()[1:]
    gaps = deltas[deltas > timedelta(minutes=delta)]
    # Print results
    if not gaps.empty:
        print(f'{len(gaps)} gaps with median gap duration: {gaps.median()}')
        print(gaps)
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
    plt.rcParams["figure.facecolor"] = "white"
    plt.ylim(0, 3)

    if start_date is not None:
        actual_series = actual_series.drop_before(start_date)
    actual_series.univariate_component(0).plot(label="actual")
    pred_series.plot(label=(f"historic {forecast_type} forecasts with {mse(actual_series.univariate_component(0), pred_series).round(4)} MSE"))
    if fig_name is not None:
        plt.title(f"{fig_name} MSE: {mse(actual_series.univariate_component(0), pred_series)}")
    else:
        plt.title(f"MSE: {mse(actual_series.univariate_component(0), pred_series)}")
    plt.legend()
    if save:
        if fig_name and model_name:
            # check if directory exists and create if not
            if not os.path.exists(f"../../../Plots/{model_name}"):
                os.makedirs(f"../../../Plots/{model_name}")
            plt.savefig(f"../../../Plots/{model_name}/{fig_name}.png")
        elif fig_name:
            plt.savefig(f"../../../Plots/{fig_name}.png")
        else:
            plt.savefig(f"../../../Plots/{forecast_type}_forecast.png")
    return plt


def eval(model, start=3000, num_examples=10, save=True, log=True, data_normalized=False, future_covariates=False):
    model.trainer_params.update({"devices": 1, "strategy": None})
    scalar = Scaler()
    weather = pd.read_csv("../../../Data/London_weather_2011-2014.csv")
    weather["DateTime"] = pd.to_datetime(weather["DateTime"])
    for _, x in tqdm(enumerate(sorted(glob.glob("../../../Data/london_clean/*.csv"))[start:start + num_examples])):
        df = pd.read_csv(x)
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = weather.merge(df, on="DateTime", how="right")
        df.fillna(method="ffill" ,inplace=True)
        series = TimeSeries.from_dataframe(df, time_col="DateTime", value_cols=["KWHhh"], freq="30min", fill_missing_dates=True, fillna_value=True).astype(np.float32)
        covarient = TimeSeries.from_dataframe(df, time_col="DateTime", value_cols=["Temperature_C", "Precip_Rate_mm"], freq="30min", fill_missing_dates=True, fillna_value=True).astype(np.float32)
        MID = len(series) // 2
        if data_normalized:
            series = scalar.fit_transform(series)
        series = series[MID:MID+600]

        # time how fast the model predicts
        start_time = time.time()

        pred_series = model.historical_forecasts(
            series,
            forecast_horizon=1,
            future_covariates=covarient if future_covariates else None,
            stride=1,
            retrain=False,
            verbose=True,
        )

        duration = time.time() - start_time

        if data_normalized:
            series = scalar.inverse_transform(series)
            pred_series = scalar.inverse_transform(pred_series)

        print(f"rmse: {mse(series, pred_series)}.")
        print(f"R2 score: {r2_score(series, pred_series)}.")

        fig = display_forecast(pred_series, series, "1 day", save=save, fig_name=f"{x.split('/')[-1].split('.')[0].split('_')[-1]}", model_name=model.model_name, fig_size=(20,10))
        if log:
            wandb.define_metric("mse", summary="min")
            wandb.define_metric("prediction_time", summary="mean")
            wandb.log({
                    "mse": mse(series, pred_series),
                    "r2": r2_score(series, pred_series),
                    "prediction_time": duration,
                    "result": fig
            })
