import enum
from multiprocessing.dummy import freeze_support
import os
import sys
from dotenv import load_dotenv
load_dotenv()
os.environ['WANDB_NOTEBOOK_NAME'] = 'pytorch_stats_own_data.ipynb'
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')

nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.utils.likelihood_models import QuantileRegression
from darts.metrics import mape, r2_score, rmse, mse
from darts import TimeSeries

import helper
import glob

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, ddp2
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from tqdm.contrib.concurrent import process_map
import tqdm


AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()
TRAINING_DATA_PATH = "../../../Data/london_clean/*.csv"
HOUSEHOLDS = 1000

print(f"Available GPUs: {AVAILABLE_GPUS}")
print(f"Available CPUs: {AVAILABLE_CPUS}")


DEFAULT_VALUES = dict(
    hidden_size=89,
    lstm_layers=7,
    num_attention_heads=6,
    dropout=0.05,
)

lowest_q, low_q, high_q, highest_q = 0.01, 0.1, 0.9, 0.99
label_q_outer = f"{int(lowest_q * 100)}-{int(highest_q * 100)}th percentiles"
label_q_inner = f"{int(low_q * 100)}-{int(high_q * 100)}th percentiles"
weather = pd.read_csv("../../../Data/London_weather_2011-2014.csv")
weather["DateTime"] = pd.to_datetime(weather["DateTime"])


def reader(x):
    df = pd.read_csv(x)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = weather.merge(df, on="DateTime", how="right")
    df.fillna(method="ffill" ,inplace=True)
    series = TimeSeries.from_dataframe(df, time_col="DateTime", value_cols=["KWHhh"], freq="30min", fill_missing_dates=True, fillna_value=True).astype(np.float32)
    covarient = TimeSeries.from_dataframe(df, time_col="DateTime", value_cols=["Temperature_C", "Humidity_%", "Dew_Point_C"], freq="30min", fill_missing_dates=True, fillna_value=True).astype(np.float32)
    list = [series, covarient]
    return list


def main():
    torch.cuda.empty_cache()
    wandb.init(project="Digital-Energy", config=DEFAULT_VALUES)
    config = wandb.config

    ## ---- LOAD DATA ---- ##

    ## data
    print("Loading data...")
    file_list = sorted(glob.glob(TRAINING_DATA_PATH))[:HOUSEHOLDS]
    if file_list == []:
        raise Exception("No files found")
    series_list = process_map(reader, file_list, chunksize=5)
    values_series = [x[0] for x in series_list]
    covariance_series = [x[1] for x in series_list]

    ## sets
    print("Splitting data...")
    training_sets = []
    validation_sets = []
    for x in values_series:
        train, val = x.split_after(0.85)
        training_sets.append(train)
        validation_sets.append(val)


    ## normalize the data
    print("Normalizing data...")
    scaler = Scaler()

    # create train and validation data
    train_transformed = []
    val_transformed = []
    for x in training_sets:
        train_transformed.append(scaler.fit_transform(x))
    for x in validation_sets:
        val_transformed.append(scaler.fit_transform(x))

    ## covariate series
    cov_transformed = []
    for x in covariance_series:
        cov_transformed.append(scaler.fit_transform(x))


    ## ---- MODEL ---- ##
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=10,
        verbose=False,
        mode="min"
        )

    wandb_logger = WandbLogger(project="Digital-Energy", log_model="all")
    # checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="max")


    quantiles = [
        0.01,
        0.05,
        0.1,
        0.15,
        0.2,
        0.25,
        0.3,
        0.4,
        0.5,
        0.6,
        0.7,
        0.75,
        0.8,
        0.85,
        0.9,
        0.95,
        0.99,
    ]

    encoders = {
    "datetime_attribute": {"future": ["DateTime"], "past": ["DateTime"]},
    "position": {"past": ["absolute"], "future": ["relative"]},
    "transformer": Scaler(),
    }


    input_chunk_length = 96
    forecast_horizon = 96

    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        hidden_size=config.hidden_size,
        lstm_layers=config.lstm_layers,
        num_attention_heads=config.num_attention_heads,
        dropout=config.dropout,
        batch_size=512,
        n_epochs=50,
        add_relative_index=True,
        add_encoders=None,
        work_dir="../../Models",
        save_checkpoints=False,
        # model_name=MODEL_NAME,
        pl_trainer_kwargs={
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "accelerator": "gpu",
        "devices": 2,
        "strategy": DDPStrategy(find_unused_parameters=False),
        "logger": wandb_logger,
        "callbacks": [early_stop_callback]
        },
        likelihood=QuantileRegression(
            quantiles=quantiles
        ),  # QuantileRegression is set per default
        # loss_fn=MSELoss(),
        random_state=42,
    )


    print("Training model...")
    model.fit(series=training_sets, val_series=validation_sets, future_covariates=cov_transformed, val_future_covariates=cov_transformed, num_loader_workers=AVAILABLE_CPUS, max_samples_per_ts=2000)


    ## ---- EVALUATE ---- ##
    print("Evaluating...")
            ## test data
    START = 3000
    for i, x in enumerate(sorted(glob.glob(TRAINING_DATA_PATH))[START:START+2]):

        df = pd.read_csv(x)
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = weather.merge(df, on="DateTime", how="right")
        df.fillna(method="ffill" ,inplace=True)
        series = TimeSeries.from_dataframe(df, time_col="DateTime", value_cols=["KWHhh"], freq="30min", fill_missing_dates=True, fillna_value=True).astype(np.float32)
        covarient = TimeSeries.from_dataframe(df, time_col="DateTime", value_cols=["Temperature_C", "Humidity_%", "Dew_Point_C"], freq="30min", fill_missing_dates=True, fillna_value=True).astype(np.float32)
        series = series[5000:5500]


        pred_series = model.historical_forecasts(
            series,
            future_covariates=covarient,
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=True,
        )

        print(f"rmse: {rmse(series, pred_series)}.")
        print(f"R2 score: {r2_score(series, pred_series)}.")

        # fig = helper.display_forecast(pred_series, series, "1 day", save=False, fig_name=f"{x.split('/')[-1]}-test", fig_size=(20,10))
        plt.figure(figsize=(20, 15))
        plt.ylim(0, 3)
        series.plot()
        pred_series.plot(low_quantile=lowest_q, high_quantile=highest_q, label=label_q_outer)
        fig = plt
        wandb.log({
                "mape": mape(series, pred_series),
                "mse": mse(series, pred_series),
                "rmse": rmse(series, pred_series),
                "r2": r2_score(series, pred_series),
                "result": fig
        })


if __name__ == "__main__":
    freeze_support()
    main()
