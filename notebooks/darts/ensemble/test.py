
from multiprocessing.dummy import freeze_support
import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)


from dotenv import load_dotenv
load_dotenv()
os.environ['WANDB_NOTEBOOK_NAME'] = 'n-beats.ipynb'
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.models import NBEATSModel, TFTModel, RegressionEnsembleModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score, rmse, mse

from darts import TimeSeries

from darts.datasets import EnergyDataset

import helper
import glob
import wandb

from pytorch_lightning.loggers import WandbLogger

from tqdm.contrib.concurrent import process_map
import tqdm


AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "ensamble"

print(f"Available GPUs: {AVAILABLE_GPUS}")
print(f"Available CPUs: {AVAILABLE_CPUS}")

# wandb.init(project="Digital-Energy", name=MODEL_NAME)


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

def splitter():
    file_list = sorted(glob.glob("../../../Data/london_clean/*.csv"))[:100]
    if file_list == []:
        raise Exception("No files found")
    return process_map(reader, file_list, chunksize=20)



def main():
    my_time_series_dataset = splitter()
    values_series = [x[0] for x in my_time_series_dataset]
    covariance_series = [x[1] for x in my_time_series_dataset]


    wandb_logger = WandbLogger(project="Digital-Energy", log_model=True)


    tft = TFTModel(
        input_chunk_length=96,
        output_chunk_length=96,
        hidden_size=64,
        lstm_layers=2,
        num_attention_heads=4,
        dropout=0.04,
        batch_size=512,
        n_epochs=5,
        add_relative_index=True,
        work_dir="../../../Models",
        save_checkpoints=False,
        pl_trainer_kwargs={
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "accelerator": "gpu",
        "devices": 2,
        "logger": wandb_logger,
        },
        # likelihood=QuantileRegression(
        #     quantiles=quantiles
        # ),  # QuantileRegression is set per default
        random_state=42,
    )

    n_beats = NBEATSModel(
        input_chunk_length=3,
        output_chunk_length=1,
        generic_architecture=False,
        #num_stacks=10,
        num_blocks=3,
        num_layers=5,
        layer_widths=512,
        n_epochs=2,
        nr_epochs_val_period=1,
        batch_size=512,
        work_dir="../../../Models",
        save_checkpoints=False,
        # model_name=MODEL_NAME,
        pl_trainer_kwargs={
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "accelerator": "gpu",
        "devices": 2,
        "logger": wandb_logger,
        },
        random_state=42
    )


    ensamble = RegressionEnsembleModel(forecasting_models=[n_beats, tft], regression_train_n_points=96)

    ensamble.fit(values_series)



    START = 3000
    SAMPLES = 10
    for i, x in enumerate(sorted(glob.glob("../../../Data/london_clean/*.csv"))[START:START+SAMPLES]):

        df = pd.read_csv(x)
        df["DateTime"] = pd.to_datetime(df["DateTime"])
        df = weather.merge(df, on="DateTime", how="right")
        df.fillna(method="ffill" ,inplace=True)
        series = TimeSeries.from_dataframe(df, time_col="DateTime", value_cols=["KWHhh"], freq="30min", fill_missing_dates=True, fillna_value=True).astype(np.float32)
        covarient = TimeSeries.from_dataframe(df, time_col="DateTime", value_cols=["Temperature_C", "Humidity_%", "Dew_Point_C"], freq="30min", fill_missing_dates=True, fillna_value=True).astype(np.float32)
        MID = len(series)//2
        series = series[MID:MID+600]



        pred_series = ensamble.historical_forecasts(
            series,
            # future_covariates=covarient,
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=True,
        )

        print(f"rmse: {rmse(series, pred_series)}.")
        print(f"R2 score: {r2_score(series, pred_series)}.")

        # fig = helper.display_forecast(pred_series, series, "1 day", save=False, fig_name=f"{x.split('/')[-1]}-test", fig_size=(20,10))
        plt.figure(figsize=(20, 20))
        plt.ylim(0, 3)
        plt.title(f"{MODEL_NAME} - {x.split('/')[-1]} - MSE: {mse(series, pred_series)}")
        series.plot()
        pred_series.plot()
        fig = plt
        wandb.log({
                "mape": mape(series, pred_series),
                "mse": mse(series, pred_series),
                "rmse": rmse(series, pred_series),
                "r2": r2_score(series, pred_series),
                "result": fig
        })
        plt.savefig(f"../../../Plots/{MODEL_NAME}/{x.split('/')[-1]}.png")

if __name__ == "__main__":
    freeze_support()
    main()
