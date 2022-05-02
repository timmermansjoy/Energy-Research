from multiprocessing.dummy import freeze_support
import os
import sys
from dotenv import load_dotenv
load_dotenv()
os.environ['WANDB_NOTEBOOK_NAME'] = 'pytorch_stats_own_data.ipynb'
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score, rmse
from darts import TimeSeries

import helper
import glob

from pytorch_lightning.loggers import WandbLogger
wandb_logger = WandbLogger(project="Digital-Energy")


AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()
MODEL_NAME = sys.argv[1]

print(f"Available GPUs: {AVAILABLE_GPUS}")
print(f"Available CPUs: {AVAILABLE_CPUS}")


def main():
    ## ---- LOAD DATA ---- ##

    ## data
    my_time_series_dataset = []
    for x in sorted(glob.glob("../../Data/london_clean/*.csv"))[:50]:
        df = pd.read_csv(f'{x}')
        df["DateTime"] = pd.to_datetime(df['DateTime'])
        df = df.groupby(pd.Grouper(key='DateTime', freq='1D')).max("KWHhh").round(3).reset_index()
        series = TimeSeries.from_dataframe(df, time_col='DateTime', value_cols='KWHhh', fill_missing_dates=True)
        my_time_series_dataset.append(series)

    ## sets
    training_sets = []
    validation_sets = []
    for x in my_time_series_dataset:
        train, val = series.split_after(0.90)
        training_sets.append(train)
        validation_sets.append(val)



    ## ---- MODEL ---- ##
    from pytorch_lightning.callbacks.early_stopping import EarlyStopping

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=10,
        verbose=False,
        mode="min"
        )

    ## check if model already exists


    model_nbeats = NBEATSModel(
        input_chunk_length=30,
        output_chunk_length=7,
        generic_architecture=False,
        num_stacks=15,
        num_blocks=5,
        num_layers=3,
        layer_widths=512,
        n_epochs=40,
        nr_epochs_val_period=1,
        batch_size=64,
        work_dir="../../Models",
        save_checkpoints=True,
        model_name=MODEL_NAME,
        pl_trainer_kwargs={
        "accelerator": "gpu",
        "devices": 1, # use all available GPUs (this can be done due to this script not being in a notebook)
        #"strategy": "ddp",
        "logger": wandb_logger,
        # "callbacks": [early_stop_callback]
        },
    )


    ## ---- TRAIN ---- ##

    model_nbeats.fit(series=training_sets, val_series=validation_sets, verbose=True, num_loader_workers=AVAILABLE_CPUS)



    ## ---- EVALUATE ---- ##

    ## test data
    for i in range(600,620):

        df = pd.read_csv(f'../../Data/london_clean/cleaned_household_MAC000{i}.csv')
        series = TimeSeries.from_dataframe(df, value_cols=['KWHhh'], time_col="DateTime", fill_missing_dates=True)
        series = series[-150:]


        pred_series = model_nbeats.historical_forecasts(
            series,
            forecast_horizon=1,
            stride=1,
            retrain=False,
        )

        print(f"rmse: {rmse(series, pred_series)}.")
        print(f"R2 score: {r2_score(series, pred_series)}.")

        helper.display_forecast(pred_series, series, "1 day", save=True, fig_name=f"household_MAC000{i}", model_name="non-generic", fig_size=(20,10))



if __name__ == "__main__":
    freeze_support()
    main()
