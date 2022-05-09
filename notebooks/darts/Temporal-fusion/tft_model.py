from multiprocessing.dummy import freeze_support
import os
import sys
from dotenv import load_dotenv
load_dotenv()
os.environ['WANDB_NOTEBOOK_NAME'] = 'n-beats.ipynb'
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler
from darts.metrics import mape, r2_score, rmse
from darts.utils.likelihood_models import QuantileRegression

import helper
import glob
import progressbar

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping


AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()

print(f"Available GPUs: {AVAILABLE_GPUS}")
print(f"Available CPUs: {AVAILABLE_CPUS}")

MODEL_NAME = sys.argv[1] if sys.argv[1] is not None else 'tft-multi-gpu'



def main():
    ## ---- LOAD DATA ---- ##

    ## data
    my_time_series_dataset = []
    for x in progressbar.progressbar(sorted(glob.glob("../../Data/london_clean/*.csv"))[:1000]):
        df = pd.read_csv(f'{x}')
        df["DateTime"] = pd.to_datetime(df['DateTime'])
        series = TimeSeries.from_dataframe(df, time_col='DateTime', value_cols='KWHhh').astype(np.float32)
        my_time_series_dataset.append(series)

    ## sets
    training_sets = []
    validation_sets = []
    for x in my_time_series_dataset:
        train, val = series.split_after(0.85)
        training_sets.append(train)
        validation_sets.append(val)


    ## ---- MODEL ---- ##
    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=10,
        verbose=False,
        mode="min"
        )

    wandb_logger = WandbLogger(project="Digital-Energy", name=MODEL_NAME, log_model="all")
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="max")


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

    input_chunk_length = 96
    forecast_horizon = 1

    model = TFTModel(
        input_chunk_length=input_chunk_length,
        output_chunk_length=forecast_horizon,
        hidden_size=64,
        lstm_layers=2,
        num_attention_heads=4,
        dropout=0.1,
        batch_size=512,
        n_epochs=100,
        add_relative_index=True,
        add_encoders=None,
        work_dir="../../Models",
        save_checkpoints=True,
        model_name=MODEL_NAME,
        pl_trainer_kwargs={
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "accelerator": "gpu",
        "devices": 2,
        "strategy": "ddp",
        "logger": wandb_logger,
        # "callbacks": [early_stop_callback, checkpoint_callback]
        },
        likelihood=QuantileRegression(
            quantiles=quantiles
        ),  # QuantileRegression is set per default
        # loss_fn=MSELoss(),
        random_state=42,
    )



    model.fit(series=training_sets, val_series=validation_sets, num_loader_workers=AVAILABLE_CPUS, max_samples_per_ts=500)



    START = 3000
    for i, x in enumerate(sorted(glob.glob("../../Data/london_clean/*.csv"))[START:START+10]):

        df = pd.read_csv(x)
        df["DateTime"] = pd.to_datetime(df['DateTime'])
        series = TimeSeries.from_dataframe(df, value_cols=['KWHhh'], time_col="DateTime", fill_missing_dates=True, freq="30min").astype(np.float32)
        series = series[-500:]


        pred_series = model.historical_forecasts(
            series,
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=True,
        )

        print(f"rmse: {rmse(series, pred_series)}.")
        print(f"R2 score: {r2_score(series, pred_series)}.")

        helper.display_forecast(pred_series, series, "1 day", save=True, fig_name=f"{i}-test", model_name=f"{MODEL_NAME}", fig_size=(20,10))


if __name__ == "__main__":
    freeze_support()
    main()
