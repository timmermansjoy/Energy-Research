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
from darts.models import NBEATSModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score, rmse, mse
from darts import TimeSeries

import helper
import glob

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from tqdm.contrib.concurrent import process_map


AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()
TRAINING_DATA_PATH = "../../../Data/london_clean/*.csv"

print(f"Available GPUs: {AVAILABLE_GPUS}")
print(f"Available CPUs: {AVAILABLE_CPUS}")


DEFAULT_VALUES = dict(
    num_stacks=1,
    num_blocks=1,
    num_layers=1,
    layer_widths=512,
)


def reader(x):
    return TimeSeries.from_csv(x, time_col='DateTime', value_cols='KWHhh',
                            fill_missing_dates=True, fillna_value=True, freq="30min").astype(np.float32)

def splitter(x):
    train, val = x.split_after(0.85)
    return train, val


def main():
    torch.cuda.empty_cache()

    wandb.init(project="Digital-Energy", config=DEFAULT_VALUES)
    config = wandb.config



    ## data
    print("Loading data...")
    # series_list = []
    # for x in progressbar.progressbar(sorted(glob.glob("../../../Data/london_clean/*.csv"))[:100]):
    #     df = pd.read_csv(f'{x}')
    #     df["DateTime"] = pd.to_datetime(df['DateTime'])
    #     series = TimeSeries.from_dataframe(df, time_col='DateTime', value_cols='KWHhh').astype(np.float32)
    #     series_list.append(series)

    # done for multiprocessing
    file_list = sorted(glob.glob(TRAINING_DATA_PATH))[:500]
    if file_list == []:
        raise Exception("No files found")
    series_list = process_map(reader, file_list, chunksize=5, max_workers=1)


    print("Creating dataset...")
    ## sets
    training_sets = []
    validation_sets = []
    for x in series_list:
        train, val = x.split_after(0.85)
        training_sets.append(train)
        validation_sets.append(val)

    # create training set validation set with multiprocessing
    # trainx, valx = process_map(lambda x: x.split_after(0.85), series_list, chunksize=5)
    # print("len(train):", len(trainx))
    # print("len(val):", len(valx))

    ## ---- MODEL ---- ##

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=1e-3,
        patience=10,
        verbose=False,
        mode="min"
        )

    ## check if model already exists

    wandb_logger = WandbLogger(project="Digital-Energy", log_model=True)
    # checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")


    model = NBEATSModel(
        input_chunk_length=96,
        output_chunk_length=96,
        generic_architecture=True,
        num_stacks=config.num_stacks,
        num_blocks=config.num_blocks,
        num_layers=config.num_layers,
        layer_widths=config.layer_widths,
        n_epochs=20,
        nr_epochs_val_period=1,
        batch_size=2048,
        work_dir="../../../Models",
        save_checkpoints=False,
        pl_trainer_kwargs={
        "enable_progress_bar": True,
        "enable_model_summary": True,
        "accelerator": "gpu",
        "devices": 2, # use all available GPUs (this can be done due to this script not being in a notebook)
        "strategy": DDPStrategy(find_unused_parameters=False),
        "logger": wandb_logger,
        "callbacks": [early_stop_callback]
        }
    )


    ## ---- TRAIN ---- ##

    model.fit(series=training_sets, val_series=validation_sets, num_loader_workers=4, max_samples_per_ts=100)


    ## ---- EVALUATE ---- ##

        ## test data
    START = 3000
    for i, x in enumerate(sorted(glob.glob(TRAINING_DATA_PATH))[START:START+2]):

        df = pd.read_csv(x)
        df["DateTime"] = pd.to_datetime(df['DateTime'])
        series = TimeSeries.from_dataframe(df, value_cols=['KWHhh'], time_col="DateTime").astype(np.float32)
        series = series[-600:]

        pred_series = model.historical_forecasts(
            series,
            forecast_horizon=1,
            stride=1,
            retrain=False,
            verbose=False,
        )

        fig = helper.display_forecast(pred_series, series, "1 day", save=True, fig_name=f"{i}-test", fig_size=(20,10))

        wandb.log({
            "mape": mape(series, pred_series),
            "mse": mse(series, pred_series),
            "rmse": rmse(series, pred_series),
            "r2": r2_score(series, pred_series),
            "result": wandb.Image(fig)
            })


if __name__ == "__main__":
    freeze_support()
    main()
