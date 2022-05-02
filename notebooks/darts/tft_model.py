from multiprocessing.dummy import freeze_support
import os
from dotenv import load_dotenv
load_dotenv()

os.environ['WANDB_NOTEBOOK_NAME'] = 'pytorch_stats_own_data.ipynb'
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.metrics import mape, r2_score, rmse

from darts import TimeSeries

import helper
import glob

from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
wandb_logger = WandbLogger(project="Digital-Energy")


AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()

print(f"Available GPUs: {AVAILABLE_GPUS}")
print(f"Available CPUs: {AVAILABLE_CPUS}")


def main():
    ## ---- LOAD DATA ---- ##

    ## data
    my_time_series_dataset = []
    for x in sorted(glob.glob("../../Data/london_clean/*.csv"))[:500]:
        series = TimeSeries.from_csv(x, time_col='DateTime', value_cols='KWHhh')
        my_time_series_dataset.append(series)

    ## sets
    training_sets = []
    validation_sets = []
    for x in my_time_series_dataset:
        train, val = series.split_after(0.90)
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



    ## ---- TRAINING ---- ##
    model = TFTModel(
        2,
        1,
    )

if __name__ == "__main__":
    freeze_support()
    main()
