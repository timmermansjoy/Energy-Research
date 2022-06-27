from multiprocessing.dummy import freeze_support
import os
import sys

from dotenv import load_dotenv
load_dotenv()
os.environ['WANDB_NOTEBOOK_NAME'] = 'pytorch_stats_own_data.ipynb'
os.environ['WANDB_API_KEY'] = os.getenv('WANDB_API_KEY')

# make is possible to find helper file
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)


from darts import TimeSeries
from darts.models import TFTModel
from darts.dataprocessing.transformers import Scaler, MissingValuesFiller
from darts.utils.likelihood_models import QuantileRegression
from darts.metrics import mape, r2_score, rmse, mse
from darts import TimeSeries

import helper
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch

import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy, ddp2
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

from tqdm.contrib.concurrent import process_map
import tqdm
import argparse

AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()
TRAINING_DATA_PATH = "../../../Data/london_clean/*.csv"
DEFAULT_VALUES = dict(
    hidden_size=90,
    lstm_layers=7,
    num_attention_heads=2,
    dropout=0.1,
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
    covarient = TimeSeries.from_dataframe(df, time_col="DateTime", value_cols=["Temperature_C", "Precip_Rate_mm"], freq="30min", fill_missing_dates=True, fillna_value=True).astype(np.float32)
    list = [series, covarient]
    return list


def main(args):
    MODEL_NAME = args.name
    HOUSEHOLDS = args.households
    TRAIN = args.no_train

    torch.cuda.empty_cache()
    wandb.init(project="Digital-Energy", name=MODEL_NAME, config=DEFAULT_VALUES, resume="allow", group="compare")
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
        patience=3,
        verbose=True,
        mode="min"
        )

    wandb_logger = WandbLogger(project="Digital-Energy", log_model="all", save_top_k=3)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")


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



    if os.path.exists(f"../../../Models/{MODEL_NAME}/checkpoints/"):
        print(f"Loading model {MODEL_NAME}...")
        model = TFTModel.load_from_checkpoint(work_dir="../../../Models/", model_name=f"{MODEL_NAME}", best=True)
    else:
        model = TFTModel(
            input_chunk_length=96,
            output_chunk_length=96,
            hidden_size=config.hidden_size,
            lstm_layers=config.lstm_layers,
            num_attention_heads=config.num_attention_heads,
            dropout=config.dropout,
            batch_size=1024,
            n_epochs=20,
            add_relative_index=False,
            add_encoders=None,
            work_dir="../../../Models",
            save_checkpoints=True,
            model_name=MODEL_NAME,
            pl_trainer_kwargs={
            "enable_progress_bar": True,
            "enable_model_summary": True,
            "accelerator": "gpu",
            "devices": [1],
            #"strategy": DDPStrategy(find_unused_parameters=False),
            "logger": wandb_logger,
            "callbacks": [early_stop_callback, checkpoint_callback],
            },
            random_state=42,
        )

    if TRAIN:
        print("Training model...")
        model.fit(series=train_transformed, val_series=val_transformed, future_covariates=cov_transformed, val_future_covariates=cov_transformed, num_loader_workers=AVAILABLE_CPUS//3, max_samples_per_ts=2000)


    ## ---- EVALUATE ---- ##
    print("Evaluating...")
    helper.eval(model, future_covariates=True, data_normalized=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="the name that the model will have while saving and in wandb")
    parser.add_argument("--no-train", help="makes it so the model doesnt get trained, usefull for checkpoints", action="store_false")
    parser.add_argument("--households", help="the number of households used to train on", type=int, default=1000)
    args = parser.parse_args()

    print(f"Available GPUs: {AVAILABLE_GPUS}")
    print(f"Available CPUs: {AVAILABLE_CPUS}")

    freeze_support()
    main(args)
