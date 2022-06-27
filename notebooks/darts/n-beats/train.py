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
import tqdm
import argparse

AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()
TRAINING_DATA_PATH = "../../../Data/london_clean/*.csv"
MODEL_NAME = sys.argv[1] if len(sys.argv) > 1 else "generic_nbeats"
DEFAULT_VALUES = dict(
    num_stacks=3,
    num_blocks=8,
    num_layers=2,
    layer_widths=1150,
)

print(f"Available GPUs: {AVAILABLE_GPUS}")
print(f"Available CPUs: {AVAILABLE_CPUS}")


def reader(x):
    return TimeSeries.from_csv(x, time_col='DateTime', value_cols='KWHhh',
                            fill_missing_dates=True, fillna_value=True, freq="30min").astype(np.float32)

def splitter(x):
    train, val = x.split_after(0.85)
    return train, val


def main(args):
    MODEL_NAME = args.name
    HOUSEHOLDS = args.households
    TRAIN = args.no_train


    torch.cuda.empty_cache()

    wandb.init(project="Digital-Energy", name=MODEL_NAME, config=DEFAULT_VALUES, resume="allow", group="compare")
    config = wandb.config



    ## data
    print("Loading data...")
    series_list = []
    # for x in tqdm.tqdm(sorted(glob.glob("../../../Data/london_clean/*.csv"))[:1000]):
    #     df = pd.read_csv(f'{x}')
    #     df["DateTime"] = pd.to_datetime(df['DateTime'])
    #     series = TimeSeries.from_dataframe(df, time_col='DateTime', value_cols='KWHhh').astype(np.float32)
    #     series_list.append(series)

    # done for multiprocessing
    file_list = sorted(glob.glob(TRAINING_DATA_PATH))[:HOUSEHOLDS]
    if file_list == []:
        raise Exception("No files found")
    series_list = process_map(reader, file_list, chunksize=5)


    print("Creating dataset...")
    ## sets
    training_sets = []
    validation_sets = []
    for x in series_list:
        train, val = x.split_after(0.85)
        training_sets.append(train)
        validation_sets.append(val)




    ## ---- MODEL ---- ##

    early_stop_callback = EarlyStopping(
        monitor="val_loss",
        min_delta=0.003,
        patience=4,
        verbose=False,
        mode="min"
        )

    ## check if model already exists

    wandb_logger = WandbLogger(log_model=True)
    checkpoint_callback = ModelCheckpoint(monitor="val_loss", mode="min")

    if os.path.exists(f"../../../Models/{MODEL_NAME}/checkpoints/"):
        print(f"Loading model {MODEL_NAME}...")
        model = NBEATSModel.load_from_checkpoint(work_dir="../../../Models/", model_name=f"{MODEL_NAME}", best=True)
    else:
        print("Creating model...")
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
            model_name = MODEL_NAME,
            save_checkpoints=True,
            pl_trainer_kwargs={
            "enable_progress_bar": True,
            "enable_model_summary": True,
            "accelerator": "gpu",
            "devices": [0], # use all available GPUs (this can be done due to this script not being in a notebook)
            # "strategy": DDPStrategy(find_unused_parameters=False),
            "logger": wandb_logger,
            "callbacks": [early_stop_callback, checkpoint_callback],
            # "profiler":"pytorch",
            }
        )


    ## ---- TRAIN ---- ##
    if TRAIN:
        model.fit(series=training_sets, val_series=validation_sets, num_loader_workers=AVAILABLE_CPUS)


    ## ---- EVALUATE ---- ##

    helper.eval(model, log=True)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--name", help="the name that the model will have while saving and in wandb")
    parser.add_argument("--no-train", help="makes it so the model doesnt get trained, usefull for checkpoints", action="store_false")
    parser.add_argument("--households", help="the number of households used to train on", type=int, default=1000)
    args = parser.parse_args()
    freeze_support()
    main(args)
