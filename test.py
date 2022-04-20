from gc import freeze
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import torch

from pytorch_forecasting import Baseline, NBeats, TimeSeriesDataSet
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import SMAPE

from pytorch_lightning.loggers import WandbLogger
# wandb_logger = WandbLogger(project="Digital-Energy")

import os
AVAILABLE_GPUS = torch.cuda.device_count()
AVAILABLE_CPUS = os.cpu_count()



def main():
    print(f"Available GPUs: {AVAILABLE_GPUS}")
    print(f"Available CPUs: {AVAILABLE_CPUS}")

    data = pd.read_parquet("./Data/ldn_df_small.parquet")



    # create dataset and dataloaders
    max_prediction_length = 48
    max_encoder_length = max_prediction_length * 5

    training_cutoff = 19523 - max_prediction_length

    context_length = max_encoder_length
    prediction_length = max_prediction_length

    training = TimeSeriesDataSet(
        data[lambda x: x.time_idx <= training_cutoff],
        time_idx="time_idx",
        target="KWH",
        categorical_encoders={"LCLid": NaNLabelEncoder().fit(data.LCLid)},
        group_ids=["LCLid"],
        # only unknown variable is "KWH" - and N-Beats can also not take any additional variables
        time_varying_unknown_reals=["KWH"],
        max_encoder_length=context_length,
        max_prediction_length=prediction_length,
        allow_missing_timesteps=True,
    )

    validation = TimeSeriesDataSet.from_dataset(training, data, min_prediction_idx=training_cutoff + 1)
    batch_size = 600
    train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=AVAILABLE_CPUS)
    val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=AVAILABLE_CPUS)






    early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")

    trainer = pl.Trainer(
        max_epochs=100,
        gpus=2,
        strategy=pl.strategies.BaguaStrategy(algorithm="async", sync_interval_ms=100),
        weights_summary="top",
        gradient_clip_val=0.01,
        callbacks=[early_stop_callback],
        limit_train_batches=30,
    )


    net = NBeats.from_dataset(
        training,
        learning_rate= 0.02,
        log_interval=10,
        log_val_interval=1,
        weight_decay=1e-2,
        widths=[32, 512],
        backcast_loss_ratio=1.0
    )

    trainer.fit(
        net,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )





    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = NBeats.load_from_checkpoint(best_model_path)


    actuals = torch.cat([y[0] for x, y in iter(val_dataloader)])
    predictions = best_model.predict(val_dataloader)
    print("score:", (actuals - predictions).abs().mean())


    raw_predictions, x = best_model.predict(val_dataloader, mode="raw", return_x=True)

    print(f"X {x}")
    print(f"Raw predictions {raw_predictions}")








if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    main()
