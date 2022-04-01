FROM pytorchlightning/pytorch_lightning


# Expose port 8888 for JupyterLab
EXPOSE 22 8888

RUN apt update && apt install -y htop neofetch && \
    pip install jupyterlab && \
    pip install darts && \
    pip install pytorch-forecasting fastparquet wandb
