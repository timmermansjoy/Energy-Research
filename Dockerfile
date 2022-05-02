FROM pytorchlightning/pytorch_lightning

RUN apt update && apt upgrade -y &&\
    apt-get install -y htop neofetch

# RUN apt-get purge *nvidia* && apt autoremove -y &&\
#     apt-get update && \
#     apt install nvidia-driver-510

RUN pip install jupyterlab pytorch-forecasting fastparquet wandb darts plotly python-dotenv
