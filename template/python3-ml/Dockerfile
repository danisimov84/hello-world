FROM openfaas/classic-watchdog:0.18.1 as watchdog

FROM glowtools/soccer:latest-2728ca6

WORKDIR /home/app/

COPY index.py           .
COPY requirements.txt   .

WORKDIR /home/app/function/
COPY function/requirements.txt	.


