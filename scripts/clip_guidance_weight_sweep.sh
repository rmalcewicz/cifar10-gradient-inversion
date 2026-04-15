#!/bin/bash

# This script runs the full experiment using Hydra's multirun with the Joblib launcher.

export HUGGINGFACE_HUB_CACHE=~/.cache/huggingface/hub

python -m scripts.run --multirun \
data.class_a=cat \
data.class_b=dog \
data.repetition=1,2,3,4,5 \
data.batch_size=1,2,4,8,16 \
skip.clip=True \
skip.capture=True \
wandb_log=True \
group_name=only_2_check \
reconstruction.oracle_guidance=False \
reconstruction.guidance=False \
reconstruction.clip_coeff=0 \
reconstruction.reruns=5 \
reconstruction.n_dummy=2

#clip_guidance_weight_tuning