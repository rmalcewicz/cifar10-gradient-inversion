#!/bin/bash

# This script runs the full experiment using Hydra's multirun with the Joblib launcher.

export HUGGINGFACE_HUB_CACHE=~/.cache/huggingface/hub

python -m scripts.run --multirun \
data.class_a=airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck \
data.class_b=airplane,automobile,bird,cat,deer,dog,frog,horse,ship,truck \
data.repetition=0,1,2,3,4 \
data.batch_size=1,2,4,8,16