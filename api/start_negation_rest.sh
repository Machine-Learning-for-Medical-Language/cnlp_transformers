#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
uvicorn negation_rest:app --host 0.0.0.0 --reload

