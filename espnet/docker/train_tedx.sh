#!/bin/bash

./run.sh --docker_gpu 4,5,6,7 --docker_egs spanish_merge/asr1 \
--ngpu 4 --stage 1 --stop_stage 999
