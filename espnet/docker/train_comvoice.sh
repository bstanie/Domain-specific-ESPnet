#!/bin/bash

./run.sh --docker_gpu 0,1,2,3 --docker_egs spanish_merge/asr1 \
--ngpu 4 --stage 0 --stop_stage 3