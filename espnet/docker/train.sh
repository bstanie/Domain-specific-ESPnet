#!/bin/bash

./run.sh --docker_gpu 0,1,2,3,4,5,6,7 --docker_egs spanish_gong/asr1 \
--ngpu 8 --stage 0 --stop_stage 999
