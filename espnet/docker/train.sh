#!/bin/bash

./run.sh --docker_gpu 0,1,2,3,4,5,6,7,8,9,10,11 --docker_egs spanish_merge/asr1 \
--ngpu 12 --stage 4 --stop_stage 999
