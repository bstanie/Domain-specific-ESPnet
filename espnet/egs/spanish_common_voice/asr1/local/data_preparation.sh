#!/bin/bash

cd "$(dirname "$0")"
pip install -r requirements.txt
python3 -m utils.prepare_data

chmod -R 777 ../data
chmod -R 777 ../downloads
