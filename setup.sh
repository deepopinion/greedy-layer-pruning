#!/bin/bash

python3 -m venv env

source env/bin/activate
pip3 install --upgrade pip

# This depends on the local cuda setup and should be adapted accordingly.
pip3 install torch==1.7.1+cu110 -f https://download.pytorch.org/whl/torch_stable.html
pip3 install -r requirements.txt