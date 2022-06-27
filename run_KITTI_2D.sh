#!/bin/bash

conda config --append envs_dirs /srip-vol/parth/myenvs
cd /srip-vol/parth/detr/
python=/srip-vol/parth/myenvs/detr/bin/python

$python main.py

echo "done"

