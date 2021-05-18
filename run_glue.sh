#!/bin/bash

source env/bin/activate

guild run prune-greedy -y
guild run prune-top -y
guild run distillation -y
guild run baseline -y