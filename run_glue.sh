#!/bin/bash

source env/bin/activate

guild run prune-greedy -y
guild run prune-top -y
guild run distilbert -y
guild run mobilebert -y
guild run baseline -y