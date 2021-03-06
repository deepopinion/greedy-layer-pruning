# Greedy-layer-pruning (GLP)

This is the original implementation of GLP [paper](https://arxiv.org/abs/2105.14839).

    @article{peer2022greedy,
      title={Greedy-layer Pruning: Speeding up Transformer Models for Natural Language Processing},
      author={Peer, David and Stabinger, Sebastian and Engl, Stefan and Rodr{\'\i}guez-S{\'a}nchez, Antonio},
      journal={Pattern Recognition Letters},
      year={2022},
      publisher={Elsevier}
    }

Greedy layer pruning (GLP) is introduced to (1) outperform current state-of-the-art for
layer-wise pruning (2) close the performance gap when compared to knowledge distillation,
while (3) using only a modest budget.

The source code contains two main stages: The first (```prune.py```) stage finds the
layers to prune either with GLP or with the optimum strategy as presented
in the paper. This stage writes so-called layer-files (```layer_files/```) which
contain an ordered list of layers that should be pruned for a given model and task.
This file is then used to evaluate the performance of different
methods on the GLUE benchmark (```run_glue.py```) where pruning is executed based
on the order of layers as written in the layer files.
Note: Experimentally we also executed GLP directly on the dev set and included the layers
files in `layer_files_dev_set` for future work.

## Setup
To install all requirements simply call the ```setup.sh``` script which
creates a virutal environment. To run the experiments you therefore have to
enable the environment before starting the experiment.

## Execute and reproduce all experiments
To first prune all models call the ```prune.sh```. This step is optional
as we already deliver the layer-files for all models and tasks.

To reproduce the results of the paper on the GLUE benchmark simply
call the ```run_glue.sh``` script. Please note that guild.ai is used
and all can therefore be evaluated with ```guild compare```. The
hyperparameter setup can be found in ```guild.yml```.
